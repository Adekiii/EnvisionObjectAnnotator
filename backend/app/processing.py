import json
import shutil
from pathlib import Path
import threading
import torch
import traceback
import os
import cv2
import numpy as np
import time

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

from .schemas import ProcessingStatus
from .state import state
from .pipeline import (
    setup_device_ultra_optimized,
    configure_torch_ultra_conservative,
    ultra_cleanup_memory,
    UltraOptimizedProcessor,
    get_video_fps,
    get_gpu_memory_info,
)
from .resource_profiler import ResourceProfiler
from .logger import get_logger, get_session_logger

log = get_logger("processing")

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BASE_DIR.parent
SESSIONS_DIR = BASE_DIR / "data" / "sessions"


MOBILESAM_CHECKPOINT = BASE_DIR / "checkpoints" / "mobile_sam.pt"
MODEL_LABEL = "MobileSAM + Cutie"

_CUTIE_CACHE = None  # (cutie_model, device) or None
_MOBILESAM_CACHE = None  # (predictor, device) or None
_CUTIE_LOAD_LOCK = threading.Lock()
_MOBILESAM_LOCK = threading.Lock()


def list_available_models():
    return [
        {
            "key": "mobilesam_cutie",
            "label": MODEL_LABEL,
            "available": MOBILESAM_CHECKPOINT.exists(),
        }
    ]


def _evict_video_predictor_cache():
    """Run GPU memory cleanup WITHOUT dropping the cached Cutie model."""
    ultra_cleanup_memory()


def _evict_image_predictor_cache():
    """Free the cached MobileSAM predictor to reclaim GPU memory."""
    global _MOBILESAM_CACHE
    if _MOBILESAM_CACHE is not None:
        log.info("evicting cached MobileSAM predictor")
        _MOBILESAM_CACHE = None
    ultra_cleanup_memory()


def _get_cutie_model(use_cache=True):
    """Load (and cache) Cutie's model weights. A fresh InferenceCore is
    built from this per propagation pass inside pipeline.py."""
    global _CUTIE_CACHE
    if use_cache and _CUTIE_CACHE is not None:
        log.debug("Cutie model cache hit")
        return _CUTIE_CACHE

    with _CUTIE_LOAD_LOCK:
        if use_cache and _CUTIE_CACHE is not None:  # re-check: lost the race to load
            return _CUTIE_CACHE

        log.info("loading Cutie model")
        from cutie.utils.get_default_model import get_default_model

        configure_torch_ultra_conservative()
        device = setup_device_ultra_optimized()
        cutie_model = get_default_model()

        cutie_model.cfg.use_long_term = True
        log.info(
            "Cutie long-term memory enabled (max_mem_frames=%s, num_prototypes=%s)",
            cutie_model.cfg.long_term.max_mem_frames,
            cutie_model.cfg.long_term.num_prototypes,
        )

        if use_cache:
            _CUTIE_CACHE = (cutie_model, device)
        return cutie_model, device


def _ensure_sam_predictor(use_mps=False):
    """Load (and cache) the MobileSAM predictor used for click-to-mask."""
    global _MOBILESAM_CACHE
    if _MOBILESAM_CACHE is not None:
        return _MOBILESAM_CACHE

    if not MOBILESAM_CHECKPOINT.exists():
        raise FileNotFoundError(f"MobileSAM checkpoint missing: {MOBILESAM_CHECKPOINT}")

    log.info("loading MobileSAM predictor")
    from mobile_sam import SamPredictor, sam_model_registry

    device = setup_device_ultra_optimized()
    if device.type == "mps" and not use_mps:
        device = torch.device("cpu")

    sam = sam_model_registry["vit_t"](checkpoint=str(MOBILESAM_CHECKPOINT))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    _MOBILESAM_CACHE = (predictor, device)
    return predictor, device


def _get_ram_info():
    if psutil is None:
        return None
    mem = psutil.virtual_memory()
    return {
        "used_gb": mem.used / 1024**3,
        "available_gb": mem.available / 1024**3,
        "total_gb": mem.total / 1024**3,
    }


def test_mask_preview(session_id, frame_index, object_name, points, bbox=None):
    """Compute a live mask preview for one frame via MobileSAM."""
    slog = get_session_logger(session_id)
    slog.info("test_mask_preview: frame=%d obj=%s pts=%d", frame_index, object_name, len(points))
    frames_dir = SESSIONS_DIR / session_id / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError("Frames not found")

    coords_list = [[float(p["x"]), float(p["y"])] for p in points]
    labels_list = [int(p["label"]) for p in points]
    if len(coords_list) == 0:
        raise ValueError("No points provided")

    frame_path = frames_dir / f"{frame_index:05d}.jpg"
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise FileNotFoundError("Frame not found")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    coords = np.array(coords_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    multimask = len(coords_list) == 1

    with _MOBILESAM_LOCK:
        predictor, _device = _ensure_sam_predictor()
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(point_coords=coords, point_labels=labels, multimask_output=multimask)
    best = int(np.argmax(scores))
    mask = masks[best].astype(bool)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [c.reshape(-1, 2).tolist() for c in contours if len(c) >= 3]
    h, w = mask.shape

    return {
        "polygons": polygons,
        "width": w,
        "height": h,
        "area_px": int(mask.sum()),
        "score": float(scores[best]),
    }




class HeadlessProcessor:
    def __init__(self, processor):
        self._processor = processor

    def process(self, points_dict, labels_dict, object_names, multiframe_data=None, bboxes_dict=None, progress_callback=None):
        return self._processor.process_video_with_memory_management(
            points_dict,
            labels_dict,
            object_names,
            debug=True,
            multiframe_data=multiframe_data,
            bboxes_dict=bboxes_dict,
            progress_callback=progress_callback,
        )

    def get_partial_results(self):
        return self._processor.get_partial_results()

    def save_video(self, results, output_path, fps, frame_limit=None, progress_callback=None):
        return self._processor.save_results_video_with_enhanced_annotations(
            results,
            output_path,
            fps=fps,
            show_original=False,
            alpha=0.5,
            frame_limit=frame_limit,
            progress_callback=progress_callback,
        )

    def save_elan(self, video_path, output_path, fps):
        return self._processor.create_elan_file(video_path, output_path, fps=fps, frame_offset=0)

    def save_csv(self, results, object_names, output_path, progress_callback=None):
        return self._processor.export_framewise_csv(
            results, object_names, output_path, progress_callback=progress_callback
        )

    def save_masks_json(self, results, object_names, output_path, progress_callback=None):
        return self._processor.export_masks_json(
            results, object_names, output_path, progress_callback=progress_callback
        )

    def cleanup_mask_store(self):
        return self._processor.cleanup_mask_store()


def _set_status(session_id, status, progress, message):
    state.set_processing(
        ProcessingStatus(
            session_id=session_id,
            status=status,
            progress=progress,
            message=message,
        )
    )


def _load_reference_annotations(session_id, reference_frame):
    annotations_dir = SESSIONS_DIR / session_id / "annotations"
    frame_file = annotations_dir / f"frame_{reference_frame:05d}.json"
    if not frame_file.exists():
        raise FileNotFoundError("No annotations found for reference frame")

    data = json.loads(frame_file.read_text())
    objects = data.get("objects", {})
    if not objects:
        raise ValueError("No objects annotated")

    bboxes_data = data.get("bboxes", {})

    points_dict = {}
    labels_dict = {}
    object_names = {}
    bboxes_dict = {}
    for idx, (name, points) in enumerate(objects.items(), start=1):
        if not points:
            continue
        coords = [[p["x"], p["y"]] for p in points]
        labels = [p["label"] for p in points]
        if len(coords) != len(labels) or not coords:
            continue
        object_names[idx] = name
        points_dict[idx] = coords
        labels_dict[idx] = labels
        if name in bboxes_data:
            bb = bboxes_data[name]
            bboxes_dict[idx] = np.array(
                [bb["x1"], bb["y1"], bb["x2"], bb["y2"]], dtype=np.float32
            )

    if not points_dict:
        raise ValueError("No valid points found for reference frame")

    return points_dict, labels_dict, object_names, bboxes_dict


def _load_multiframe_annotations(session_id):
    """Load annotations from all annotated frames.

    Scans all frame_*.json files and builds a consistent object ID mapping
    so the same object name gets the same ID across all frames.

    Returns:
        Tuple of (multiframe_data, object_name_to_id) where multiframe_data
        is {frame_idx: (points_dict, labels_dict, object_names)}.
    """
    annotations_dir = SESSIONS_DIR / session_id / "annotations"
    if not annotations_dir.exists():
        raise FileNotFoundError("No annotations directory found")

    frame_annotations = {}
    frame_bboxes = {}
    global_object_names: set[str] = set()

    for frame_file in sorted(annotations_dir.glob("frame_*.json")):
        try:
            frame_idx = int(frame_file.stem.split("_")[1])
            data = json.loads(frame_file.read_text())
            objects = data.get("objects", {})
            if objects:
                frame_annotations[frame_idx] = objects
                frame_bboxes[frame_idx] = data.get("bboxes", {})
                global_object_names.update(objects.keys())
        except (ValueError, json.JSONDecodeError):
            continue

    if not frame_annotations:
        raise FileNotFoundError("No valid annotations found in any frame")

    # Consistent object IDs across frames
    object_name_to_id = {
        name: idx for idx, name in enumerate(sorted(global_object_names), start=1)
    }

    multiframe_data = {}
    all_bboxes_dict = {}
    for frame_idx, objects in frame_annotations.items():
        points_dict = {}
        labels_dict = {}
        object_names = {}
        bboxes_for_frame = frame_bboxes.get(frame_idx, {})

        for obj_name, points in objects.items():
            if not points:
                continue
            coords = [[p["x"], p["y"]] for p in points]
            labels = [p["label"] for p in points]
            if len(coords) != len(labels) or not coords:
                continue

            obj_id = object_name_to_id[obj_name]
            object_names[obj_id] = obj_name
            points_dict[obj_id] = coords
            labels_dict[obj_id] = labels
            if obj_name in bboxes_for_frame:
                bb = bboxes_for_frame[obj_name]
                all_bboxes_dict[obj_id] = np.array(
                    [bb["x1"], bb["y1"], bb["x2"], bb["y2"]], dtype=np.float32
                )

        if points_dict:
            multiframe_data[frame_idx] = (points_dict, labels_dict, object_names)

    if not multiframe_data:
        raise ValueError("No valid points found in any annotated frame")

    return multiframe_data, object_name_to_id, all_bboxes_dict


def run_processing(session_id):
    slog = get_session_logger(session_id)
    error_log = SESSIONS_DIR / session_id / "processing_error.log"
    debug_log = SESSIONS_DIR / session_id / "processing_debug.log"
    started_at = time.perf_counter()

    def _append_log(line):
        try:
            debug_log.parent.mkdir(parents=True, exist_ok=True)
            with debug_log.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:
            pass

    def _log_debug(message):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        _append_log(f"[{stamp}] {message}")

    def _log_trace():
        trace = traceback.format_exc().rstrip()
        if not trace:
            return
        for line in trace.splitlines():
            _append_log(f"    {line}")

    slog.info("run_processing: starting")
    _log_debug(f"run start: session_id={session_id}")
    try:
        session = state.get_session(session_id)
    except KeyError:
        slog.error("run_processing: session not found")
        _set_status(session_id, "error", 0.0, "Session not found")
        _log_debug("session not found")
        return

    frames_dir = SESSIONS_DIR / session_id / "frames"
    if not frames_dir.exists():
        _set_status(session_id, "error", 0.0, "Frames not found. Extract frames first.")
        _log_debug(f"frames not found: {frames_dir}")
        return

    preview_dir = SESSIONS_DIR / session_id / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / "latest.jpg"

    # Start resource profiler (samples GPU/CPU/RAM every 2 seconds)
    profile_output_dir = (
        Path(session.output_dir) if session.output_dir
        else Path(session.video_path).parent if session.video_path
        else SESSIONS_DIR / session_id
    )
    profiler = ResourceProfiler(
        output_dir=profile_output_dir,
        interval_seconds=2.0,
        session_id=session_id,
    )
    profiler.start()
    _log_debug("resource profiler started")

    def _save_preview(frame):
        try:
            cv2.imwrite(str(preview_path), frame)
        except Exception:
            pass

    config = session.config or {}
    overlap_threshold = float(config.get("overlap_threshold", 0.1))
    auto_tune_info = None  # kept only so the profiling dict below has the key
    fps = None
    if "video_fps" in config:
        try:
            fps = float(config["video_fps"])
        except (TypeError, ValueError):
            pass
    if fps is None and session.video_path:
        try:
            fps, _ = get_video_fps(session.video_path)
        except Exception:
            pass
    if fps is None:
        fps = 30.0
    export_video = bool(config.get("export_video", True))
    export_elan = bool(config.get("export_elan", True))
    export_csv = bool(config.get("export_csv", True))
    reference_frame = int(config.get("reference_frame", 0))
    process_start_frame = config.get("process_start_frame")
    process_end_frame = config.get("process_end_frame")
    try:
        process_start_frame = int(process_start_frame)
    except (TypeError, ValueError):
        process_start_frame = None
    try:
        process_end_frame = int(process_end_frame)
    except (TypeError, ValueError):
        process_end_frame = None
    use_mps = bool(config.get("use_mps", False))
    enable_bidirectional = bool(config.get("enable_bidirectional", True))
    resolved_model_key = "mobilesam_cutie"
    model_label = MODEL_LABEL
    frame_count = len(list(frames_dir.glob("*.jpg")))
    gpu_start = get_gpu_memory_info()
    ram_start = _get_ram_info()
    _log_debug(
        "config: reference_frame=%s frames=%s bidirectional=%s"
        % (reference_frame, frame_count, enable_bidirectional)
    )

    multiframe_data = None
    is_multiframe = False

    try:
        _set_status(session_id, "initializing", 0.05, "Loading annotations")
        mf_data, object_name_to_id, bboxes_dict = _load_multiframe_annotations(session_id)

        # Build global object names registry
        object_names = {obj_id: obj_name for obj_name, obj_id in object_name_to_id.items()}

        if len(mf_data) == 1:
            # Single frame — use traditional approach
            frame_idx = list(mf_data.keys())[0]
            points_dict, labels_dict, _ = mf_data[frame_idx]
            reference_frame = frame_idx
            is_multiframe = False
            _log_debug(
                f"single reference frame: frame={reference_frame} objects={len(object_names)} "
                f"points={sum(len(v) for v in points_dict.values())}"
            )
        else:
            # Multiple frames — use earliest as primary, rest as conditioning
            sorted_frames = sorted(mf_data.keys())
            reference_frame = sorted_frames[0]
            points_dict, labels_dict, _ = mf_data[reference_frame]
            multiframe_data = mf_data
            is_multiframe = True
            _log_debug(
                f"multiframe mode: frames={sorted_frames} total_objects={len(object_names)} "
                f"primary_ref={reference_frame}"
            )
    except Exception as exc:
        _set_status(session_id, "error", 0.05, f"Annotation error: {exc}")
        _log_debug(f"annotation error: {exc}")
        _log_trace()
        return
    _log_debug(
        "annotations loaded: objects=%s points=%s"
        % (len(object_names), sum(len(v) for v in points_dict.values()))
    )
    annotations_done = time.perf_counter()

    try:
        _set_status(session_id, "initializing", 0.1, f"Loading {MODEL_LABEL}")
        os.environ["HYDRA_FULL_ERROR"] = "1"

        # Reuse the cached Cutie model rather than forcing a reload: its
        # get_default_model() initializes hydra's process-wide global state,
        # which cannot be safely re-initialized within the same process.
        cutie_model, device = _get_cutie_model()
        with _MOBILESAM_LOCK:
            sam_predictor, _sam_device = _ensure_sam_predictor(use_mps=use_mps)
        _log_debug("model ready: %s device=%s" % (MODEL_LABEL, device))

        processor = UltraOptimizedProcessor(
            cutie_model,
            sam_predictor,
            str(frames_dir),
            overlap_threshold=overlap_threshold,
            reference_frame=reference_frame,
            preview_callback=_save_preview,
            log_callback=_log_debug,
            preview_stride=max(1, int(frame_count / 200)) if frame_count else 1,
            preview_max_dim=720,
            mask_store_dir=str(SESSIONS_DIR / session_id / "mask_cache"),
            process_start_frame=process_start_frame,
            process_end_frame=process_end_frame,
            enable_bidirectional=enable_bidirectional,
        )
        wrapped = HeadlessProcessor(processor)
    except ImportError as exc:
        error_log.write_text(traceback.format_exc())
        _set_status(session_id, "error", 0.1, f"Missing dependency: {exc}")
        _log_debug(f"missing dependency: {exc}")
        _log_trace()
        return
    except Exception as exc:
        error_log.write_text(traceback.format_exc())
        _set_status(session_id, "error", 0.1, f"Model init error: {exc}")
        _log_debug(f"model init error: {exc}")
        _log_trace()
        return
    model_done = time.perf_counter()

    processing_done = None

    def _write_primary_outputs(results_to_save, suffix="", frame_limit=None, fps_override=None):
        output_dir = Path(session.output_dir) if session.output_dir else Path(session.video_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        file_stem = session_id
        fps = fps_override
        if fps is None:
            fps, _ = get_video_fps(session.video_path)

        outputs = {}
        if export_video:
            video_path = output_dir / f"{file_stem}{suffix}_ANNOTATED.mp4"

            def _save_progress(frame_idx, max_frame):
                total = max(1, int(max_frame) + 1)
                current = min(int(frame_idx) + 1, total)
                progress = 0.85 + 0.15 * (current / float(total))
                _set_status(
                    session_id,
                    "saving",
                    min(progress, 0.99),
                    f"Writing outputs: {current}/{total}",
                )

            wrapped.save_video(
                results_to_save,
                str(video_path),
                fps=fps,
                frame_limit=frame_limit,
                progress_callback=_save_progress,
            )
            outputs["annotated_video"] = str(video_path)
        if export_elan:
            elan_path = output_dir / f"{file_stem}{suffix}_ELAN_TIMELINE.eaf"
            elan_video = str(video_path) if export_video else session.video_path
            wrapped.save_elan(elan_video, str(elan_path), fps=fps)
            if elan_path.exists():
                outputs["elan"] = str(elan_path)
            else:
                _log_debug("ELAN file not created (no targets registered?)")

        return outputs, output_dir, file_stem

    def _update_outputs_meta(csv_status=None, csv_progress=None, csv_error=None, outputs_override=None):
        try:
            session_state = state.get_session(session_id)
        except KeyError:
            return
        config_state = session_state.config or {}
        outputs_state = outputs_override or config_state.get("outputs", {}) or {}
        outputs_meta = config_state.get("outputs_meta", {}) or {}
        if csv_status is not None:
            outputs_meta["csv_status"] = csv_status
        if csv_progress is not None:
            outputs_meta["csv_progress"] = csv_progress
        if csv_error is not None:
            outputs_meta["csv_error"] = csv_error
        updated_config = {**config_state, "outputs": outputs_state, "outputs_meta": outputs_meta}
        state.update_session(session_id, config=updated_config)

    def _export_masks(results_to_save, outputs, output_dir, file_stem, suffix=""):
        """Export segmentation masks as JSON before mask store cleanup."""
        try:
            masks_path = output_dir / f"{file_stem}{suffix}_MASKS.json"
            wrapped.save_masks_json(results_to_save, object_names, str(masks_path))
            outputs["masks_json"] = str(masks_path)
        except Exception as exc:
            _log_debug(f"masks json export error: {exc}")
            _log_trace()

    def _start_csv_export(results_to_save, outputs, output_dir, file_stem, suffix=""):
        if not export_csv:
            _update_outputs_meta(csv_status="disabled", csv_progress=0.0, outputs_override=outputs)
            _export_masks(results_to_save, outputs, output_dir, file_stem, suffix)
            try:
                wrapped.cleanup_mask_store()
            except Exception:
                pass
            return

        _update_outputs_meta(csv_status="pending", csv_progress=0.0, outputs_override=outputs)

        def _csv_worker():
            _update_outputs_meta(csv_status="running", csv_progress=0.0)
            try:
                csv_path = output_dir / f"{file_stem}{suffix}_FRAME_BY_FRAME.csv"

                def _csv_progress(done, total):
                    total = max(1, int(total))
                    current = min(int(done), total)
                    progress = current / float(total)
                    _update_outputs_meta(csv_status="running", csv_progress=progress)

                wrapped.save_csv(
                    results_to_save,
                    object_names,
                    str(csv_path),
                    progress_callback=_csv_progress,
                )
                _export_masks(results_to_save, outputs, output_dir, file_stem, suffix)
                outputs_done = {**outputs, "csv": str(csv_path)}
                _update_outputs_meta(
                    csv_status="completed",
                    csv_progress=1.0,
                    outputs_override=outputs_done,
                )
                try:
                    wrapped.cleanup_mask_store()
                except Exception:
                    pass
            except Exception as exc:
                _log_debug(f"csv export error: {exc}")
                _log_trace()
                _update_outputs_meta(csv_status="error", csv_error=str(exc))

        worker = threading.Thread(target=_csv_worker, daemon=True)
        worker.start()

    try:
        _set_status(session_id, "processing", 0.2, "Processing frames")

        def _frame_progress(current, total):
            pct = 0.2 + 0.65 * (current / max(total, 1))
            _set_status(session_id, "processing", min(pct, 0.85), f"Processing frames: {current}/{total}")

        mf_data = multiframe_data if is_multiframe else None
        results = wrapped.process(points_dict, labels_dict, object_names, multiframe_data=mf_data, bboxes_dict=bboxes_dict, progress_callback=_frame_progress)
        if not results:
            _set_status(session_id, "error", 0.6, "Processing failed")
            _log_debug("processing failed: no results")
            return
    except Exception as exc:
        slog.error("run_processing: failed — %s", exc, exc_info=True)
        error_log.write_text(traceback.format_exc())
        _log_debug(f"processing error: {exc}")
        _log_trace()

        partial_results = {}
        try:
            partial_results = wrapped.get_partial_results() or {}
        except Exception:
            partial_results = {}

        if partial_results:
            _set_status(session_id, "saving", 0.85, "Writing partial outputs")
            try:
                frame_limit = max(partial_results.keys()) if partial_results else None
                outputs, output_dir, file_stem = _write_primary_outputs(
                    partial_results,
                    suffix="_PARTIAL",
                    frame_limit=frame_limit,
                    fps_override=fps,
                )
                failed_at = time.perf_counter()
                processing_seconds = max(failed_at - model_done, 1e-6)
                profiling = {
                    "model_key": resolved_model_key,
                    "model_label": model_label,
                    "device": str(device),
                    "frames_total": frame_count,
                    "frames_processed": len(partial_results),
                    "objects_total": len(object_names),
                    "processing_fps": len(partial_results) / processing_seconds,
                    "auto_tune": auto_tune_info,
                    "partial": True,
                    "error": str(exc),
                    "timings_s": {
                        "load_annotations": annotations_done - started_at,
                        "model_init": model_done - annotations_done,
                        "processing": failed_at - model_done,
                        "total": failed_at - started_at,
                    },
                    "gpu_mem_start": gpu_start,
                    "gpu_mem_end": get_gpu_memory_info(),
                    "ram_start": ram_start,
                    "ram_end": _get_ram_info(),
                }
                updated_config = {
                    **session.config,
                    "outputs": outputs,
                    "profiling": profiling,
                }
                state.update_session(session_id, config=updated_config)
                _start_csv_export(
                    partial_results,
                    outputs,
                    output_dir,
                    file_stem,
                    suffix="_PARTIAL",
                )
                _set_status(session_id, "error", 0.9, "Processing failed; partial outputs saved")
            except Exception as save_exc:
                error_log.write_text(traceback.format_exc())
                _set_status(session_id, "error", 0.9, f"Output error: {save_exc}")
                _log_debug(f"output error: {save_exc}")
                _log_trace()
        else:
            _set_status(session_id, "error", 0.6, f"Processing error: {exc}")
        return
    processing_done = time.perf_counter()

    try:
        _set_status(session_id, "saving", 0.85, "Writing outputs")
        outputs, output_dir, file_stem = _write_primary_outputs(results, fps_override=fps)

        finished_at = time.perf_counter()
        processing_seconds = max(processing_done - model_done, 1e-6)
        profiling = {
            "model_key": resolved_model_key,
            "model_label": model_label,
            "device": str(device),
            "frames_total": frame_count,
            "objects_total": len(object_names),
            "processing_fps": frame_count / processing_seconds,
            "auto_tune": auto_tune_info,
            "timings_s": {
                "load_annotations": annotations_done - started_at,
                "model_init": model_done - annotations_done,
                "processing": processing_done - model_done,
                "saving": finished_at - processing_done,
                "total": finished_at - started_at,
            },
            "gpu_mem_start": gpu_start,
            "gpu_mem_end": get_gpu_memory_info(),
            "ram_start": ram_start,
            "ram_end": _get_ram_info(),
        }
        # Stop profiler early so we can include the path in outputs
        try:
            profile_html = profiler.stop()
            outputs["resource_profile"] = str(profile_html)
            profiling["resource_profile"] = str(profile_html)
        except Exception:
            pass
        updated_config = {**session.config, "outputs": outputs, "profiling": profiling}
        state.update_session(session_id, config=updated_config)
        _start_csv_export(results, outputs, output_dir, file_stem)
        message = "Processing complete"
        if export_csv:
            message = "Video ready. CSV exporting..."
        _set_status(session_id, "completed", 1.0, message)
        slog.info("run_processing: completed — %d frames", frame_count)
        _log_debug("processing complete")

        try:
            removed_count = 0
            if frames_dir.exists():
                frame_files = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
                for f in frame_files:
                    f.unlink()
                removed_count = len(frame_files)
            session_dir = frames_dir.parent
            for d in session_dir.iterdir():
                if d.is_dir() and d.name not in ("frames", "thumbs", "mask_cache"):
                    shutil.rmtree(d, ignore_errors=True)
            if removed_count:
                slog.info("run_processing: cleaned up %d frames", removed_count)
                _log_debug(f"frame cleanup: removed {removed_count} files + temp dirs")
        except Exception as cleanup_exc:
            slog.warning("run_processing: frame cleanup failed — %s", cleanup_exc)
    except Exception as exc:
        slog.error("run_processing: output error — %s", exc, exc_info=True)
        error_log.write_text(traceback.format_exc())
        _set_status(session_id, "error", 0.9, f"Output error: {exc}")
        _log_debug(f"output error: {exc}")
        _log_trace()
    finally:
        # Stop resource profiler and save results
        try:
            profile_html = profiler.stop()
            _log_debug(f"resource profile saved: {profile_html}")
        except Exception as prof_exc:
            _log_debug(f"profiler stop error: {prof_exc}")
        # Free GPU memory so the next session starts clean
        try:
            _evict_video_predictor_cache()
            _evict_image_predictor_cache()
            slog.info("run_processing: GPU cleanup completed")
            _log_debug("GPU cleanup completed")
        except Exception as gpu_exc:
            slog.warning("run_processing: GPU cleanup failed — %s", gpu_exc)
        state.clear_thread(session_id)


def start_background_job(session_id):
    worker = threading.Thread(target=run_processing, args=(session_id,), daemon=True)
    state.set_thread(session_id, worker)
    worker.start()
    return worker
