import os
import json
import shutil
import contextlib
from pathlib import Path
from datetime import datetime
import traceback

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

RAM_PRESSURE_THRESHOLD = 80


def _format_tensor_info(value):
    if value is None:
        return "none"
    if torch.is_tensor(value):
        try:
            return (
                f"shape={list(value.shape)} dtype={value.dtype} device={value.device} "
                f"contig={value.is_contiguous()} stride={list(value.stride())} "
                f"storage_offset={value.storage_offset()}"
            )
        except Exception as exc:
            return f"tensor<unavailable> error={exc}"
    if isinstance(value, np.ndarray):
        return f"ndarray shape={list(value.shape)} dtype={value.dtype}"
    return f"{type(value)}"


def _format_points_info(points):
    if points is None:
        return "none"
    try:
        arr = np.asarray(points)
        if arr.size == 0:
            return "empty"
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        return f"shape={list(arr.shape)} min={mins.tolist()} max={maxs.tolist()}"
    except Exception as exc:
        return f"unavailable error={exc}"


def get_gpu_memory_info():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "free_gb": total - reserved,
            "utilization_pct": (reserved / total) * 100,
        }
    return None


def get_system_memory_info():
    if psutil is None:
        return None
    try:
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / 1024**3,
            "available_gb": mem.available / 1024**3,
            "used_gb": mem.used / 1024**3,
            "percent_used": mem.percent,
        }
    except Exception:
        return None


def check_memory_pressure(threshold_pct=None):
    if threshold_pct is None:
        threshold_pct = RAM_PRESSURE_THRESHOLD
    mem_info = get_system_memory_info()
    if mem_info and mem_info["percent_used"] >= threshold_pct:
        return True, mem_info
    return False, mem_info


def ultra_cleanup_memory():
    """Memory cleanup."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def configure_torch_ultra_conservative():
    """Basic torch setup. Cutie/MobileSAM are both lightweight (a few hundred
    MB of VRAM) compared to SAM2's video predictor, so the aggressive
    per-process memory-fraction capping and forced autocast context that used
    to live here aren't needed -- autocast is applied per-call instead, where
    the actual propagation happens."""
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def setup_device_ultra_optimized():
    """Setup computation device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.set_default_dtype(torch.float32)
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device


class DiskBackedMaskStore:
    """Store masks on disk with a small in-memory cache.

    Used to bound RAM on long videos instead of holding every frame's masks
    in memory. This used to write one np.savez_compressed .npz file PER
    OBJECT PER FRAME -- for a several-thousand-frame video with a handful of
    objects and bidirectional propagation that's tens of thousands of
    synchronous, zlib-compressed writes inside the main per-frame tracking
    loop, which starves the GPU (observed ~15-20% utilization on an 8GB card
    that should be much busier -- the compression, not the model, was the
    bottleneck). A frame's object masks are mutually exclusive per pixel
    anyway (Cutie's own output is a single label mask, not independent
    per-object masks), so instead we combine them into one uint8 label
    array (0 = background, else object id) and write ONE uncompressed .npy
    per frame -- one fifth as many writes for a 4-object video, and no
    compression overhead at all.
    """

    def __init__(self, base_dir, max_in_memory=50):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index = {}
        self._in_memory = {}
        self._max_in_memory = max_in_memory

    def _frame_path(self, frame_idx):
        return self.base_dir / f"frame_{int(frame_idx):06d}.npy"

    def store(self, frame_idx, frame_results):
        frame_idx = int(frame_idx)
        obj_ids = []
        combined = None
        claim_count = None
        for obj_id, mask in frame_results.items():
            if mask is None:
                continue
            obj_id = int(obj_id)
            mask_bool = np.asarray(mask, dtype=bool)
            if combined is None:
                combined = np.zeros(mask_bool.shape, dtype=np.uint8)
                claim_count = np.zeros(mask_bool.shape, dtype=np.uint8)
            combined[mask_bool] = obj_id
            claim_count[mask_bool] += 1
            obj_ids.append(obj_id)
        cleaned = {}
        if combined is not None:
            # Two of the incoming masks can legitimately disagree on a
            # pixel (e.g. the forward and backward propagation passes
            # merged at this frame) even though each is individually
            # exclusive. Collapsing that to whichever object happened to be
            # assigned last would silently bake one object's pixels into
            # another's stored mask -- leave genuinely ambiguous pixels as
            # background instead, same policy as the seed-mask overlap fix.
            combined[claim_count > 1] = 0
            np.save(self._frame_path(frame_idx), combined)
            # Cache the SAME deduplicated view we just wrote, not the raw
            # (possibly overlapping) input -- otherwise a load() that hits
            # the in-memory cache would disagree with one that reads back
            # from disk.
            cleaned = {obj_id: combined == obj_id for obj_id in obj_ids}
        self._index[frame_idx] = obj_ids
        self._in_memory[frame_idx] = cleaned
        if len(self._in_memory) > self._max_in_memory:
            oldest = min(self._in_memory.keys())
            del self._in_memory[oldest]
        return obj_ids

    def load(self, frame_idx):
        frame_idx = int(frame_idx)
        if frame_idx in self._in_memory:
            return self._in_memory[frame_idx]
        obj_ids = self._index.get(frame_idx)
        if not obj_ids:
            return {}
        path = self._frame_path(frame_idx)
        if not path.exists():
            return {}
        combined = np.load(path)
        return {obj_id: combined == obj_id for obj_id in obj_ids}

    def has_frame(self, frame_idx):
        return int(frame_idx) in self._index

    def frame_indices(self):
        return sorted(self._index.keys())

    def cleanup(self):
        try:
            shutil.rmtree(self.base_dir)
        except Exception:
            pass

def get_video_fps(video_path):
    """Get video FPS using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


class EnhancedOverlapDetector:
    """Enhanced overlap detector that properly handles inclusion and complex overlaps."""

    def __init__(self, overlap_threshold=0.1):
        self.overlap_threshold = overlap_threshold
        self.inclusion_threshold = 0.1

    def calculate_detailed_overlap(self, mask1, mask2):
        """Enhanced overlap detection with both pixel overlap and spatial containment."""
        if mask1.shape != mask2.shape:
            print(f"Shape mismatch: {mask1.shape} vs {mask2.shape}")
            return None

        if len(mask1.shape) > 2:
            mask1 = mask1.squeeze()
        if len(mask2.shape) > 2:
            mask2 = mask2.squeeze()

        if mask1.dtype in [np.float32, np.float64]:
            mask1_bool = mask1 > 0.0001
        else:
            mask1_bool = mask1 > 0

        if mask2.dtype in [np.float32, np.float64]:
            mask2_bool = mask2 > 0.0001
        else:
            mask2_bool = mask2 > 0

        area1 = np.sum(mask1_bool)
        area2 = np.sum(mask2_bool)

        if area1 == 0 or area2 == 0:
            return None

        intersection = mask1_bool & mask2_bool
        intersection_area = np.sum(intersection)

        overlap_pct_1 = intersection_area / area1 if area1 > 0 else 0
        overlap_pct_2 = intersection_area / area2 if area2 > 0 else 0
        max_overlap = max(overlap_pct_1, overlap_pct_2)

        spatial_relationship = False
        containment_type = None

        try:
            contours1, _ = cv2.findContours(mask1_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours2, _ = cv2.findContours(mask2_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours1 and contours2:
                contour1 = max(contours1, key=cv2.contourArea)
                contour2 = max(contours2, key=cv2.contourArea)

                M1 = cv2.moments(contour1)
                M2 = cv2.moments(contour2)

                if M1["m00"] != 0 and M2["m00"] != 0:
                    cx1, cy1 = int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"])
                    cx2, cy2 = int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"])

                    mask1_contour = np.zeros_like(mask1_bool, dtype=np.uint8)
                    mask2_contour = np.zeros_like(mask2_bool, dtype=np.uint8)
                    cv2.fillPoly(mask1_contour, [contour1], 1)
                    cv2.fillPoly(mask2_contour, [contour2], 1)

                    object2_in_contour1 = np.any(mask2_bool & mask1_contour)
                    object1_in_contour2 = np.any(mask1_bool & mask2_contour)

                    if object2_in_contour1:
                        spatial_relationship = True
                        containment_type = "object2_partial_inside_object1"
                    elif object1_in_contour2:
                        spatial_relationship = True
                        containment_type = "object1_partial_inside_object2"
                    else:
                        inside_1 = cv2.pointPolygonTest(contour1, (cx2, cy2), False) >= 0
                        inside_2 = cv2.pointPolygonTest(contour2, (cx1, cy1), False) >= 0

                        if inside_1:
                            spatial_relationship = True
                            containment_type = "object2_centroid_inside_object1"
                        elif inside_2:
                            spatial_relationship = True
                            containment_type = "object1_centroid_inside_object2"
        except Exception as exc:
            print(f"    Error in spatial containment detection: {exc}")
            spatial_relationship = False

        has_meaningful_pixel_overlap = intersection_area > 0 and max_overlap >= self.overlap_threshold
        has_spatial_relationship = spatial_relationship

        if not has_meaningful_pixel_overlap and not has_spatial_relationship:
            return None

        meets_basic_threshold = has_meaningful_pixel_overlap or has_spatial_relationship
        meets_continuation_threshold = has_meaningful_pixel_overlap or has_spatial_relationship

        if has_meaningful_pixel_overlap and has_spatial_relationship:
            relationship_type = "overlap_and_containment"
        elif has_meaningful_pixel_overlap:
            relationship_type = "pixel_overlap"
        elif has_spatial_relationship:
            relationship_type = "spatial_containment"
        else:
            relationship_type = "none"

        return {
            "intersection_area": intersection_area,
            "overlap_pct_1": overlap_pct_1,
            "overlap_pct_2": overlap_pct_2,
            "min_overlap_pct": min(overlap_pct_1, overlap_pct_2),
            "max_overlap_pct": max_overlap,
            "spatial_relationship": spatial_relationship,
            "containment_type": containment_type,
            "has_meaningful_pixel_overlap": has_meaningful_pixel_overlap,
            "has_spatial_relationship": has_spatial_relationship,
            "relationship_type": relationship_type,
            "meets_threshold": meets_basic_threshold,
            "meets_continuation_threshold": meets_continuation_threshold,
        }


class ImprovedTargetOverlapTracker:
    """Improved overlap tracker with better inclusion detection and annotations."""

    def __init__(self, overlap_threshold=0.1):
        self.overlap_threshold = overlap_threshold
        self.overlap_events = {}
        self.target_objects = {}
        self.detector = EnhancedOverlapDetector(overlap_threshold)

    def register_target(self, obj_id, obj_name):
        """Register target objects."""
        if "target" in obj_name.lower():
            self.target_objects[obj_id] = obj_name
            self.overlap_events[obj_id] = []
            print(f"Target registered: {obj_name} (ID: {obj_id})")
            return True
        return False

    def get_overlap_summary(self):
        """Get overlap summary."""
        summary = {}
        for target_id, events in self.overlap_events.items():
            target_name = self.target_objects[target_id]
            summary[target_name] = {
                "total_events": len(events),
                "events": events,
                "total_overlap_frames": sum(event["duration_frames"] for event in events),
            }
        return summary

    def finalize_tracking(self, last_frame_idx):
        """Finalize events without extending duration."""
        for target_id, events in self.overlap_events.items():
            if events and events[-1].get("end_frame") is None:
                last_event = events[-1]
                if "duration_frames" in last_event and last_event["duration_frames"] > 0:
                    last_event["end_frame"] = last_event["start_frame"] + last_event["duration_frames"] - 1
                else:
                    last_event["end_frame"] = last_event["start_frame"]
                    last_event["duration_frames"] = 1

    def analyze_frame_overlaps(self, frame_results, object_names):
        """Enhanced frame analysis with continuation validation."""
        frame_analysis = {
            "target_overlaps": {},
            "object_relationships": {},
            "looking_at_events": [],
        }

        try:
            for target_id in self.target_objects:
                if target_id not in frame_results:
                    continue

                target_mask = frame_results[target_id]
                if len(target_mask.shape) > 2:
                    target_mask = target_mask.squeeze()

                target_name = self.target_objects[target_id]
                looking_at_objects = []

                has_ongoing_event = (
                    self.overlap_events[target_id] and not self.overlap_events[target_id][-1].get("end_frame")
                )

                for obj_id, mask in frame_results.items():
                    if obj_id == target_id:
                        continue

                    if len(mask.shape) > 2:
                        mask = mask.squeeze()

                    obj_name = object_names.get(obj_id, f"Object_{obj_id}")

                    try:
                        overlap_info = self.detector.calculate_detailed_overlap(target_mask, mask)
                        if overlap_info:
                            if has_ongoing_event:
                                if overlap_info.get("meets_continuation_threshold", False):
                                    looking_at_objects.append(
                                        {
                                            "object_id": obj_id,
                                            "object_name": obj_name,
                                            "overlap_pct": overlap_info.get("max_overlap_pct", 0),
                                            "event_type": "looking_at",
                                            "relationship_desc": f"OVERLAPS {obj_name} (continuing)",
                                        }
                                    )
                            else:
                                if overlap_info.get("meets_threshold", False):
                                    looking_at_objects.append(
                                        {
                                            "object_id": obj_id,
                                            "object_name": obj_name,
                                            "overlap_pct": overlap_info.get("max_overlap_pct", 0),
                                            "event_type": "looking_at",
                                            "relationship_desc": f"LOOKING AT {obj_name}",
                                        }
                                    )

                            if looking_at_objects and looking_at_objects[-1]["object_id"] == obj_id:
                                frame_analysis["looking_at_events"].append(
                                    {
                                        "target_id": target_id,
                                        "target_name": target_name,
                                        "object_id": obj_id,
                                        "object_name": obj_name,
                                    }
                                )
                    except Exception as exc:
                        print(f"      Error checking {obj_name}: {exc}")
                        continue

                if looking_at_objects:
                    frame_analysis["target_overlaps"][target_id] = looking_at_objects
        except Exception as exc:
            print(f"  Error in analyze_frame_overlaps: {exc}")
        return frame_analysis

    def track_frame_overlaps_batch(self, frame_idx, frame_results, object_names):
        """Track 'looking at' events with accurate offset detection."""
        try:
            frame_analysis = self.analyze_frame_overlaps(frame_results, object_names)

            if not hasattr(self, "frame_analyses"):
                self.frame_analyses = {}
            self.frame_analyses[frame_idx] = frame_analysis

            for target_id in self.target_objects:
                current_overlaps = []
                if target_id in frame_analysis.get("target_overlaps", {}):
                    looking_at_objects = frame_analysis["target_overlaps"][target_id]
                    current_overlaps = [
                        (obj["object_name"], obj.get("overlap_pct", 0))
                        for obj in looking_at_objects
                    ]

                self._update_overlap_event(target_id, frame_idx, current_overlaps)

            return frame_analysis
        except Exception as exc:
            print(f"  Error in track_frame_overlaps_batch for frame {frame_idx}: {exc}")
            return {
                "target_overlaps": {},
                "object_relationships": {},
                "looking_at_events": [],
            }

    def _update_overlap_event(self, target_id, frame_idx, overlapping_tuples):
        """Update overlap events. overlapping_tuples is list of (name, pct)."""
        events = self.overlap_events[target_id]
        current_names = set(name for name, _ in overlapping_tuples)
        overlap_pcts = {name: pct for name, pct in overlapping_tuples}

        if events and events[-1].get("end_frame") is None:
            last_event = events[-1]
            last_overlap_set = set(last_event["overlapping_objects"])

            if current_names == last_overlap_set and current_names:
                last_event["duration_frames"] = frame_idx - last_event["start_frame"] + 1
                # Update running average overlap percentages
                n = last_event["duration_frames"]
                for name, pct in overlap_pcts.items():
                    prev = last_event.get("overlap_pcts", {}).get(name, pct)
                    last_event.setdefault("overlap_pcts", {})[name] = prev + (pct - prev) / n
            else:
                last_event["end_frame"] = frame_idx - 1
                last_event["duration_frames"] = last_event["end_frame"] - last_event["start_frame"] + 1

                if current_names:
                    new_event = {
                        "start_frame": frame_idx,
                        "end_frame": None,
                        "duration_frames": 1,
                        "overlapping_objects": [name for name, _ in overlapping_tuples],
                        "overlap_pcts": overlap_pcts,
                        "event_type": "looking_at",
                    }
                    events.append(new_event)
        else:
            if current_names:
                new_event = {
                    "start_frame": frame_idx,
                    "end_frame": None,
                    "duration_frames": 1,
                    "overlapping_objects": [name for name, _ in overlapping_tuples],
                    "overlap_pcts": overlap_pcts,
                    "event_type": "looking_at",
                }
                events.append(new_event)

    def has_targets(self):
        return bool(self.target_objects)


class UltraOptimizedProcessor:
    """Ultra memory-optimized processor with improved overlap detection."""

    def __init__(
        self,
        cutie_model,
        sam_predictor,
        video_dir,
        overlap_threshold=0.1,
        reference_frame=0,
        preview_callback=None,
        log_callback=None,
        preview_stride=15,
        preview_max_dim=720,
        process_start_frame=None,
        process_end_frame=None,
        mask_store_dir=None,
        disk_store_max_in_memory=50,
        ram_pressure_threshold=RAM_PRESSURE_THRESHOLD,
        disk_store_enabled=True,
        enable_bidirectional=True,
        max_internal_size=480,
    ):
        self.cutie_model = cutie_model
        self.sam_predictor = sam_predictor
        self.max_internal_size = max_internal_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.full_video_dir = video_dir
        self.video_dir = video_dir
        self.overlap_threshold = overlap_threshold
        self.reference_frame = reference_frame
        self.reference_frame_full = reference_frame
        self.preview_callback = preview_callback
        self.log_callback = log_callback
        self.preview_stride = max(1, int(preview_stride)) if preview_stride else None
        self.preview_max_dim = int(preview_max_dim)
        self.process_start_frame_full = process_start_frame
        self.process_end_frame_full = process_end_frame
        self.process_start_frame = None
        self.process_end_frame = None
        self.mask_store_dir = mask_store_dir
        self.disk_store_max_in_memory = disk_store_max_in_memory
        self.ram_pressure_threshold = ram_pressure_threshold
        self.disk_store_enabled = disk_store_enabled
        self.enable_bidirectional = enable_bidirectional
        self._mask_store = None
        self._disk_store_active = False
        self.frame_index_map = None
        self.roi_info = None  # kept only so downstream export code's getattr(self, "roi_info", None) no-ops
        self._prepared = False

        self.overlap_tracker = ImprovedTargetOverlapTracker(overlap_threshold)
        self.partial_results = {}

        self.full_frame_names = sorted(
            [
                p for p in os.listdir(self.full_video_dir)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
            ],
            key=lambda p: int(os.path.splitext(p)[0]),
        )

        if not self.full_frame_names:
            raise ValueError("No frames found in the specified directory!")
        self.frame_names = list(self.full_frame_names)

        try:
            _cmap = plt.get_cmap("tab10")
            self._preview_colors = [tuple(int(c * 255) for c in _cmap(i)[:3][::-1]) for i in range(10)]
        except Exception:
            self._preview_colors = [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255),
                (128, 128, 0),
                (128, 0, 128),
                (0, 128, 128),
                (200, 200, 200),
            ]

    def _compress_mask(self, mask):
        if mask is None:
            return None
        if mask is None:
            return None
        if isinstance(mask, dict) and mask.get("_packed"):
            return mask
        m = mask[0] if hasattr(mask, "shape") and len(mask.shape) == 3 else mask
        m = np.asarray(m).astype(np.uint8)
        flat = m.reshape(-1)
        packed = np.packbits(flat)
        return {
            "_packed": True,
            "shape": tuple(m.shape),
            "size": int(m.size),
            "data": packed,
        }

    def _decompress_mask(self, mask):
        if not (isinstance(mask, dict) and mask.get("_packed")):
            return mask
        shape = tuple(mask.get("shape") or ())
        size = int(mask.get("size") or 0)
        data = mask.get("data")
        if data is None or not shape or size <= 0:
            return None
        flat = np.unpackbits(data)[:size]
        return flat.reshape(shape).astype(bool)

    def _expand_roi_mask(self, mask):
        if mask is None or not self.roi_info:
            return mask
        x0 = int(self.roi_info["x0"])
        y0 = int(self.roi_info["y0"])
        x1 = int(self.roi_info["x1"])
        y1 = int(self.roi_info["y1"])
        full_w = int(self.roi_info["full_width"])
        full_h = int(self.roi_info["full_height"])
        roi_w = x1 - x0 + 1
        roi_h = y1 - y0 + 1

        if roi_w <= 0 or roi_h <= 0 or full_w <= 0 or full_h <= 0:
            return mask

        if mask.shape != (roi_h, roi_w) and mask.shape[0] > 0 and mask.shape[1] > 0:
            try:
                mask = cv2.resize(mask.astype(np.float32), (roi_w, roi_h), interpolation=cv2.INTER_LINEAR) > 0.5
            except Exception:
                return mask

        full_mask = np.zeros((full_h, full_w), dtype=bool)
        full_mask[y0 : y1 + 1, x0 : x1 + 1] = mask
        return full_mask

    def _log(self, message):
        cb = getattr(self, "log_callback", None)
        if cb is None:
            return
        try:
            cb(message)
        except Exception:
            pass

    def cleanup_mask_store(self):
        if self._mask_store is None:
            return
        try:
            self._mask_store.cleanup()
        except Exception:
            pass
        self._mask_store = None
        self._disk_store_active = False

    def _normalize_frame_results(self, frame_results):
        normalized = {}
        if not frame_results:
            return normalized
        for obj_id, mask in frame_results.items():
            if mask is None:
                continue
            m = self._decompress_mask(mask)
            if m is None:
                continue
            if hasattr(m, "shape") and len(m.shape) == 3:
                m = m[0]
            normalized[obj_id] = m.astype(bool)
        return normalized

    def _ensure_disk_store(self):
        if self._mask_store is not None or not self.disk_store_enabled:
            return self._mask_store
        base_dir = self.mask_store_dir
        if not base_dir:
            base_dir = os.path.join(os.path.dirname(self.full_video_dir), "mask_cache")
        self._mask_store = DiskBackedMaskStore(base_dir, max_in_memory=self.disk_store_max_in_memory)
        self._disk_store_active = True
        return self._mask_store

    def _enable_disk_store(self, results, reason=None):
        store = self._ensure_disk_store()
        if store is None:
            return False
        if reason:
            self._log(f"disk store enabled: {reason}")
        # Offload any in-memory results to disk
        for frame_idx, frame_results in list(results.items()):
            if not frame_results:
                continue
            if isinstance(frame_results, dict) and frame_results.get("_disk_only"):
                continue
            normalized = self._normalize_frame_results(frame_results)
            if normalized:
                store.store(frame_idx, normalized)
                results[frame_idx] = {"_disk_only": True}
        ultra_cleanup_memory()
        return True

    def _store_frame_results(self, results, frame_idx, frame_results):
        if self._mask_store is not None:
            self._mask_store.store(frame_idx, frame_results)
            results[frame_idx] = {"_disk_only": True}
            return
        results[frame_idx] = frame_results

    def _get_frame_results(self, results, frame_idx):
        frame_results = results.get(frame_idx)
        if self._mask_store is not None:
            if not frame_results or (isinstance(frame_results, dict) and frame_results.get("_disk_only")):
                return self._mask_store.load(frame_idx)
        return frame_results or {}

    def get_partial_results(self):
        return self.partial_results or {}

    def process_video_with_memory_management(self, points_dict, labels_dict, object_names, debug=True, multiframe_data=None, bboxes_dict=None, progress_callback=None):
        """Track every annotated object through the whole video with Cutie."""
        self.partial_results = {}
        self._progress_callback = progress_callback
        try:
            if debug:
                self._log(
                    "processing start (Cutie): frames=%s objects=%s bidirectional=%s"
                    % (len(self.frame_names), len(object_names), self.enable_bidirectional)
                )
            return self._run_cutie_tracking(points_dict, labels_dict, object_names, multiframe_data, debug)
        except Exception as exc:
            if debug:
                self._log("processing failed: %s" % exc)
                self._log(traceback.format_exc())
            raise
        finally:
            ultra_cleanup_memory()

    def _frame_to_tensor(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float() / 255.0

    def _seed_mask_from_points(self, frame_bgr, points, labels):
        """Run MobileSAM on one frame to turn stored click points into a mask."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(rgb)
        coords = np.array(points, dtype=np.float32)
        labs = np.array(labels, dtype=np.int32)
        multimask = len(points) == 1
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=coords, point_labels=labs, multimask_output=multimask
        )
        best = int(np.argmax(scores))
        return masks[best].astype(bool)

    def _build_seed_masks(self, points_dict, labels_dict, object_names, multiframe_data):
        """Turn stored click points into boolean seed masks, per frame."""
        if multiframe_data:
            frames_points = {fidx: pts for fidx, (pts, labs, _names) in multiframe_data.items()}
            frames_labels = {fidx: labs for fidx, (pts, labs, _names) in multiframe_data.items()}
        else:
            frames_points = {self.reference_frame: points_dict}
            frames_labels = {self.reference_frame: labels_dict}

        first_seen_frame = {}
        for frame_idx, pts_for_frame in frames_points.items():
            for obj_id in pts_for_frame:
                if obj_id not in first_seen_frame or frame_idx < first_seen_frame[obj_id]:
                    first_seen_frame[obj_id] = frame_idx

        all_seeds_by_frame = {}
        first_seed_by_frame = {}
        for frame_idx in sorted(frames_points.keys()):
            pts_for_frame = frames_points[frame_idx]
            labs_for_frame = frames_labels[frame_idx]
            frame_path = os.path.join(self.full_video_dir, self.full_frame_names[frame_idx])
            frame_bgr = cv2.imread(frame_path)
            if frame_bgr is None:
                continue
            for obj_id, pts in pts_for_frame.items():
                if not pts:
                    continue
                labs = labs_for_frame.get(obj_id)
                mask_bool = self._seed_mask_from_points(frame_bgr, pts, labs)
                all_seeds_by_frame.setdefault(frame_idx, []).append((obj_id, mask_bool))
                if first_seen_frame.get(obj_id) == frame_idx:
                    first_seed_by_frame.setdefault(frame_idx, []).append((obj_id, mask_bool))

        return all_seeds_by_frame, first_seed_by_frame

    def _run_cutie_tracking(self, points_dict, labels_dict, object_names, multiframe_data, debug):
        from cutie.inference.inference_core import InferenceCore

        self.object_names = object_names

        for obj_id, obj_name in object_names.items():
            self.overlap_tracker.register_target(obj_id, obj_name)

        all_seeds_by_frame, first_seed_by_frame = self._build_seed_masks(
            points_dict, labels_dict, object_names, multiframe_data
        )
        if not all_seeds_by_frame:
            raise ValueError("No valid seed points to track")

        total_frames = len(self.frame_names)
        last_frame = total_frames - 1
        if self.process_end_frame_full is not None:
            last_frame = min(last_frame, int(self.process_end_frame_full))

        global_start = min(all_seeds_by_frame.keys())
        if self.process_start_frame_full is not None:
            global_start = max(global_start, int(self.process_start_frame_full))
        max_first_seed = max(first_seed_by_frame.keys())
        run_backward = self.enable_bidirectional and max_first_seed > 0

        forward_total = max(0, last_frame - global_start + 1)
        backward_total = (max_first_seed + 1) if run_backward else 0
        grand_total = max(1, forward_total + backward_total)

        results = {}
        self.partial_results = results
        frame_analyses = {}

        mem_info = get_system_memory_info()
        if self.disk_store_enabled:
            if total_frames >= 2000:
                self._enable_disk_store(results, reason="long video")
            elif mem_info and mem_info.get("percent_used", 0) >= self.ram_pressure_threshold:
                self._enable_disk_store(results, reason=f"ram usage {mem_info.get('percent_used', 0):.0f}%")

        done_so_far = 0
        progress_cb = getattr(self, "_progress_callback", None)

        def _report(done, total):
            if progress_cb and (done % 20 == 0 or done == total):
                try:
                    progress_cb(done_so_far + done, grand_total)
                except Exception:
                    pass

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )

        forward_processor = InferenceCore(self.cutie_model, cfg=self.cutie_model.cfg)
        forward_processor.max_internal_size = self.max_internal_size
        with torch.inference_mode(), autocast_ctx:
            self._propagate_cutie(
                forward_processor,
                range(global_start, last_frame + 1),
                all_seeds_by_frame,
                object_names,
                results,
                frame_analyses,
                merge_mode="overwrite",
                progress_cb=_report,
            )
        done_so_far = forward_total

        if run_backward:
            backward_processor = InferenceCore(self.cutie_model, cfg=self.cutie_model.cfg)
            backward_processor.max_internal_size = self.max_internal_size
            with torch.inference_mode(), autocast_ctx:
                self._propagate_cutie(
                    backward_processor,
                    range(max_first_seed, -1, -1),
                    first_seed_by_frame,
                    object_names,
                    results,
                    frame_analyses,
                    merge_mode="fill_missing",
                    progress_cb=_report,
                )

        self.overlap_tracker.finalize_tracking(last_frame)
        self.frame_analyses = frame_analyses
        return results

    def _propagate_cutie(self, processor, frame_range, seeds_by_frame, object_names, results, frame_analyses, merge_mode, progress_cb):
        """Propagates Cutie using the "frame_range" (ascending=forward, descending=backward)"""
        targets_found = self.overlap_tracker.has_targets()
        total = len(frame_range)
        for done, frame_idx in enumerate(frame_range, start=1):
            frame_path = os.path.join(self.full_video_dir, self.full_frame_names[frame_idx])
            frame_bgr = cv2.imread(frame_path)
            if frame_bgr is None:
                break
            image = self._frame_to_tensor(frame_bgr)

            seeds_here = seeds_by_frame.get(frame_idx)
            if seeds_here:
                h, w = frame_bgr.shape[:2]
                combined = np.zeros((h, w), dtype=np.uint8)
                claim_count = np.zeros((h, w), dtype=np.uint8)
                obj_ids = []
                for obj_id, mask_bool in seeds_here:
                    combined[mask_bool] = obj_id
                    claim_count[mask_bool] += 1
                    obj_ids.append(obj_id)
                combined[claim_count > 1] = 0

                already_known = set(processor.object_manager.all_obj_ids)
                reintroducing = [oid for oid in obj_ids if oid in already_known]
                if reintroducing:
                    processor.delete_objects(reintroducing)

                mask_tensor = torch.from_numpy(combined).to(self.device)
                output_prob = processor.step(image, mask_tensor, objects=sorted(obj_ids))
            else:
                output_prob = processor.step(image)

            pred_mask = processor.output_prob_to_mask(output_prob).cpu().numpy().astype(np.uint8)
            present_ids = [oid for oid in np.unique(pred_mask).tolist() if oid != 0]
            frame_results = {oid: (pred_mask == oid) for oid in present_ids}

            if merge_mode == "fill_missing":
                existing = self._get_frame_results(results, frame_idx)
                merged = dict(existing)
                for oid, mask in frame_results.items():
                    merged.setdefault(oid, mask)
                frame_results = merged

            self._store_frame_results(results, frame_idx, frame_results)

            frame_analysis = None
            if targets_found:
                frame_analysis = self.overlap_tracker.track_frame_overlaps_batch(frame_idx, frame_results, object_names)
                frame_analyses[frame_idx] = frame_analysis
            self._maybe_emit_preview(frame_idx, frame_results, frame_analysis)

            progress_cb(done, total)

    def _maybe_emit_preview(self, frame_idx, frame_results, frame_analysis):
        cb = getattr(self, "preview_callback", None)
        stride = getattr(self, "preview_stride", None)
        if cb is None or stride is None:
            return
        if frame_idx % max(1, int(stride)) != 0:
            return

        try:
            frame_path = os.path.join(self.video_dir, self.frame_names[frame_idx])
            frame = cv2.imread(frame_path)
            if frame is None:
                return
            H, W = frame.shape[:2]
            overlay = frame.copy()

            colors = getattr(self, "_preview_colors", [(0, 255, 0)])
            for i, (obj_id, mask) in enumerate(frame_results.items()):
                if mask is None:
                    continue
                m = mask[0] if hasattr(mask, "shape") and len(mask.shape) == 3 else mask
                if m is None:
                    continue

                if getattr(m, "shape", None) != (H, W):
                    try:
                        m = cv2.resize(m.astype(float), (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
                    except Exception:
                        continue
                else:
                    m = m.astype(float) > 0.5

                color = colors[i % len(colors)]
                idx = m.astype(bool)
                if np.any(idx):
                    overlay[idx] = (0.6 * np.array(color, dtype=np.float32) + 0.4 * overlay[idx]).astype(np.uint8)

            text = f"Frame {frame_idx}"
            try:
                if frame_analysis and frame_analysis.get("target_overlaps"):
                    text += " | overlaps"
            except Exception:
                pass
            cv2.putText(overlay, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            max_side = int(getattr(self, "preview_max_dim", 720))
            if max(H, W) > max_side and max_side > 0:
                scale = max_side / max(H, W)
                overlay = cv2.resize(overlay, (int(W * scale), int(H * scale)))

            cb(overlay)
        except Exception as exc:
            print(f"[preview] skipped: {exc}")

    def save_results_video_with_enhanced_annotations(
        self,
        results,
        output_path,
        fps=30,
        show_original=False,
        alpha=0.5,
        frame_limit=None,
        progress_callback=None,
    ):
        """Save results video with enhanced visual feedback for looking-at events."""
        if not results:
            print("No results to save!")
            return

        frame_dir = self.full_video_dir if self.full_video_dir else self.video_dir
        frame_names = self.full_frame_names if self.full_frame_names else self.frame_names
        first_frame = cv2.imread(os.path.join(frame_dir, frame_names[0]))
        height, width = first_frame.shape[:2]

        out_width = width * 2 if show_original else width
        out = None
        # Try H.264 first, mp4v (old-style software MPEG-4) used to be
        # tried first and almost always "succeeds" (opens fine), so H264/
        # avc1 never got a chance even when available.
        for codec in ("avc1", "H264", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (int(out_width), int(height)))
            if out.isOpened():
                print(f"VideoWriter initialized with codec {codec}")
                break
        if out is None or not out.isOpened():
            raise RuntimeError("Failed to initialize video writer (check codec support).")

        cmap = plt.get_cmap("tab10")
        overlap_frame_count = 0
        last_target_states = {}

        max_frame = len(frame_names) - 1
        if frame_limit is not None:
            try:
                frame_limit = int(frame_limit)
            except (TypeError, ValueError):
                frame_limit = None
        if frame_limit is not None:
            max_frame = min(max_frame, max(0, frame_limit))

        for frame_idx in tqdm(range(max_frame + 1), desc="Saving frames"):
            frame = cv2.imread(os.path.join(frame_dir, frame_names[frame_idx]))
            if frame is None:
                continue

            overlay = frame.copy()
            frame_analysis = None
            if hasattr(self, "frame_analyses") and frame_idx in self.frame_analyses:
                frame_analysis = self.frame_analyses[frame_idx]

            has_looking_at_events = (
                frame_analysis
                and frame_analysis.get("target_overlaps")
                and any(frame_analysis["target_overlaps"].values())
            )
            if has_looking_at_events:
                overlap_frame_count += 1

            # Read once -- with the disk-backed mask store this re-reads and
            # reconstructs the frame's masks from disk, so calling it twice
            # per frame (once here, once below) doubled that cost on every
            # frame that has any target-overlap data, which is most/all
            # frames whenever any object is marked as a target.
            frame_results = self._get_frame_results(results, frame_idx)

            looking_at_info = {}
            if frame_analysis:
                for obj_id in frame_results:
                    looking_at_info[obj_id] = {
                        "is_target": False,
                        "looking_at": [],
                        "looked_at_by": [],
                        "is_being_looked_at": False,
                    }

                for target_id, looking_at_objects in frame_analysis.get("target_overlaps", {}).items():
                    if target_id in looking_at_info:
                        looking_at_info[target_id]["is_target"] = True
                        looking_at_info[target_id]["looking_at"] = [
                            obj["object_name"] for obj in looking_at_objects
                        ]

                    for obj_info in looking_at_objects:
                        obj_id = obj_info["object_id"]
                        if obj_id in looking_at_info:
                            target_name = self.overlap_tracker.target_objects.get(target_id, f"Target_{target_id}")
                            looking_at_info[obj_id]["looked_at_by"].append(target_name)
                            looking_at_info[obj_id]["is_being_looked_at"] = True

            if frame_results:
                for obj_id, mask in frame_results.items():
                    mask = self._decompress_mask(mask)
                    mask = self._expand_roi_mask(mask)
                    if mask is None:
                        continue
                    if len(mask.shape) == 3:
                        mask = mask[0]

                    if mask.shape != (height, width) and mask.shape[0] > 0 and mask.shape[1] > 0:
                        try:
                            mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR) > 0.5
                        except cv2.error:
                            continue

                    if mask.shape == (height, width):
                        mask = mask.astype(bool)
                        obj_info = looking_at_info.get(obj_id, {})
                        is_target = obj_info.get("is_target", False)
                        is_being_looked_at = obj_info.get("is_being_looked_at", False)
                        looking_at = obj_info.get("looking_at", [])
                        looked_at_by = obj_info.get("looked_at_by", [])

                        base_color = np.array(cmap(obj_id % 10)[:3]) * 255
                        if is_target and looking_at:
                            color = np.minimum(base_color + [100, 100, 0], 255)
                            border_color = (0, 255, 255)
                            border_thickness = 8
                        elif is_being_looked_at:
                            color = np.minimum(base_color + [120, 0, 0], 255)
                            border_color = (0, 0, 255)
                            border_thickness = 8
                        else:
                            color = base_color
                            border_color = (255, 255, 255)
                            border_thickness = 2

                        # Blend only inside the mask to avoid darkening the whole frame
                        for c in range(3):
                            channel = overlay[:, :, c]
                            channel[mask] = (
                                channel[mask] * (1 - alpha) + color[c] * alpha
                            ).astype(np.uint8)

                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(overlay, contours, -1, border_color, border_thickness)

                        obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
                        if is_target and looking_at:
                            if len(looking_at) == 1:
                                status_text = f"{obj_name} -> OVERLAPS {looking_at[0]}"
                            else:
                                status_text = f"{obj_name} -> OVERLAPS {len(looking_at)} OBJECTS"
                        elif is_being_looked_at:
                            continue
                        else:
                            status_text = obj_name
                        if is_target and looking_at:
                            pass
                        elif is_being_looked_at:
                            pass
                        # place label near mask centroid (fallback to top-left stack)
                        ys, xs = np.where(mask)
                        if len(xs) > 0:
                            cx, cy = int(xs.mean()), int(ys.mean())
                        else:
                            cx, cy = 10, 30 + (obj_id * 25) % max(25, height - 30)
                        font_scale = 0.6 if len(status_text) > 30 else 0.7
                        (tw, th), baseline = cv2.getTextSize(
                            status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
                        )
                        x = max(0, min(cx - tw // 2, width - tw - 2))
                        y = max(th + 4, min(cy, height - 4))
                        if is_target and looking_at:
                            bg_color = (0, 100, 200)
                            text_color = (0, 255, 255)
                            padding = 10
                        elif is_being_looked_at:
                            bg_color = (0, 0, 200)
                            text_color = (255, 255, 255)
                            padding = 10
                        else:
                            bg_color = (0, 0, 0)
                            text_color = (255, 255, 255)
                            padding = 5
                        cv2.rectangle(
                            overlay,
                            (x - padding, y - th - padding),
                            (x + tw + padding, y + baseline + padding),
                            bg_color,
                            -1,
                        )
                        cv2.putText(
                            overlay,
                            status_text,
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            text_color,
                            2,
                        )

            status_messages = []
            event_messages = []
            if frame_analysis:
                target_overlaps = frame_analysis.get("target_overlaps", {}) or {}
                if target_overlaps and any(target_overlaps.values()):
                    target_events = []
                    for target_id, looking_at_objects in target_overlaps.items():
                        target_name = self.overlap_tracker.target_objects.get(
                            target_id, f"Target_{target_id}"
                        )
                        object_names = [obj["object_name"] for obj in looking_at_objects]
                        if len(object_names) == 1:
                            target_events.append(f"{target_name} -> {object_names[0]}")
                        else:
                            target_events.append(f"{target_name} -> {len(object_names)} objects")
                    if target_events:
                        status_messages.append(f"LOOKING AT DETECTED: {'; '.join(target_events)}")

                for target_id, target_name in self.overlap_tracker.target_objects.items():
                    current = set(
                        obj["object_name"] for obj in (target_overlaps.get(target_id) or [])
                    )
                    last = last_target_states.get(target_id, set())
                    if current and current == last:
                        event_messages.append(
                            f"EVENT CONTINUES: {target_name} -> {', '.join(sorted(current))}"
                        )
                    elif current and not last:
                        event_messages.append(
                            f"EVENT STARTED: {target_name} -> {', '.join(sorted(current))}"
                        )
                    elif not current and last:
                        event_messages.append(
                            f"EVENT ENDED: {target_name} -> {', '.join(sorted(last))}"
                        )
                    elif current and last and current != last:
                        event_messages.append(
                            f"EVENT ENDED: {target_name} -> {', '.join(sorted(last))}"
                        )
                        event_messages.append(
                            f"EVENT STARTED: {target_name} -> {', '.join(sorted(current))}"
                        )
                    last_target_states[target_id] = current

            info_y = 30
            for message in status_messages:
                text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(
                    overlay,
                    (5, info_y - 25),
                    (text_size[0] + 15, info_y + 10),
                    (0, 0, 180),
                    -1,
                )
                cv2.putText(
                    overlay,
                    message,
                    (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                info_y += 35

            for message in event_messages:
                text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(
                    overlay,
                    (5, info_y - 22),
                    (text_size[0] + 15, info_y + 8),
                    (0, 100, 200),
                    -1,
                )
                cv2.putText(
                    overlay,
                    message,
                    (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                info_y += 28

            if show_original:
                output_frame = np.concatenate([frame, overlay], axis=1)
            else:
                output_frame = overlay

            out.write(output_frame)
            if progress_callback and (frame_idx % 50 == 0 or frame_idx == max_frame):
                try:
                    progress_callback(frame_idx, max_frame)
                except Exception:
                    pass

        out.release()

    def create_elan_file(self, video_path, output_path, fps, frame_offset=0):
        """Create ELAN file with corrected timing alignment."""
        if not self.overlap_tracker.has_targets():
            print("No targets found - skipping ELAN export")
            return

        # FPS is now computed as effective_fps (extracted_frames / duration)
        # upstream, so no need to re-read from the original video file.

        summary = self.overlap_tracker.get_overlap_summary()

        header = f'''<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="SAM2_Looking_At_Events" DATE="{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}" FORMAT="3.0" VERSION="3.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv3.0.xsd">
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">
        <MEDIA_DESCRIPTOR MEDIA_URL="file://{os.path.abspath(video_path)}"
            MIME_TYPE="video/mp4" RELATIVE_MEDIA_URL="{os.path.basename(video_path)}"/>
        <PROPERTY NAME="lastUsedAnnotationId">0</PROPERTY>
    </HEADER>
    <TIME_ORDER>
'''

        time_slots = []
        time_slot_id = 1
        time_slot_refs = {}

        all_time_points = set()
        for target_name, target_data in summary.items():
            for event in sorted(target_data["events"], key=lambda e: e["start_frame"]):
                start_frame_corrected = event["start_frame"] + frame_offset
                end_frame_corrected = event["end_frame"] + frame_offset

                start_time = start_frame_corrected / fps
                end_time = end_frame_corrected / fps

                all_time_points.add(start_time)
                all_time_points.add(end_time)

        for time_point in sorted(all_time_points):
            time_ms = int(time_point * 1000)
            time_slots.append(f'        <TIME_SLOT TIME_SLOT_ID="ts{time_slot_id}" TIME_VALUE="{time_ms}"/>')
            time_slot_refs[time_ms] = f"ts{time_slot_id}"
            time_slot_id += 1

        header += "\n".join(time_slots) + "\n    </TIME_ORDER>\n"

        tier_content = ""
        annotation_id = 1

        for target_name, target_data in summary.items():
            tier_id = target_name.upper().replace(" ", "_").replace("-", "_")
            tier_content += f'    <TIER DEFAULT_LOCALE="en" LINGUISTIC_TYPE_REF="default" TIER_ID="{tier_id}_LOOKING_AT">\n'

            for event in sorted(target_data["events"], key=lambda e: e["start_frame"]):
                start_frame_corrected = event["start_frame"] + frame_offset
                end_frame_corrected = event["end_frame"] + frame_offset

                start_time = start_frame_corrected / fps
                end_time = end_frame_corrected / fps
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)

                start_slot = time_slot_refs[start_ms]
                end_slot = time_slot_refs[end_ms]

                overlap_pcts = event.get("overlap_pcts", {})
                parts = []
                for obj_name in event["overlapping_objects"]:
                    pct = overlap_pcts.get(obj_name, 0)
                    parts.append(f"{obj_name} ({pct * 100:.0f}%)")
                annotation_value = f"{target_name} overlapped by {', '.join(parts)}"

                annotation = f'''        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="a{annotation_id}" TIME_SLOT_REF1="{start_slot}" TIME_SLOT_REF2="{end_slot}">
                <ANNOTATION_VALUE>{annotation_value}</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>'''

                tier_content += annotation + "\n"
                annotation_id += 1

            tier_content += "    </TIER>\n"

        footer = '''    <LINGUISTIC_TYPE GRAPHIC_REFERENCES="false" LINGUISTIC_TYPE_ID="default" TIME_ALIGNABLE="true"/>
    <LOCALE LANGUAGE_CODE="en"/>
    <CONSTRAINT DESCRIPTION="Time subdivision of parent annotation's time interval, no time gaps allowed within this interval" STEREOTYPE="Time_Subdivision"/>
    <CONSTRAINT DESCRIPTION="Symbolic subdivision of a parent annotation. Annotations cannot be time-aligned" STEREOTYPE="Symbolic_Subdivision"/>
    <CONSTRAINT DESCRIPTION="1-1 association with a parent annotation" STEREOTYPE="Symbolic_Association"/>
    <CONSTRAINT DESCRIPTION="Time alignable annotations within the parent annotation's time interval, gaps are allowed" STEREOTYPE="Included_In"/>
</ANNOTATION_DOCUMENT>'''

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header + tier_content + footer)

    def export_framewise_csv(self, results, object_names, csv_path, progress_callback=None):
        import csv

        if not results:
            print("No results to export to CSV.")
            return

        analyses = getattr(self, "frame_analyses", {}) or {}
        target_ids = set(
            getattr(self, "overlap_tracker", None).target_objects.keys()
            if getattr(self, "overlap_tracker", None)
            else []
        )

        fieldnames = [
            "frame_idx",
            "obj_id",
            "obj_name",
            "is_target",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "centroid_x",
            "centroid_y",
            "area_px",
            "overlapped_by_targets",
            "target_looking_at",
            "looking_at_count",
        ]

        roi_info = getattr(self, "roi_info", None)
        roi_x0 = int(roi_info["x0"]) if roi_info else 0
        roi_y0 = int(roi_info["y0"]) if roi_info else 0
        roi_w = int(roi_info["x1"] - roi_info["x0"] + 1) if roi_info else None
        roi_h = int(roi_info["y1"] - roi_info["y0"] + 1) if roi_info else None

        def _mask_stats(m):
            try:
                if roi_info and m is not None and roi_w and roi_h and m.shape != (roi_h, roi_w):
                    try:
                        m = cv2.resize(m.astype(np.float32), (roi_w, roi_h), interpolation=cv2.INTER_LINEAR) > 0.5
                    except Exception:
                        pass
                m = m.astype(np.uint8)
                ys, xs = np.where(m > 0)
                if ys.size == 0:
                    return (None,) * 7
                x0, y0 = int(xs.min()), int(ys.min())
                x1, y1 = int(xs.max()), int(ys.max())
                w, h = (x1 - x0 + 1), (y1 - y0 + 1)
                cx = float(xs.mean())
                cy = float(ys.mean())
                area = int(ys.size)
                if roi_info:
                    x0 += roi_x0
                    y0 += roi_y0
                    cx += roi_x0
                    cy += roi_y0
                return x0, y0, w, h, cx, cy, area
            except Exception:
                return (None,) * 7

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            total_frames = len(results)
            processed_frames = 0
            for frame_idx in sorted(results.keys()):
                frame_results = self._get_frame_results(results, frame_idx)
                frame_analysis = analyses.get(frame_idx, {}) or {}
                target_overlaps = frame_analysis.get("target_overlaps", {}) or {}

                overlapped_by = {}
                for t_id, objs in target_overlaps.items():
                    t_name = object_names.get(t_id, f"Object_{t_id}")
                    for entry in objs or []:
                        oid = entry.get("object_id")
                        if oid is None:
                            continue
                        overlapped_by.setdefault(oid, []).append(t_name)

                for obj_id, mask in frame_results.items():
                    obj_name = object_names.get(obj_id, f"Object_{obj_id}")
                    mask = self._decompress_mask(mask)
                    if mask is None:
                        continue
                    if hasattr(mask, "shape") and len(mask.shape) == 3:
                        mask = mask.squeeze()

                    bx, by, bw, bh, cx, cy, area = _mask_stats(mask)

                    is_target = obj_id in target_ids

                    looking_at = []
                    if is_target and obj_id in target_overlaps:
                        looking_at = [
                            e.get("object_name", f"Object_{e.get('object_id')}")
                            for e in (target_overlaps.get(obj_id) or [])
                        ]

                    row = dict(
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        obj_name=obj_name,
                        is_target=bool(is_target),
                        bbox_x=bx,
                        bbox_y=by,
                        bbox_w=bw,
                        bbox_h=bh,
                        centroid_x=cx,
                        centroid_y=cy,
                        area_px=area,
                        overlapped_by_targets=";".join(overlapped_by.get(obj_id, [])),
                        target_looking_at=";".join(looking_at) if is_target else "",
                        looking_at_count=(len(looking_at) if is_target else 0),
                    )
                    writer.writerow(row)
                processed_frames += 1
                if progress_callback and (
                    processed_frames % 50 == 0 or processed_frames == total_frames
                ):
                    try:
                        progress_callback(processed_frames, total_frames)
                    except Exception:
                        pass

        print(f"CSV exported: {csv_path}")

    @staticmethod
    def _rle_encode(mask):
        """Run-length encode a binary mask (COCO-style)."""
        flat = mask.astype(np.uint8).ravel(order="F")
        diffs = np.diff(flat, prepend=0, append=0)
        change_indices = np.where(diffs != 0)[0]
        counts = np.diff(np.concatenate(([0], change_indices, [len(flat)])))
        # If mask starts with 1, prepend a 0-length run of 0s
        if flat[0] == 1:
            counts = np.concatenate(([0], counts))
        return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts.tolist()}

    def export_masks_json(self, results, object_names, output_path, progress_callback=None):
        """Export segmentation masks as JSON with RLE encoding."""
        import json as _json

        if not results:
            print("No results to export masks.")
            return

        roi_info = getattr(self, "roi_info", None)
        roi_x0 = int(roi_info["x0"]) if roi_info else 0
        roi_y0 = int(roi_info["y0"]) if roi_info else 0
        roi_w = int(roi_info["x1"] - roi_info["x0"] + 1) if roi_info else None
        roi_h = int(roi_info["y1"] - roi_info["y0"] + 1) if roi_info else None

        frames_data = {}
        total_frames = len(results)
        processed = 0

        for frame_idx in sorted(results.keys()):
            frame_results = self._get_frame_results(results, frame_idx)
            frame_objects = {}

            for obj_id, mask in frame_results.items():
                mask = self._decompress_mask(mask)
                if mask is None:
                    continue
                if hasattr(mask, "shape") and len(mask.shape) == 3:
                    mask = mask.squeeze()

                # Resize to ROI dimensions if needed
                if roi_info and mask is not None and roi_w and roi_h and mask.shape != (roi_h, roi_w):
                    try:
                        mask = cv2.resize(mask.astype(np.float32), (roi_w, roi_h), interpolation=cv2.INTER_LINEAR) > 0.5
                    except Exception:
                        pass

                mask = mask.astype(np.uint8)
                ys, xs = np.where(mask > 0)

                if ys.size == 0:
                    continue

                x0, y0 = int(xs.min()), int(ys.min())
                x1, y1 = int(xs.max()), int(ys.max())
                w, h = (x1 - x0 + 1), (y1 - y0 + 1)
                cx = float(xs.mean())
                cy = float(ys.mean())
                area = int(ys.size)

                if roi_info:
                    x0 += roi_x0
                    y0 += roi_y0
                    cx += roi_x0
                    cy += roi_y0

                obj_name = object_names.get(obj_id, f"Object_{obj_id}")
                frame_objects[str(obj_id)] = {
                    "name": obj_name,
                    "bbox": [x0, y0, w, h],
                    "centroid": [round(cx, 2), round(cy, 2)],
                    "area": area,
                    "mask_rle": self._rle_encode(mask),
                }

            if frame_objects:
                frames_data[str(frame_idx)] = frame_objects

            processed += 1
            if progress_callback and (processed % 100 == 0 or processed == total_frames):
                try:
                    progress_callback(processed, total_frames)
                except Exception:
                    pass

        output = {
            "object_names": {str(k): v for k, v in object_names.items()},
            "total_frames": total_frames,
            "frames": frames_data,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            _json.dump(output, f)

        print(f"Masks JSON exported: {output_path}")
