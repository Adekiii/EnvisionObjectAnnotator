from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from uuid import uuid4
import subprocess
import json
import cv2
import math
import mimetypes

import numpy as np
import tempfile
import shutil

from .schemas import (
    SessionCreate,
    Session,
    ConfigUpdate,
    AnnotationPayload,
    ProcessingStartRequest,
    ProcessingStatus,
    ResultsResponse,
    FrameExtractionRequest,
    FrameListResponse,
    SampleClipRequest,
    DetectGreyRequest,
    DetectGreyResponse,
    FrameSuggestionResponse,
)
from .state import state
from .processing import start_background_job, test_mask_preview, list_available_models
from .frame_analysis import suggest_optimal_frames
from . import interaction_log as ilog


def _parse_ffprobe_rate(value):
    value = (value or "").strip()
    if not value:
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            num_f = float(num)
            den_f = float(den)
        except ValueError:
            return None
        if den_f == 0:
            return None
        return num_f / den_f
    try:
        return float(value)
    except ValueError:
        return None


def _probe_video_fps(video_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        fps = _parse_ffprobe_rate(line)
        if fps and fps > 0 and math.isfinite(fps):
            return fps
    return None


def _probe_video_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return None
    try:
        duration = float(result.stdout.strip())
        if duration > 0 and math.isfinite(duration):
            return duration
    except (ValueError, TypeError):
        pass
    return None

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
SESSIONS_DIR = BASE_DIR / "data" / "sessions"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="EnvisionObjectAnnotator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sessions/{session_id}/interaction-log")
def get_interaction_log(session_id: str):
    """Get the interaction log for a session."""
    try:
        state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"session_id": session_id, "events": ilog.get_session_log(session_id)}


@app.get("/diagnostics")
def diagnostics():
    """Run system diagnostics to verify setup is correct."""
    import sys
    import platform

    results = {
        "python": {
            "version": sys.version,
            "ok": sys.version_info >= (3, 10),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "pytorch": {"ok": False, "version": None, "cuda_available": False, "cuda_version": None, "gpu_name": None},
        "ffmpeg": {"ok": False, "version": None},
        "models": {"ok": False, "available": [], "count": 0},
        "disk": {"ok": True, "sessions_dir_exists": False},
    }

    # Check PyTorch
    try:
        import torch
        results["pytorch"]["version"] = torch.__version__
        results["pytorch"]["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            results["pytorch"]["cuda_version"] = torch.version.cuda
            results["pytorch"]["gpu_name"] = torch.cuda.get_device_name(0)
            results["pytorch"]["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        results["pytorch"]["ok"] = True
    except Exception as e:
        results["pytorch"]["error"] = str(e)

    # Check ffmpeg
    try:
        proc = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            first_line = proc.stdout.split("\n")[0] if proc.stdout else ""
            results["ffmpeg"]["version"] = first_line
            results["ffmpeg"]["ok"] = True
    except Exception as e:
        results["ffmpeg"]["error"] = str(e)

    # Check models
    try:
        models = list_available_models()
        available = [m for m in models if m["available"]]
        results["models"]["available"] = [m["label"] for m in available]
        results["models"]["count"] = len(available)
        results["models"]["ok"] = len(available) > 0
    except Exception as e:
        results["models"]["error"] = str(e)

    # Check sessions directory
    results["disk"]["sessions_dir_exists"] = SESSIONS_DIR.exists()

    # Overall status
    results["all_ok"] = all([
        results["python"]["ok"],
        results["pytorch"]["ok"],
        results["ffmpeg"]["ok"],
        results["models"]["ok"],
    ])

    return results


@app.get("/system/stats")
def system_stats():
    """Get current system resource usage (GPU, CPU, RAM)."""
    stats = {
        "gpu": None,
        "cpu": None,
        "ram": None,
    }

    # GPU stats
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            stats["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "used_pct": round((reserved / total) * 100, 1) if total > 0 else 0,
            }
    except Exception:
        pass

    # CPU and RAM stats (requires psutil)
    try:
        import psutil
        stats["cpu"] = {
            "percent": psutil.cpu_percent(interval=0.1),
            "cores": psutil.cpu_count(),
        }
        mem = psutil.virtual_memory()
        stats["ram"] = {
            "used_gb": round(mem.used / 1024**3, 2),
            "available_gb": round(mem.available / 1024**3, 2),
            "total_gb": round(mem.total / 1024**3, 2),
            "used_pct": mem.percent,
        }
    except ImportError:
        pass
    except Exception:
        pass

    return stats


@app.get("/models")
def get_models():
    return {"models": list_available_models()}


@app.post("/sessions", response_model=Session)
def create_session(payload: SessionCreate):
    session_id = str(uuid4())
    session = Session(id=session_id, name=payload.name)
    state.create_session(session)
    ilog.log_session_created(session_id, payload.name)
    return session


@app.get("/sessions/{session_id}", response_model=Session)
def get_session(session_id: str):
    try:
        return state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/uploads/video", response_model=Session)
def upload_video(session_id: str, file: UploadFile = File(...)):
    try:
        session = state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    dest = session_dir / file.filename

    with dest.open("wb") as f:
        content = file.file.read()
        f.write(content)
        file_size = len(content)

    ilog.log_video_uploaded(session_id, file.filename, file_size)
    return state.update_session(session_id, video_path=str(dest))


@app.post("/uploads/sample", response_model=Session)
def create_sample_clip(payload: SampleClipRequest):
    try:
        session = state.get_session(payload.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.video_path:
        raise HTTPException(status_code=400, detail="Video not uploaded")

    try:
        duration = float(payload.duration_seconds)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        raise HTTPException(status_code=400, detail="Duration must be > 0 seconds")

    source_path = Path(session.video_path)
    duration_label = int(round(duration))
    output_name = f"{source_path.stem}_sample_{duration_label}s.mp4"
    output_path = source_path.parent / output_name

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-t",
        str(duration),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr.strip() or "Sample clip failed")

    return state.update_session(payload.session_id, video_path=str(output_path))


@app.post("/frames/extract")
def extract_frames(payload: FrameExtractionRequest):
    try:
        session = state.get_session(payload.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.video_path:
        raise HTTPException(status_code=400, detail="Video not uploaded")

    frames_dir = SESSIONS_DIR / payload.session_id / "frames"
    thumbs_dir = SESSIONS_DIR / payload.session_id / "thumbs"
    if frames_dir.exists():
        for item in frames_dir.glob("*.jpg"):
            try:
                item.unlink()
            except Exception:
                pass
    if thumbs_dir.exists():
        for item in thumbs_dir.glob("*.jpg"):
            try:
                item.unlink()
            except Exception:
                pass
    frames_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(frames_dir / "%05d.jpg")

    # Probe FPS before extraction so we can force it with -r
    fps = _probe_video_fps(session.video_path) or 30.0

    # Build ffmpeg command with optional trim
    cmd = ["ffmpeg", "-y"]
    if payload.start_time is not None:
        cmd.extend(["-ss", str(payload.start_time)])
    cmd.extend(["-i", session.video_path])
    if payload.end_time is not None:
        if payload.start_time is not None:
            # Use duration instead of end time
            duration = payload.end_time - payload.start_time
            cmd.extend(["-t", str(duration)])
        else:
            cmd.extend(["-t", str(payload.end_time)])
    cmd.extend(["-r", str(fps), "-q:v", str(payload.quality), "-fps_mode", "cfr", "-start_number", "0", output_pattern])

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr.strip() or "Frame extraction failed")

    # Thumbnails with same trim
    thumb_pattern = str(thumbs_dir / "%05d.jpg")
    thumb_cmd = ["ffmpeg", "-y"]
    if payload.start_time is not None:
        thumb_cmd.extend(["-ss", str(payload.start_time)])
    thumb_cmd.extend(["-i", session.video_path])
    if payload.end_time is not None:
        if payload.start_time is not None:
            duration = payload.end_time - payload.start_time
            thumb_cmd.extend(["-t", str(duration)])
        else:
            thumb_cmd.extend(["-t", str(payload.end_time)])
    thumb_cmd.extend(["-r", str(fps), "-q:v", "5", "-fps_mode", "cfr", "-vf", "scale='min(640,iw)':-1", "-start_number", "0", thumb_pattern])
    thumb_result = subprocess.run(thumb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if thumb_result.returncode != 0:
        print(f"[thumbs] generation failed: {thumb_result.stderr.strip()}")

    frame_count = len(list(frames_dir.glob("*.jpg")))

    # Validate frame count against expected
    video_duration = _probe_video_duration(session.video_path)
    if video_duration and video_duration > 0:
        trimmed_duration = video_duration
        if payload.start_time is not None or payload.end_time is not None:
            start = payload.start_time or 0.0
            end = payload.end_time or video_duration
            trimmed_duration = min(end, video_duration) - start
        expected = fps * trimmed_duration
        if abs(frame_count - expected) > 2:
            print(f"[warn] Frame count mismatch: extracted={frame_count}, expected={expected:.0f} ({fps}fps x {trimmed_duration:.2f}s)")

    config = session.config or {}
    updated_config = {**config, "video_fps": float(fps)}
    state.update_session(payload.session_id, config=updated_config)

    ilog.log_frames_extracted(payload.session_id, frame_count, payload.quality)

    return {"status": "ok", "frames_dir": str(frames_dir)}


def _is_blank_frame(img: np.ndarray, min_std: float = 15.0, min_unique_ratio: float = 0.01) -> bool:
    """
    Check if a frame is blank/grey/uniform.

    A frame is considered blank if:
    1. Standard deviation across all pixels is very low (uniform color)
    2. Very few unique intensity values (solid color or near-solid)
    """
    # Convert to grayscale for simpler analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Check 1: Standard deviation - blank frames have very low std
    std = np.std(gray)
    if std < min_std:
        return True

    # Check 2: Count unique values - blank frames have few unique intensities
    # Sample for speed (every 4th pixel)
    sampled = gray[::4, ::4].flatten()
    unique_count = len(np.unique(sampled))
    total_sampled = len(sampled)
    unique_ratio = unique_count / 256.0  # ratio of possible values used

    if unique_ratio < min_unique_ratio:
        return True

    # Check 3: If 90% of pixels are within 20 intensity levels, it's blank
    hist, _ = np.histogram(sampled, bins=256, range=(0, 256))
    sorted_hist = np.sort(hist)[::-1]
    top_bins_coverage = np.sum(sorted_hist[:20]) / total_sampled
    if top_bins_coverage > 0.95:
        return True

    return False


@app.post("/frames/detect-grey-start", response_model=DetectGreyResponse)
def detect_grey_start(payload: DetectGreyRequest):
    """Scan video for first non-grey/blank frame. Returns timestamp for trim start."""
    try:
        session = state.get_session(payload.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.video_path:
        raise HTTPException(status_code=400, detail="Video not uploaded")

    # Get video FPS for timestamp calculation
    fps = _probe_video_fps(session.video_path) or 30.0

    # Create temp directory for analysis frames
    temp_dir = Path(tempfile.mkdtemp(prefix="grey_detect_"))
    try:
        # Extract frames at low quality for quick analysis
        output_pattern = str(temp_dir / "%05d.jpg")
        cmd = [
            "ffmpeg", "-y",
            "-i", session.video_path,
            "-vframes", str(payload.max_frames),
            "-q:v", "10",  # low quality is fine for analysis
            "-vf", "scale=320:-1",  # small size for speed
            "-start_number", "0",
            output_pattern
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail="Failed to extract frames for analysis")

        # Analyze frames for blankness
        frame_files = sorted(temp_dir.glob("*.jpg"))
        first_valid = 0

        for i, frame_path in enumerate(frame_files):
            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            if not _is_blank_frame(img):
                first_valid = i
                break
        else:
            # All frames are blank, return 0
            first_valid = 0

        first_valid_time = first_valid / fps

        return DetectGreyResponse(
            first_valid_frame=first_valid,
            first_valid_time=round(first_valid_time, 2),
            frames_scanned=len(frame_files)
        )
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/frames/list/{session_id}", response_model=FrameListResponse)
def list_frames(session_id: str):
    try:
        state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    frames_dir = SESSIONS_DIR / session_id / "frames"
    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail="Frames not extracted")

    frame_files = sorted([p.name for p in frames_dir.glob("*.jpg")])
    frame_width = None
    frame_height = None
    if frame_files:
        sample_path = frames_dir / frame_files[0]
        img = cv2.imread(str(sample_path))
        if img is not None:
            frame_height, frame_width = img.shape[:2]
    thumbs_dir = SESSIONS_DIR / session_id / "thumbs"
    has_thumbs = thumbs_dir.exists() and any(thumbs_dir.glob("*.jpg"))
    return FrameListResponse(
        session_id=session_id,
        frame_count=len(frame_files),
        frame_files=frame_files,
        frame_width=frame_width,
        frame_height=frame_height,
        has_thumbnails=has_thumbs,
    )


@app.get("/frames/suggest/{session_id}", response_model=FrameSuggestionResponse)
def suggest_frames(session_id: str, top_k: int = 7, use_dinov2: bool = True):
    """Suggest optimal frames for annotation based on quality and content."""
    try:
        state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    frames_dir = SESSIONS_DIR / session_id / "frames"
    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail="Frames not extracted")

    try:
        suggested = suggest_optimal_frames(
            frames_dir=frames_dir,
            top_k=top_k,
            use_dinov2=use_dinov2,
            max_samples=50,
        )

        return FrameSuggestionResponse(
            session_id=session_id,
            suggested_frames=suggested,
            total_analyzed=len(list(frames_dir.glob("*.jpg"))),
            method_used=suggested[0]["method"] if suggested else "none",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Frame analysis failed: {e}"
        )


@app.get("/frames/{session_id}/{frame_name}")
def get_frame(session_id: str, frame_name: str):
    try:
        state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    frame_path = SESSIONS_DIR / session_id / "frames" / frame_name
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")

    return FileResponse(frame_path)


@app.get("/frames/thumbs/{session_id}/{frame_name}")
def get_frame_thumb(session_id: str, frame_name: str):
    try:
        state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    thumb_path = SESSIONS_DIR / session_id / "thumbs" / frame_name
    frame_path = SESSIONS_DIR / session_id / "frames" / frame_name
    if thumb_path.exists():
        return FileResponse(thumb_path)
    if frame_path.exists():
        return FileResponse(frame_path)
    raise HTTPException(status_code=404, detail="Frame not found")


@app.get("/annotation/frames/{session_id}/{frame_index}")
def get_frame_annotations(session_id: str, frame_index: int):
    try:
        state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    annotations_dir = SESSIONS_DIR / session_id / "annotations"
    frame_file = annotations_dir / f"frame_{frame_index:05d}.json"
    if not frame_file.exists():
        return {"frame_index": frame_index, "objects": {}}

    try:
        data = json.loads(frame_file.read_text())
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Corrupted annotation file")

    return data


@app.get("/annotation/objects/{session_id}")
def list_annotation_objects(session_id: str):
    """Gets a list of all annotated objects."""
    try:
        state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    annotations_dir = SESSIONS_DIR / session_id / "annotations"
    objects = {}
    if annotations_dir.exists():
        for frame_file in sorted(annotations_dir.glob("frame_*.json")):
            try:
                data = json.loads(frame_file.read_text())
            except json.JSONDecodeError:
                continue
            frame_idx = data.get("frame_index")
            if frame_idx is None:
                try:
                    frame_idx = int(frame_file.stem.split("_")[1])
                except (IndexError, ValueError):
                    continue
            for name, pts in (data.get("objects") or {}).items():
                if not pts:
                    continue
                entry = objects.setdefault(name, {"frames": [], "point_count": 0})
                entry["frames"].append(frame_idx)
                entry["point_count"] += len(pts)

    for entry in objects.values():
        entry["frames"].sort()

    return {"objects": objects}


@app.post("/config", response_model=Session)
def update_config(payload: ConfigUpdate):
    try:
        session = state.get_session(payload.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    config = payload.model_dump(exclude={"session_id"}, exclude_none=True)
    updated_config = {**session.config, **config}
    ilog.log_config_updated(payload.session_id, config)
    return state.update_session(payload.session_id, config=updated_config, output_dir=payload.output_dir)


@app.post("/annotation/points")
def add_annotation_points(payload: AnnotationPayload):
    try:
        session = state.get_session(payload.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    annotations_dir = SESSIONS_DIR / payload.session_id / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    frame_file = annotations_dir / f"frame_{payload.frame_index:05d}.json"

    existing = {"frame_index": payload.frame_index, "objects": {}}
    if frame_file.exists():
        try:
            existing = json.loads(frame_file.read_text())
        except json.JSONDecodeError:
            existing = {"frame_index": payload.frame_index, "objects": {}}

    existing["objects"][payload.object_name] = [
        {"x": p.x, "y": p.y, "label": p.label} for p in payload.points
    ]
    # Save bounding box if provided
    if payload.bbox:
        existing.setdefault("bboxes", {})
        existing["bboxes"][payload.object_name] = {
            "x1": payload.bbox.x1, "y1": payload.bbox.y1,
            "x2": payload.bbox.x2, "y2": payload.bbox.y2,
        }
    prev_name = payload.previous_object_name
    if prev_name and prev_name != payload.object_name:
        existing["objects"].pop(prev_name, None)
        if "bboxes" in existing:
            existing["bboxes"].pop(prev_name, None)
    frame_file.write_text(json.dumps(existing, indent=2))
    config = session.config or {}
    if config.get("reference_frame") != payload.frame_index:
        updated_config = {**config, "reference_frame": payload.frame_index}
        state.update_session(payload.session_id, config=updated_config)

    # Log the annotation save
    ilog.log_points_saved(
        payload.session_id,
        payload.frame_index,
        payload.object_name,
        [p.model_dump() for p in payload.points],
    )

    return {
        "status": "saved",
        "points": len(payload.points),
        "frame": payload.frame_index,
        "object": payload.object_name,
        "path": str(frame_file),
        "reference_frame": payload.frame_index,
    }


@app.delete("/annotation/object/{session_id}/{object_name}")
def delete_annotation_object(session_id: str, object_name: str):
    try:
        state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    annotations_dir = SESSIONS_DIR / session_id / "annotations"
    if not annotations_dir.exists():
        return {"status": "ok", "deleted_from_frames": 0}

    deleted_count = 0
    for frame_file in sorted(annotations_dir.glob("frame_*.json")):
        try:
            data = json.loads(frame_file.read_text())
        except json.JSONDecodeError:
            continue
        if object_name in data.get("objects", {}):
            del data["objects"][object_name]
            frame_file.write_text(json.dumps(data, indent=2))
            deleted_count += 1

    return {"status": "ok", "object": object_name, "deleted_from_frames": deleted_count}


@app.post("/annotation/test-mask")
def test_mask(payload: AnnotationPayload):
    try:
        state.get_session(payload.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    if not payload.points:
        raise HTTPException(status_code=400, detail="No points provided")

    ilog.log_test_mask(payload.session_id, payload.frame_index, payload.object_name, len(payload.points))

    try:
        result = test_mask_preview(
            payload.session_id,
            payload.frame_index,
            payload.object_name,
            [p.model_dump() for p in payload.points],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "status": "ok",
        "message": "Mask preview generated",
        **result,
    }


@app.post("/processing/start", response_model=ProcessingStatus)
def start_processing(payload: ProcessingStartRequest):
    try:
        state.get_session(payload.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    active = state.get_thread(payload.session_id)
    if active and active.is_alive():
        raise HTTPException(status_code=400, detail="Processing already running")

    # Count objects from annotations
    annotations_dir = SESSIONS_DIR / payload.session_id / "annotations"
    object_count = 0
    if annotations_dir.exists():
        for f in annotations_dir.glob("frame_*.json"):
            try:
                data = json.loads(f.read_text())
                object_count = max(object_count, len(data.get("objects", {})))
            except (json.JSONDecodeError, OSError):
                pass

    ilog.log_processing_started(payload.session_id, object_count)

    status = ProcessingStatus(
        session_id=payload.session_id,
        status="starting",
        progress=0.0,
        message="Starting processing",
    )
    state.set_processing(status)
    start_background_job(payload.session_id)
    return status


@app.get("/processing/status/{session_id}", response_model=ProcessingStatus)
def get_processing_status(session_id: str):
    try:
        return state.get_processing(session_id)
    except KeyError:
        # Check if session exists and has outputs (completed processing)
        try:
            session = state.get_session(session_id)
            config = session.config or {}
            outputs = config.get("outputs", {})
            if outputs:
                # Session has outputs - processing completed
                return ProcessingStatus(
                    session_id=session_id,
                    status="completed",
                    progress=1.0,
                    message="Processing complete",
                )
            else:
                # Session exists but no processing started
                return ProcessingStatus(
                    session_id=session_id,
                    status="idle",
                    progress=0.0,
                    message="Ready to process",
                )
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")


@app.get("/processing/preview/{session_id}")
def get_processing_preview(session_id: str):
    preview_path = SESSIONS_DIR / session_id / "previews" / "latest.jpg"
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview not available")
    return FileResponse(preview_path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/results/{session_id}", response_model=ResultsResponse)
def get_results(session_id: str):
    try:
        session = state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    outputs = {}
    if session.config:
        outputs = session.config.get("outputs", {}) or {}
    outputs_meta = session.config.get("outputs_meta", {}) if session.config else None
    profiling = session.config.get("profiling") if session.config else None

    output_keys = {
        "annotated_video": outputs.get("annotated_video"),
        "csv": outputs.get("csv"),
        "elan": outputs.get("elan"),
        "resource_profile": outputs.get("resource_profile"),
        "masks_json": outputs.get("masks_json"),
    }
    file_exists = {}
    for key, path_str in output_keys.items():
        if path_str:
            file_exists[key] = Path(path_str).exists()
        else:
            file_exists[key] = False

    return ResultsResponse(
        session_id=session_id,
        outputs=output_keys,
        file_exists=file_exists,
        profiling=profiling,
        outputs_meta=outputs_meta,
    )


@app.get("/results/download/{session_id}")
def download_result(session_id: str, kind: str):
    try:
        session = state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    outputs = {}
    if session.config:
        outputs = session.config.get("outputs", {}) or {}

    path = outputs.get(kind)
    if not path:
        raise HTTPException(status_code=404, detail="Output not found")

    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File missing on disk")

    media_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    headers = {"Content-Disposition": f'attachment; filename="{file_path.name}"'}
    return FileResponse(file_path, media_type=media_type, headers=headers)


@app.get("/results/download-all/{session_id}")
def download_all_results(session_id: str):
    import zipfile
    import io

    try:
        session = state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    outputs = {}
    if session.config:
        outputs = session.config.get("outputs", {}) or {}

    # Collect all existing output files
    files_to_zip = []
    for key, path_str in outputs.items():
        if path_str and Path(path_str).exists():
            files_to_zip.append(Path(path_str))

    if not files_to_zip:
        raise HTTPException(status_code=404, detail="No output files available")

    # Determine ZIP filename from video stem
    video_stem = "results"
    if session.video_path:
        video_stem = Path(session.video_path).stem

    # Create ZIP in memory (skip compression for already-compressed formats)
    text_exts = {'.csv', '.eaf', '.json', '.html', '.xml', '.txt'}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for file_path in files_to_zip:
            compress = zipfile.ZIP_DEFLATED if file_path.suffix.lower() in text_exts else zipfile.ZIP_STORED
            zf.write(file_path, file_path.name, compress_type=compress)
    buf.seek(0)

    zip_filename = f"{video_stem}_ALL_RESULTS.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
    )
