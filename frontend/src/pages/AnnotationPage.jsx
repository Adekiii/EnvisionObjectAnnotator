import React, { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  API_BASE,
  listFrames,
  submitAnnotation,
  testMask,
  fetchFrameAnnotations,
  fetchSessionObjects,
  suggestFrames,
  deleteAnnotationObject,
} from "../api.js";
import {
  Button,
  Grid,
  Column,
  Tile,
  TextInput,
  Checkbox,
  Slider,
  Tag,
  InlineNotification,
  InlineLoading,
  Modal,
  ProgressIndicator,
  ProgressStep,
  StructuredListWrapper,
  StructuredListHead,
  StructuredListRow,
  StructuredListCell,
  StructuredListBody,
  DefinitionTooltip,
} from "@carbon/react";
import {
  ArrowLeft,
  ArrowRight,
  Add,
  View,
  Undo,
  ChevronLeft,
  ChevronRight,
  CircleFilled,
  AddAlt,
  SubtractAlt,
  Keyboard,
  Image,
  WarningAlt,
  Close,
  ZoomIn,
  TouchInteraction,
  Analytics,
  Checkmark,
} from "@carbon/icons-react";

export default function AnnotationPage() {
  const [sessionId, setSessionId] = useState("");
  const [frames, setFrames] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [points, setPoints] = useState([]);
  const [objectName, setObjectName] = useState("");
  const [sessionObjects, setSessionObjects] = useState({});
  const sessionObjectsRef = useRef({});
  const [selectedObject, setSelectedObject] = useState("");
  const [frameObjects, setFrameObjects] = useState({});
  const [hydratedFrame, setHydratedFrame] = useState(null);
  const [savedCounts, setSavedCounts] = useState({});
  const [videoDims, setVideoDims] = useState(null);
  const [useThumbs, setUseThumbs] = useState(true);
  const [labelMode, setLabelMode] = useState(1);
  const [status, setStatus] = useState("Load a session to begin annotating.");
  const [statusKind, setStatusKind] = useState("info");
  const [busy, setBusy] = useState(false);
  // Live mask preview
  const [previewPolygons, setPreviewPolygons] = useState(null);
  const [previewMeta, setPreviewMeta] = useState(null);
  const [markAsTarget, setMarkAsTarget] = useState(false);
  const [targetToggleTouched, setTargetToggleTouched] = useState(false);
  const [loading, setLoading] = useState(true);
  const [suggestedFrames, setSuggestedFrames] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedSuggestions, setSelectedSuggestions] = useState(new Set());
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);
  const [annotationMode, setAnnotationMode] = useState("point"); // "point" or "bbox"
  const [bboxDraw, setBboxDraw] = useState(null); // {startFx, startFy} while dragging
  const [objectBboxes, setObjectBboxes] = useState({}); // {objectName: {x1,y1,x2,y2,fx1,fy1,fx2,fy2}}
  const annotationsCache = useRef(new Map());
  const maskCache = useRef(new Map());
  const previewRequestSeq = useRef(0);
  const imgRef = useRef(null);
  const objectInputRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    const stored = localStorage.getItem("eoa_session");
    if (stored) {
      fetch(`${API_BASE}/sessions/${stored}`)
        .then((res) => {
          if (res.ok) {
            setSessionId(stored);
          } else {
            localStorage.removeItem("eoa_session");
            setStatus("Session expired. Please return to Config page to start a new session.");
            setStatusKind("warning");
          }
        })
        .catch(() => {
          localStorage.removeItem("eoa_session");
          setStatus("Backend unavailable. Please check if the server is running.");
          setStatusKind("error");
        })
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    function handleKeyDown(e) {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      switch (e.key.toLowerCase()) {
        case "c":
          e.preventDefault();
          objectInputRef.current?.focus();
          break;
        case "t":
          e.preventDefault();
          handleTestMask();
          break;
        case "z":
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            undoLastPoint();
          }
          break;
        case "arrowleft":
          e.preventDefault();
          setCurrentIndex((prev) => Math.max(0, prev - 1));
          break;
        case "arrowright":
          e.preventDefault();
          setCurrentIndex((prev) => Math.min(frames.length - 1, prev + 1));
          break;
        case "1":
          e.preventDefault();
          setLabelMode(1);
          break;
        case "2":
          e.preventDefault();
          setLabelMode(0);
          break;
        default:
          break;
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [frames.length, selectedObject, frameObjects, sessionId, currentIndex]);

  useEffect(() => {
    annotationsCache.current.clear();
    maskCache.current.clear();
    setVideoDims(null);
    setUseThumbs(true);
    setMarkAsTarget(false);
    setTargetToggleTouched(false);
    setSessionObjectsBoth({});
  }, [sessionId]);

  // Keeps sessionObjects (state, for render) and sessionObjectsRef (for
  // synchronous reads inside closures like hydrateFrame/handleDeleteObject,
  // where waiting for a re-render would read stale data) in lockstep.
  function setSessionObjectsBoth(next) {
    sessionObjectsRef.current = next;
    setSessionObjects(next);
  }

  async function refreshSessionObjects() {
    if (!sessionId) return;
    try {
      const data = await fetchSessionObjects(sessionId);
      setSessionObjectsBoth(data.objects || {});
    } catch (err) {
      // Non-fatal: the persistent object panel just keeps showing whatever
      // it already had until the next successful refresh.
    }
  }

  useEffect(() => {
    refreshSessionObjects();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const objectList = Object.keys(sessionObjects);

  useEffect(() => {
    async function fetchFrames() {
      if (!sessionId) return;
      try {
        const data = await listFrames(sessionId);
        setFrames(data.frame_files || []);
        setCurrentIndex(0);
        if (data.frame_width && data.frame_height) {
          setVideoDims({ width: data.frame_width, height: data.frame_height });
        }
        setUseThumbs(Boolean(data.has_thumbnails));
        setStatus(data.frame_files?.length ? "Frames ready." : "No frames found.");
        setStatusKind("info");
      } catch (err) {
        setStatus(err.message);
        setStatusKind("error");
      }
    }
    fetchFrames();
  }, [sessionId]);

  useEffect(() => {
    async function hydrateFrame() {
      if (!sessionId) return;
      try {
        const cacheKey = `${sessionId}:${currentIndex}`;
        const normalizeObjects = (objects) => {
          const normalized = {};
          Object.entries(objects || {}).forEach(([name, pts]) => {
            normalized[name] = (pts || []).map((p, idx) => ({
              ...p,
              fx: videoDims && videoDims.width ? p.x / videoDims.width : p.fx,
              fy: videoDims && videoDims.height ? p.y / videoDims.height : p.fy,
              id: p.id || `${currentIndex}-${name}-${idx}-${p.x}-${p.y}-${p.label}`,
            }));
          });
          return normalized;
        };

        // Pick which object stays selected after a frame change. If the
        // currently-selected object still exists anywhere in the session
        // (globally, not just on this specific frame), keep it selected --
        // this is what makes drift correction possible: select "pig" on
        // frame 0, jump to frame 1000 where "pig" has no points yet, and
        // it's still the selected object (with an empty point list) ready
        // for new correction clicks, instead of the selection jumping to
        // whatever happens to be first on frame 1000 or disappearing.
        const pickSelection = (normalized) => {
          const globalNames = Object.keys(sessionObjectsRef.current);
          const keepSelected =
            selectedObject &&
            (Object.prototype.hasOwnProperty.call(normalized, selectedObject) ||
              globalNames.includes(selectedObject));
          return keepSelected ? selectedObject : Object.keys(normalized)[0] || globalNames[0] || "";
        };

        const cached = annotationsCache.current.get(cacheKey);
        if (cached) {
          const normalized = normalizeObjects(cached);
          setFrameObjects(normalized);
          setHydratedFrame(currentIndex);
          const counts = {};
          Object.entries(normalized).forEach(([name, pts]) => {
            counts[name] = Array.isArray(pts) ? pts.length : 0;
          });
          setSavedCounts(counts);
          const nextSelected = pickSelection(normalized);
          setSelectedObject(nextSelected);
          setPoints(nextSelected ? normalized[nextSelected] || [] : []);
          setMarkAsTarget(isTargetName(nextSelected));
          setTargetToggleTouched(false);
          return;
        }

        const data = await fetchFrameAnnotations(sessionId, currentIndex);
        const objects = data.objects || {};
        const normalized = normalizeObjects(objects);
        annotationsCache.current.set(cacheKey, normalized);
        setFrameObjects(normalized);
        setHydratedFrame(currentIndex);
        // Load bboxes if present
        const bboxes = data.bboxes || {};
        const loadedBboxes = {};
        const baseW = videoDims?.width || 1;
        const baseH = videoDims?.height || 1;
        Object.entries(bboxes).forEach(([name, bb]) => {
          loadedBboxes[name] = {
            x1: bb.x1, y1: bb.y1, x2: bb.x2, y2: bb.y2,
            fx1: bb.x1 / baseW, fy1: bb.y1 / baseH,
            fx2: bb.x2 / baseW, fy2: bb.y2 / baseH,
          };
        });
        setObjectBboxes(loadedBboxes);
        const counts = {};
        Object.entries(normalized).forEach(([name, pts]) => {
          counts[name] = Array.isArray(pts) ? pts.length : 0;
        });
        setSavedCounts(counts);
        const nextSelected = pickSelection(normalized);
        setSelectedObject(nextSelected);
        setPoints(nextSelected ? normalized[nextSelected] || [] : []);
        setMarkAsTarget(isTargetName(nextSelected));
        setTargetToggleTouched(false);
      } catch (err) {
        setStatus(err.message);
        setStatusKind("error");
      }
    }
    hydrateFrame();
  }, [currentIndex, sessionId, videoDims]);

  // Keep the inline mask preview in sync with whichever object/frame is
  // actually showing -- without this, switching frames or objects would
  // leave a stale mask from the previous context overlaid on the new one.
  // Re-deriving from `frameObjects[selectedObject]` (rather than only firing
  // on selectedObject/currentIndex) also means a redundant call right after
  // a click just hits generateMaskPreview's own signature cache instead of
  // re-requesting anything.
  //
  // Skip while `hydratedFrame` doesn't match `currentIndex` yet: right after
  // navigating to a different frame, this effect (and the auto-save one
  // below) still see the PREVIOUS frame's `frameObjects` in their closure --
  // `currentIndex` already updated, but hydrateFrame's own state update for
  // the new frame hasn't landed. Without this guard we'd briefly preview the
  // old frame's mask at the wrong coordinates. It corrects itself the moment
  // hydrateFrame's update lands and re-triggers this effect with the frame's
  // real (possibly empty) points.
  useEffect(() => {
    if (hydratedFrame !== currentIndex) return;
    const currentPoints = frameObjects[selectedObject] || [];
    if (currentPoints.length > 0) {
      generateMaskPreview(selectedObject, currentPoints, { silent: true });
    } else {
      previewRequestSeq.current += 1;
      setPreviewPolygons(null);
      setPreviewMeta(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedObject, currentIndex, frameObjects, hydratedFrame]);

  // Auto-save: persist the current object's points to the backend as soon
  // as they change (or the "mark as target" checkbox is toggled), instead
  // of requiring a manual "Save" click. A bad click is undone instantly
  // with Undo/click-to-remove, so there's no real risk in saving eagerly --
  // same "make the workflow as smooth as possible" reasoning as the live
  // mask preview above. There is no Save button; this is the only place
  // points ever reach the backend.
  //
  // Skip while `hydratedFrame` doesn't match `currentIndex` yet -- same race
  // as the mask-preview effect above, but with a much worse consequence:
  // without this guard, navigating from a frame with points to one without
  // any would fire this effect with the OLD frame's `frameObjects` and the
  // NEW frame's `currentIndex` still in its closure, silently saving the old
  // frame's points onto the new frame even though nothing was clicked there.
  useEffect(() => {
    if (hydratedFrame !== currentIndex) return;
    // No length guard here: autoSavePoints itself decides whether there's
    // anything to do, including the zero-points case (clearing a
    // previously-saved object after an Undo removes its last point).
    autoSavePoints();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedObject, currentIndex, frameObjects, markAsTarget, targetToggleTouched, hydratedFrame]);

  function updateCache(frameIndex, updatedObjects) {
    if (!sessionId) return;
    annotationsCache.current.set(`${sessionId}:${frameIndex}`, updatedObjects);
  }

  function isTargetName(name) {
    return String(name || "").toLowerCase().includes("target");
  }

  function normalizeObjectName(name, targetFlag) {
    const trimmed = String(name || "").trim();
    if (!trimmed) return "";
    if (targetFlag && !isTargetName(trimmed)) {
      return `target_${trimmed}`;
    }
    return trimmed;
  }

  function displayObjectName(name) {
    const raw = String(name || "");
    const lower = raw.toLowerCase();
    if (lower.startsWith("target_")) return raw.slice("target_".length);
    if (lower.startsWith("target-")) return raw.slice("target-".length);
    if (lower.startsWith("target ")) return raw.slice("target ".length);
    return raw;
  }

  function mergePoints(primary, secondary) {
    const merged = [];
    const seen = new Set();
    const addPoint = (point) => {
      if (!point) return;
      const key = `${Math.round(point.x)}:${Math.round(point.y)}:${point.label}`;
      if (seen.has(key)) return;
      seen.add(key);
      merged.push(point);
    };
    (primary || []).forEach(addPoint);
    (secondary || []).forEach(addPoint);
    return merged;
  }

  function removePoint(pointId) {
    if (!selectedObject) return;
    const current = frameObjects[selectedObject] || [];
    const updatedPoints = current.filter((point) => point.id !== pointId);
    const updated = { ...frameObjects, [selectedObject]: updatedPoints };
    setFrameObjects(updated);
    setPoints(updatedPoints);
    updateCache(currentIndex, updated);
    // The live-preview effect (keyed on frameObjects) picks this up and
    // regenerates or clears the mask preview on its own.
  }

  async function handleDeleteObject(name) {
    if (!sessionId) return;
    try {
      await deleteAnnotationObject(sessionId, name);
      const updated = { ...frameObjects };
      delete updated[name];
      setFrameObjects(updated);
      // The backend removes this object from EVERY frame's annotation file,
      // not just the current one -- purge it from the whole client cache
      // too, or navigating back to a previously-visited frame would
      // resurrect it from stale cached data.
      for (const [key, obj] of annotationsCache.current.entries()) {
        if (Object.prototype.hasOwnProperty.call(obj, name)) {
          const clone = { ...obj };
          delete clone[name];
          annotationsCache.current.set(key, clone);
        }
      }
      const nextSessionObjects = { ...sessionObjectsRef.current };
      delete nextSessionObjects[name];
      setSessionObjectsBoth(nextSessionObjects);
      setSavedCounts((prev) => {
        const next = { ...prev };
        delete next[name];
        return next;
      });
      if (selectedObject === name) {
        const remainingGlobal = Object.keys(nextSessionObjects);
        const nextSelected = remainingGlobal[0] || "";
        setSelectedObject(nextSelected);
        setPoints(nextSelected ? updated[nextSelected] || [] : []);
      }
      updateCache(currentIndex, updated);
      setStatus(`Deleted object "${name.replace("TARGET:", "")}"`);
      setStatusKind("success");
    } catch (err) {
      setStatus(err.message);
      setStatusKind("error");
    }
  }

  function undoLastPoint() {
    if (!selectedObject) return;
    const current = frameObjects[selectedObject] || [];
    if (current.length === 0) return;
    const updatedPoints = current.slice(0, -1);
    const updated = { ...frameObjects, [selectedObject]: updatedPoints };
    setFrameObjects(updated);
    setPoints(updatedPoints);
    updateCache(currentIndex, updated);
    // The live-preview effect (keyed on frameObjects/selectedObject/currentIndex)
    // picks this change up on its own and regenerates or clears the preview.
  }

  function buildMaskSignature(pointsForObject) {
    return (pointsForObject || [])
      .map((p) => `${Math.round(p.x)}:${Math.round(p.y)}:${p.label}`)
      .join("|");
  }

  function _getImageCoords(event) {
    const rect = imgRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const fx = Math.max(0, Math.min(1, x / rect.width));
    const fy = Math.max(0, Math.min(1, y / rect.height));
    const baseW = videoDims?.width || imgRef.current.naturalWidth || 1;
    const baseH = videoDims?.height || imgRef.current.naturalHeight || 1;
    return { fx, fy, nx: fx * baseW, ny: fy * baseH };
  }

  function handleImageMouseDown(event) {
    if (!imgRef.current || !selectedObject) return;
    if (annotationMode !== "bbox") return;
    event.preventDefault();
    const { fx, fy } = _getImageCoords(event);
    setBboxDraw({ startFx: fx, startFy: fy, endFx: fx, endFy: fy });
  }

  function handleImageMouseMove(event) {
    if (!bboxDraw || annotationMode !== "bbox") return;
    const { fx, fy } = _getImageCoords(event);
    setBboxDraw((prev) => ({ ...prev, endFx: fx, endFy: fy }));
  }

  function handleImageMouseUp(event) {
    if (!bboxDraw || annotationMode !== "bbox" || !selectedObject) return;
    const { fx, fy, nx, ny } = _getImageCoords(event);
    const baseW = videoDims?.width || imgRef.current.naturalWidth || 1;
    const baseH = videoDims?.height || imgRef.current.naturalHeight || 1;
    const x1 = Math.min(bboxDraw.startFx, fx) * baseW;
    const y1 = Math.min(bboxDraw.startFy, fy) * baseH;
    const x2 = Math.max(bboxDraw.startFx, fx) * baseW;
    const y2 = Math.max(bboxDraw.startFy, fy) * baseH;
    // Only save if the box has meaningful size
    if (Math.abs(x2 - x1) > 5 && Math.abs(y2 - y1) > 5) {
      setObjectBboxes((prev) => ({
        ...prev,
        [selectedObject]: {
          x1, y1, x2, y2,
          fx1: Math.min(bboxDraw.startFx, fx),
          fy1: Math.min(bboxDraw.startFy, fy),
          fx2: Math.max(bboxDraw.startFx, fx),
          fy2: Math.max(bboxDraw.startFy, fy),
        },
      }));
    }
    setBboxDraw(null);
  }

  function handleImageClick(event) {
    if (!imgRef.current || !selectedObject) return;
    if (annotationMode === "bbox") return; // bbox uses mousedown/up
    const { fx, fy, nx, ny } = _getImageCoords(event);

    const next = [
      ...(frameObjects[selectedObject] || []),
      { x: nx, y: ny, label: labelMode, fx, fy, id: `${Date.now()}-${points.length}` },
    ];
    const updated = { ...frameObjects, [selectedObject]: next };
    setFrameObjects(updated);
    setPoints(next);
    updateCache(currentIndex, updated);
    // The live-preview effect (keyed on frameObjects/selectedObject/currentIndex)
    // picks this up and regenerates the mask with the new point set -- no
    // separate call needed here, and no manual "Test Mask" click required.
  }

  // Persists the current object's points for the current frame, including
  // committing a pending "mark as target" rename. Runs automatically
  // whenever points change or the target checkbox is toggled (see the
  // effect above) -- there's no manual Save button, so this never needs to
  // talk to the user except on a genuine failure.
  async function autoSavePoints() {
    if (!sessionId || !selectedObject) return;
    if (!frameObjects[selectedObject] || frameObjects[selectedObject].length === 0) {
      // An Undo/remove-click can bring an object back down to zero points
      // on this frame. If it was previously synced to the backend, fall
      // through and push the now-empty list so the clear actually takes
      // effect there too -- otherwise Process would still pick up the
      // stale point the user thought they'd removed. Nothing to do if it
      // was never saved in the first place.
      if (!(savedCounts[selectedObject] > 0)) {
        return;
      }
    }
    let objectNameToSave = selectedObject;
    let updatedObjects = frameObjects;
    let previousObjectName = null;
    if (markAsTarget) {
      const targetName = normalizeObjectName(selectedObject, true);
      if (targetName !== selectedObject) {
        updatedObjects = { ...frameObjects };
        if (updatedObjects[targetName]) {
          updatedObjects[targetName] = mergePoints(updatedObjects[targetName], updatedObjects[selectedObject]);
        } else {
          updatedObjects[targetName] = updatedObjects[selectedObject];
        }
        delete updatedObjects[selectedObject];
        objectNameToSave = targetName;
        previousObjectName = selectedObject;
      }
    } else if (isTargetName(selectedObject) && targetToggleTouched) {
      const baseName = displayObjectName(selectedObject).trim() || selectedObject;
      if (baseName !== selectedObject) {
        updatedObjects = { ...frameObjects };
        if (updatedObjects[baseName]) {
          updatedObjects[baseName] = mergePoints(updatedObjects[baseName], updatedObjects[selectedObject]);
        } else {
          updatedObjects[baseName] = updatedObjects[selectedObject];
        }
        delete updatedObjects[selectedObject];
        objectNameToSave = baseName;
        previousObjectName = selectedObject;
      }
    }
    const pointsToSave = (updatedObjects[objectNameToSave] || []).map(({ x, y, label }) => ({
      x,
      y,
      label,
    }));
    if (pointsToSave.length === 0 && !(savedCounts[objectNameToSave] > 0 || savedCounts[selectedObject] > 0)) {
      return;
    }
    try {
      const payload = {
        session_id: sessionId,
        frame_index: currentIndex,
        object_name: objectNameToSave,
        previous_object_name: previousObjectName,
        points: pointsToSave,
      };
      const bbox = objectBboxes[objectNameToSave] || objectBboxes[selectedObject];
      if (bbox) {
        payload.bbox = { x1: bbox.x1, y1: bbox.y1, x2: bbox.x2, y2: bbox.y2 };
      }
      await submitAnnotation(payload);
      setSavedCounts((prev) => ({
        ...prev,
        ...(previousObjectName ? { [previousObjectName]: 0 } : {}),
        [objectNameToSave]: pointsToSave.length,
      }));
      if (updatedObjects !== frameObjects) {
        setFrameObjects(updatedObjects);
        setSelectedObject(objectNameToSave);
        setPoints(updatedObjects[objectNameToSave] || []);
      }
      updateCache(currentIndex, updatedObjects);
      setMarkAsTarget(isTargetName(objectNameToSave));
      setTargetToggleTouched(false);
      // Refresh the session-wide object registry so the persistent panel
      // picks up this frame as a new/updated seed for objectNameToSave (and
      // drops previousObjectName on a target-toggle rename) -- the backend
      // is authoritative here since save can rename/merge objects.
      await refreshSessionObjects();
    } catch (err) {
      setStatus(`Auto-save failed: ${err.message}`);
      setStatusKind("error");
    }
  }

  async function handleProcessClick() {
    // Save ALL objects across ALL annotated frames before navigating to processing
    if (!sessionId) {
      navigate("/processing");
      return;
    }
    setBusy(true);
    setStatus("Saving all annotations...");
    setStatusKind("info");
    try {
      // Collect all frames from the cache
      const framesToSave = new Map();
      for (const [key, objects] of annotationsCache.current.entries()) {
        if (!key.startsWith(`${sessionId}:`)) continue;
        const frameIdx = parseInt(key.split(":")[1], 10);
        if (isNaN(frameIdx)) continue;
        framesToSave.set(frameIdx, objects);
      }
      // Also include the current frame's live state
      if (Object.keys(frameObjects).length > 0) {
        framesToSave.set(currentIndex, frameObjects);
      }
      // Save each object in each frame
      let totalSaved = 0;
      for (const [frameIdx, objects] of framesToSave.entries()) {
        for (const [objName, objPoints] of Object.entries(objects)) {
          if (!objPoints || objPoints.length === 0) continue;
          const pointsPayload = objPoints.map(({ x, y, label }) => ({
            x,
            y,
            label: label ?? 1,
          }));
          await submitAnnotation({
            session_id: sessionId,
            frame_index: frameIdx,
            object_name: objName,
            points: pointsPayload,
          });
          totalSaved++;
        }
      }
      setStatus(`Saved ${totalSaved} object(s). Starting processing...`);
      setStatusKind("success");
      navigate("/processing");
    } catch (err) {
      setStatus(`Failed to save annotations: ${err.message}`);
      setStatusKind("error");
    } finally {
      setBusy(false);
    }
  }

  // Core mask-preview call, shared by the live per-click preview and the
  // manual "Test Mask" button. Takes the object name and point list
  // explicitly rather than reading `selectedObject`/`frameObjects` back from
  // state, so a call fired right after a click always uses that click's
  // point, not a possibly-stale render.
  async function generateMaskPreview(objectNameArg, pointsList, { silent = false } = {}) {
    if (!sessionId || !objectNameArg || !pointsList || pointsList.length === 0) return;

    const signature = buildMaskSignature(pointsList);
    const cacheKey = `${sessionId}:${currentIndex}:${objectNameArg}:${signature}`;
    const cached = maskCache.current.get(cacheKey);
    if (cached) {
      setPreviewPolygons(cached.polygons);
      setPreviewMeta({ width: cached.width, height: cached.height });
      if (!silent) {
        setStatus("Loaded cached mask preview.");
        setStatusKind("info");
      }
      return;
    }

    const requestId = ++previewRequestSeq.current;
    if (!silent) {
      setStatus("Generating mask preview...");
      setStatusKind("info");
    }
    try {
      const testPayload = {
        session_id: sessionId,
        frame_index: currentIndex,
        object_name: objectNameArg,
        points: pointsList.map(({ x, y, label }) => ({ x, y, label })),
      };
      const response = await testMask(testPayload);
      if (requestId !== previewRequestSeq.current) return; // superseded by a newer click/undo
      const cacheValue = { polygons: response.polygons || [], width: response.width, height: response.height };
      maskCache.current.set(cacheKey, cacheValue);
      setPreviewPolygons(cacheValue.polygons);
      setPreviewMeta({ width: cacheValue.width, height: cacheValue.height });
      if (!silent) {
        setStatus(response.message || "Mask preview generated.");
        setStatusKind("success");
      }
    } catch (err) {
      if (requestId !== previewRequestSeq.current) return;
      console.error("Mask preview error:", err);
      // Errors surface even in silent/auto mode -- staying quiet about a
      // genuine failure (as opposed to the routine "generating..." noise)
      // would just leave the user wondering why nothing happened.
      setStatus(`Mask preview failed: ${err.message}`);
      setStatusKind("error");
    }
  }

  async function handleTestMask() {
    if (!sessionId) {
      setStatus("No session. Return to Config page to start a new session.");
      setStatusKind("error");
      return;
    }
    if (!selectedObject) {
      setStatus("Select an object to test.");
      setStatusKind("warning");
      return;
    }
    if (!frameObjects[selectedObject] || frameObjects[selectedObject].length === 0) {
      setStatus("Add at least one point before testing.");
      setStatusKind("warning");
      return;
    }
    setBusy(true);
    try {
      await generateMaskPreview(selectedObject, frameObjects[selectedObject]);
    } finally {
      setBusy(false);
    }
  }

  function handleAddObject() {
    const name = normalizeObjectName(objectName, markAsTarget);
    if (!name) {
      setStatus("Enter an object name.");
      setStatusKind("warning");
      return;
    }
    // Check the SESSION-wide registry, not just this frame's points -- an
    // object seeded on another frame (e.g. frame 0) but not this one should
    // be selected (with an empty point list here), not treated as new.
    if (Object.prototype.hasOwnProperty.call(sessionObjectsRef.current, name) || frameObjects[name]) {
      setSelectedObject(name);
      setPoints(frameObjects[name] || []);
      setObjectName("");
      setMarkAsTarget(isTargetName(name));
      setTargetToggleTouched(false);
      return;
    }
    const updated = { ...frameObjects, [name]: [] };
    setFrameObjects(updated);
    setSessionObjectsBoth({ ...sessionObjectsRef.current, [name]: { frames: [], point_count: 0 } });
    setSavedCounts((prev) => ({ ...prev, [name]: 0 }));
    setSelectedObject(name);
    setPoints([]);
    updateCache(currentIndex, updated);
    setObjectName("");
    setMarkAsTarget(isTargetName(name));
    setTargetToggleTouched(false);
  }

  // --- Frame suggestion handlers ---
  async function handleSuggestFrames() {
    if (suggestedFrames.length > 0) {
      setShowSuggestions(true);
      return;
    }
    if (!sessionId || frames.length === 0) {
      setStatus("Load frames before requesting suggestions.");
      setStatusKind("warning");
      return;
    }
    setLoadingSuggestions(true);
    setStatus("Analyzing frames for optimal suggestions...");
    setStatusKind("info");
    try {
      const result = await suggestFrames(sessionId, 7, true);
      setSuggestedFrames(result.suggested_frames || []);
      setShowSuggestions(true);
      setSelectedSuggestions(new Set());
      setStatus(`Found ${result.suggested_frames?.length || 0} optimal frames (${result.method_used})`);
      setStatusKind("success");
    } catch (err) {
      setStatus(err.message);
      setStatusKind("error");
      setSuggestedFrames([]);
    } finally {
      setLoadingSuggestions(false);
    }
  }

  function toggleSuggestionSelection(frameIndex) {
    setSelectedSuggestions((prev) => {
      const updated = new Set(prev);
      if (updated.has(frameIndex)) {
        updated.delete(frameIndex);
      } else {
        updated.add(frameIndex);
      }
      return updated;
    });
  }

  function navigateToSuggestion(frameIndex) {
    setCurrentIndex(frameIndex);
    setShowSuggestions(false);
  }

  const hasSavedObject = Object.entries(savedCounts).some(
    ([name, count]) => count > 0 && (frameObjects[name]?.length || 0) === count
  );
  const missingSteps = [];
  if (!sessionId) missingSteps.push("Load a session from the Config page.");
  if (frames.length === 0) missingSteps.push("Extract frames on the Config page.");
  if (!hasSavedObject) missingSteps.push("Save points for at least one object.");

  const frameName = frames[currentIndex];
  const frameUrl =
    sessionId && frameName
      ? `${API_BASE}${useThumbs ? "/frames/thumbs" : "/frames"}/${sessionId}/${frameName}`
      : "";

  return (
    <div className="app-shell">
      {/* Header */}
      <header
        style={{
          borderBottom: "1px solid #e0e0e0",
          padding: "0 1rem",
          height: "48px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          backgroundColor: "#fff",
          position: "sticky",
          top: 0,
          zIndex: 100,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <TouchInteraction size={20} />
          <span style={{ fontWeight: 600 }}>Annotation</span>
          {selectedObject && (
            <Tag type={isTargetName(selectedObject) ? "cyan" : "gray"} size="sm">
              {displayObjectName(selectedObject)}
            </Tag>
          )}
        </div>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <Button kind="secondary" size="sm" renderIcon={ArrowLeft} as={Link} to="/config">
            Back
          </Button>
          <Button
            size="sm"
            renderIcon={ArrowRight}
            onClick={handleProcessClick}
            disabled={busy || missingSteps.length > 0}
          >
            {busy ? "Saving..." : "Process"}
          </Button>
        </div>
      </header>

      <main className="app-content">
        <div className="page-container">
          {/* Progress indicator */}
          <div style={{ marginBottom: "1.5rem" }}>
            <ProgressIndicator currentIndex={2} spaceEqually>
              <ProgressStep label="Upload" secondaryLabel="Complete" />
              <ProgressStep label="Configure" secondaryLabel="Complete" />
              <ProgressStep label="Annotate" secondaryLabel="Mark objects" />
              <ProgressStep label="Process" secondaryLabel="Track & detect" />
            </ProgressIndicator>
          </div>

          <Grid>
            {/* Left column - Canvas */}
            <Column lg={12} md={6} sm={4}>
              <Tile style={{ padding: 0, overflow: "hidden", marginBottom: "1rem" }}>
                <div
                  style={{
                    backgroundColor: "#161616",
                    position: "relative",
                    cursor: selectedObject ? "crosshair" : "default",
                    minHeight: "300px",
                    userSelect: annotationMode === "bbox" ? "none" : undefined,
                  }}
                  onClick={handleImageClick}
                  onMouseDown={handleImageMouseDown}
                  onMouseMove={handleImageMouseMove}
                  onMouseUp={handleImageMouseUp}
                >
                  {loading ? (
                    <div style={{ padding: "4rem", textAlign: "center" }}>
                      <InlineLoading description="Loading session..." />
                    </div>
                  ) : frameUrl ? (
                    <img
                      ref={imgRef}
                      src={frameUrl}
                      alt="Frame"
                      style={{ width: "100%", height: "auto", display: "block" }}
                      onError={() => {
                        if (useThumbs) setUseThumbs(false);
                      }}
                    />
                  ) : (
                    <div style={{ padding: "4rem", display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center", color: "#8d8d8d" }}>
                      {!sessionId ? (
                        <>
                          <WarningAlt size={48} style={{ marginBottom: "1rem", color: "#f1c21b" }} />
                          <p style={{ fontWeight: 600, marginBottom: "0.5rem" }}>No Active Session</p>
                          <p style={{ fontSize: "0.875rem", marginBottom: "1rem" }}>
                            Your session may have expired or the backend was restarted.
                          </p>
                          <Button kind="secondary" size="sm" as={Link} to="/config" renderIcon={ArrowLeft}>
                            Go to Config
                          </Button>
                        </>
                      ) : (
                        <>
                          <Image size={48} style={{ marginBottom: "1rem", opacity: 0.5 }} />
                          <p>No frames loaded</p>
                        </>
                      )}
                    </div>
                  )}
                  {/* Live mask preview: drawn inline over the frame, updated on
                      every click -- no popup, nothing blocking the workflow. */}
                  {previewPolygons && previewPolygons.length > 0 && previewMeta && (
                    <svg
                      viewBox={`0 0 ${previewMeta.width} ${previewMeta.height}`}
                      preserveAspectRatio="none"
                      style={{
                        position: "absolute",
                        inset: 0,
                        width: "100%",
                        height: "100%",
                        pointerEvents: "none",
                      }}
                    >
                      {previewPolygons.map((poly, i) => (
                        <polygon
                          key={i}
                          points={poly.map(([x, y]) => `${x},${y}`).join(" ")}
                          fill="rgba(66, 190, 101, 0.45)"
                          stroke="rgba(66, 190, 101, 0.9)"
                          strokeWidth={Math.max(1, previewMeta.width / 400)}
                        />
                      ))}
                    </svg>
                  )}
                  {/* Point markers */}
                  {(frameObjects[selectedObject] || []).map((point) => (
                    <span
                      key={point.id || `${point.x}-${point.y}`}
                      style={{
                        position: "absolute",
                        left: `${(point.fx || 0) * 100}%`,
                        top: `${(point.fy || 0) * 100}%`,
                        width: "18px",
                        height: "18px",
                        borderRadius: "50%",
                        border: "3px solid white",
                        transform: "translate(-50%, -50%)",
                        backgroundColor: point.label === 1 ? "#42be65" : "#da1e28",
                        cursor: "pointer",
                        boxShadow: "0 2px 6px rgba(0,0,0,0.4)",
                        transition: "transform 0.15s ease",
                      }}
                      onClick={(event) => {
                        event.stopPropagation();
                        removePoint(point.id);
                      }}
                      title="Click to remove"
                    />
                  ))}
                  {/* Saved bbox overlay */}
                  {selectedObject && objectBboxes[selectedObject] && (
                    <div style={{
                      position: "absolute",
                      left: `${objectBboxes[selectedObject].fx1 * 100}%`,
                      top: `${objectBboxes[selectedObject].fy1 * 100}%`,
                      width: `${(objectBboxes[selectedObject].fx2 - objectBboxes[selectedObject].fx1) * 100}%`,
                      height: `${(objectBboxes[selectedObject].fy2 - objectBboxes[selectedObject].fy1) * 100}%`,
                      border: "2px dashed #ff832b",
                      backgroundColor: "rgba(255, 131, 43, 0.1)",
                      pointerEvents: "none",
                    }} />
                  )}
                  {/* Bbox drawing overlay */}
                  {bboxDraw && (
                    <div style={{
                      position: "absolute",
                      left: `${Math.min(bboxDraw.startFx, bboxDraw.endFx) * 100}%`,
                      top: `${Math.min(bboxDraw.startFy, bboxDraw.endFy) * 100}%`,
                      width: `${Math.abs(bboxDraw.endFx - bboxDraw.startFx) * 100}%`,
                      height: `${Math.abs(bboxDraw.endFy - bboxDraw.startFy) * 100}%`,
                      border: "2px dashed #0f62fe",
                      backgroundColor: "rgba(15, 98, 254, 0.1)",
                      pointerEvents: "none",
                    }} />
                  )}
                </div>
              </Tile>

              {/* Frame navigation */}
              <Tile>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
                  <Button
                    kind="ghost"
                    size="sm"
                    renderIcon={ChevronLeft}
                    onClick={() => setCurrentIndex((prev) => Math.max(0, prev - 1))}
                    disabled={currentIndex === 0}
                    iconDescription="Previous frame"
                  >
                    Prev
                  </Button>
                  <span style={{ fontSize: "0.875rem", color: "#525252" }}>
                    Frame {frames.length ? currentIndex + 1 : 0} / {frames.length}
                  </span>
                  <Button
                    kind="ghost"
                    size="sm"
                    renderIcon={ChevronRight}
                    onClick={() => setCurrentIndex((prev) => Math.min(frames.length - 1, prev + 1))}
                    disabled={currentIndex >= frames.length - 1}
                    iconDescription="Next frame"
                  >
                    Next
                  </Button>
                </div>
                <Slider
                  id="frame-slider"
                  labelText="Jump to frame"
                  min={0}
                  max={Math.max(0, frames.length - 1)}
                  value={currentIndex}
                  onChange={({ value }) => setCurrentIndex(value)}
                  disabled={frames.length === 0}
                  hideTextInput
                />
              </Tile>
            </Column>

            {/* Right column - Controls */}
            <Column lg={4} md={2} sm={4}>
              <div style={{ maxHeight: "calc(100vh - 120px)", overflowY: "auto", position: "sticky", top: "60px" }}>

              {/* Objects panel */}
              <Tile style={{ marginBottom: "1rem" }}>
                <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "1rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <View size={16} /> Objects
                </h3>
                <TextInput
                  ref={objectInputRef}
                  id="object-name"
                  labelText="Object name"
                  placeholder="e.g., hand, cup, ball"
                  value={objectName}
                  onChange={(e) => setObjectName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      handleAddObject();
                    }
                  }}
                  size="sm"
                />
                <Checkbox
                  id="mark-as-target"
                  labelText="Mark as target"
                  checked={markAsTarget}
                  onChange={(_, { checked }) => {
                    setMarkAsTarget(checked);
                    setTargetToggleTouched(true);
                  }}
                  style={{ marginTop: "0.75rem" }}
                />
                <div style={{ marginTop: "0.75rem" }}>
                  <Button kind="secondary" size="sm" onClick={handleAddObject} renderIcon={Add} style={{ width: "100%" }}>
                    Add
                  </Button>
                </div>

                {/* Object list */}
                <div style={{ marginTop: "1rem" }}>
                  {objectList.length === 0 ? (
                    <p style={{ fontSize: "0.875rem", color: "#6f6f6f" }}>No objects yet.</p>
                  ) : (
                    <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                      {objectList.map((name) => (
                        <div
                          key={name}
                          onClick={() => {
                            setSelectedObject(name);
                            setPoints(frameObjects[name] || []);
                            setMarkAsTarget(isTargetName(name));
                            setTargetToggleTouched(false);
                          }}
                          style={{
                            padding: "0.75rem",
                            borderRadius: "4px",
                            border: `2px solid ${selectedObject === name ? "#0f62fe" : "#e0e0e0"}`,
                            backgroundColor: selectedObject === name ? "#e5f6ff" : "#fff",
                            cursor: "pointer",
                            transition: "all 0.15s ease",
                          }}
                        >
                          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                            <span style={{ fontWeight: 500, display: "flex", alignItems: "center", gap: "0.5rem" }}>
                              {displayObjectName(name)}
                              {isTargetName(name) && <Tag type="cyan" size="sm">Target</Tag>}
                            </span>
                            <div style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
                              <Tag type="gray" size="sm">{(frameObjects[name] || []).length} pts here</Tag>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDeleteObject(name);
                                }}
                                title="Delete object"
                                style={{
                                  background: "none",
                                  border: "none",
                                  cursor: "pointer",
                                  padding: "2px",
                                  display: "flex",
                                  alignItems: "center",
                                  color: "#da1e28",
                                  opacity: 0.6,
                                }}
                                onMouseEnter={(e) => (e.currentTarget.style.opacity = 1)}
                                onMouseLeave={(e) => (e.currentTarget.style.opacity = 0.6)}
                              >
                                <Close size={16} />
                              </button>
                            </div>
                          </div>
                          {/* Seed frames: jump between the frames this object
                              already has points on, e.g. to compare frame 0's
                              seed against drift spotted later at frame 1000
                              and correct it right here. */}
                          {(sessionObjects[name]?.frames?.length || 0) > 0 && (
                            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.25rem", marginTop: "0.5rem" }}>
                              {sessionObjects[name].frames.map((f) => (
                                <button
                                  key={f}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setCurrentIndex(f);
                                  }}
                                  title={`Jump to frame ${f + 1}`}
                                  style={{
                                    fontSize: "0.7rem",
                                    lineHeight: "1.4",
                                    padding: "1px 6px",
                                    borderRadius: "10px",
                                    border: f === currentIndex ? "1px solid #0f62fe" : "1px solid #c6c6c6",
                                    backgroundColor: f === currentIndex ? "#e5f6ff" : "#f4f4f4",
                                    color: "#525252",
                                    cursor: "pointer",
                                  }}
                                >
                                  {f + 1}
                                </button>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                {objectList.length > 0 && !objectList.some((n) => isTargetName(n)) && (
                  <InlineNotification
                    kind="info"
                    lowContrast
                    hideCloseButton
                    subtitle="Mark at least one object as a target to enable gaze detection and ELAN export."
                    style={{ marginTop: "0.75rem", minBlockSize: "auto" }}
                  />
                )}
              </Tile>

              {/* Point mode */}
              <Tile style={{ marginBottom: "1rem" }}>
                <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "1rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <CircleFilled size={16} /> Point Mode
                </h3>
                <div style={{ display: "flex", gap: "0.5rem" }}>
                  <Button
                    kind={labelMode === 1 ? "primary" : "secondary"}
                    size="sm"
                    onClick={() => setLabelMode(1)}
                    renderIcon={AddAlt}
                    style={{ flex: 1 }}
                  >
                    Positive
                  </Button>
                  <Button
                    kind={labelMode === 0 ? "danger" : "secondary"}
                    size="sm"
                    onClick={() => setLabelMode(0)}
                    renderIcon={SubtractAlt}
                    style={{ flex: 1 }}
                  >
                    Negative
                  </Button>
                </div>
                <p style={{ fontSize: "0.75rem", color: "#525252", marginTop: "0.5rem" }}>
                  {labelMode === 1 ? "Click to mark object regions (green)" : "Click to exclude regions (red)"}
                </p>
                <div style={{ display: "flex", gap: "0.5rem", marginTop: "1rem" }}>
                  <Button
                    kind="ghost"
                    size="sm"
                    onClick={undoLastPoint}
                    disabled={!selectedObject || (frameObjects[selectedObject] || []).length === 0}
                    renderIcon={Undo}
                    style={{ flex: 1 }}
                  >
                    Undo
                  </Button>
                  <Button
                    kind="tertiary"
                    size="sm"
                    onClick={handleTestMask}
                    disabled={busy}
                    renderIcon={ZoomIn}
                    style={{ flex: 1 }}
                  >
                    Test Mask
                  </Button>
                </div>
              </Tile>

              {/* Suggest Frames */}
              <Tile style={{ marginBottom: "1rem" }}>
                {loadingSuggestions ? (
                  <InlineLoading
                    description="Analyzing frames for optimal suggestions..."
                  />
                ) : (
                  <Button
                    kind="tertiary"
                    size="sm"
                    renderIcon={Analytics}
                    onClick={handleSuggestFrames}
                    disabled={frames.length === 0}
                    style={{ width: "100%" }}
                  >
                    {suggestedFrames.length > 0
                      ? `View Suggested Frames (${suggestedFrames.length})`
                      : "Suggest Optimal Frames"}
                  </Button>
                )}
              </Tile>

              {/* Keyboard shortcuts */}
              <Tile style={{ marginBottom: "1rem" }}>
                <h3 style={{ fontSize: "0.875rem", fontWeight: 600, marginBottom: "0.75rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <Keyboard size={16} /> Shortcuts
                </h3>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.25rem", fontSize: "0.75rem" }}>
                  <div><Tag type="gray" size="sm">C</Tag> Focus name</div>
                  <div><Tag type="gray" size="sm">T</Tag> Test mask</div>
                  <div><Tag type="gray" size="sm">Ctrl+Z</Tag> Undo</div>
                  <div><Tag type="gray" size="sm">1</Tag> Positive</div>
                  <div><Tag type="gray" size="sm">2</Tag> Negative</div>
                  <div><Tag type="gray" size="sm">←→</Tag> Frames</div>
                </div>
              </Tile>

              {/* Status */}
              {status && (
                <InlineNotification
                  kind={statusKind}
                  title={status}
                  lowContrast
                  hideCloseButton
                  style={{ marginBottom: "1rem" }}
                />
              )}

              {/* Missing steps warning */}
              {missingSteps.length > 0 && (
                <InlineNotification
                  kind="warning"
                  title="Complete before continuing"
                  lowContrast
                  hideCloseButton
                >
                  <ul style={{ margin: "0.5rem 0 0 1rem", padding: 0 }}>
                    {missingSteps.map((step) => (
                      <li key={step} style={{ fontSize: "0.875rem" }}>{step}</li>
                    ))}
                  </ul>
                </InlineNotification>
              )}
              </div>
            </Column>
          </Grid>
        </div>
      </main>

      {/* Frame suggestion modal */}
      <Modal
        open={showSuggestions}
        onRequestClose={() => setShowSuggestions(false)}
        modalHeading="Suggested Optimal Frames"
        passiveModal
        size="lg"
      >
        <p style={{ fontSize: "0.875rem", color: "#525252", marginBottom: "1rem" }}>
          Select frame(s) to annotate. Higher scores indicate better quality and content.
        </p>
        {suggestedFrames.length > 0 ? (
          <>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: "1rem" }}>
              {suggestedFrames.map((frame) => {
                const isSelected = selectedSuggestions.has(frame.frame_index);
                const frameName = frames[frame.frame_index];
                const thumbUrl = frameName
                  ? `${API_BASE}/frames/thumbs/${sessionId}/${frameName}`
                  : "";

                return (
                  <label
                    key={frame.frame_index}
                    htmlFor={`suggest-check-${frame.frame_index}`}
                    style={{
                      border: `2px solid ${isSelected ? "#0f62fe" : "#e0e0e0"}`,
                      borderRadius: "4px",
                      overflow: "hidden",
                      cursor: "pointer",
                      backgroundColor: isSelected ? "#e5f6ff" : "#fff",
                      transition: "all 0.15s ease",
                      display: "block",
                    }}
                  >
                    <div style={{ padding: "0.75rem", display: "flex", alignItems: "center", gap: "0.75rem" }}>
                      <Checkbox
                        id={`suggest-check-${frame.frame_index}`}
                        checked={isSelected}
                        onChange={() => toggleSuggestionSelection(frame.frame_index)}
                        labelText={`Frame ${frame.frame_index + 1}`}
                        style={{ minWidth: 0 }}
                      />
                      <Tag type={frame.method === "dinov2" ? "green" : "blue"} size="sm">
                        {(frame.score * 100).toFixed(0)}%
                      </Tag>
                    </div>
                    {thumbUrl && (
                      <img
                        src={thumbUrl}
                        alt={`Frame ${frame.frame_index}`}
                        style={{ width: "100%", height: "120px", objectFit: "cover" }}
                      />
                    )}
                    <div style={{ padding: "0.5rem 0.75rem", fontSize: "0.75rem" }}>
                      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                        <span style={{ color: "#525252" }}>Sharpness: <strong>{frame.sharpness.toFixed(1)}</strong></span>
                        <span style={{ color: "#525252" }}>Brightness: <strong>{(frame.brightness * 100).toFixed(0)}%</strong></span>
                      </div>
                    </div>
                  </label>
                );
              })}
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "1.5rem", paddingTop: "1rem", borderTop: "1px solid #e0e0e0" }}>
              <span style={{ fontSize: "0.875rem", color: "#525252" }}>
                {selectedSuggestions.size > 0
                  ? `${selectedSuggestions.size} frame(s) selected`
                  : "Click frames to select"}
              </span>
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <Button
                  kind="secondary"
                  size="sm"
                  onClick={() => {
                    if (selectedSuggestions.size > 0) {
                      navigateToSuggestion(Math.min(...Array.from(selectedSuggestions)));
                    }
                  }}
                  disabled={selectedSuggestions.size === 0}
                >
                  Go to Selected
                </Button>
                <Button
                  kind="primary"
                  size="sm"
                  renderIcon={Checkmark}
                  onClick={() => {
                    if (selectedSuggestions.size > 0) {
                      const indices = Array.from(selectedSuggestions).sort((a, b) => a - b);
                      setCurrentIndex(indices[0]);
                      setShowSuggestions(false);
                      setStatus(`Ready to annotate ${indices.length} selected frame(s).`);
                      setStatusKind("success");
                    }
                  }}
                  disabled={selectedSuggestions.size === 0}
                >
                  Start Annotating ({selectedSuggestions.size})
                </Button>
              </div>
            </div>
          </>
        ) : (
          <p style={{ color: "#6f6f6f", textAlign: "center", padding: "2rem 0" }}>
            No frame suggestions available.
          </p>
        )}
      </Modal>
    </div>
  );
}
