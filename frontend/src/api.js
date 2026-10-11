export const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

// Helper to extract error detail from API responses
async function getErrorDetail(res, fallback) {
  try {
    const data = await res.json();
    return data.detail || fallback;
  } catch {
    return fallback;
  }
}

export async function createSession(name) {
  const res = await fetch(`${API_BASE}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) {
    throw new Error("Failed to create session");
  }
  return res.json();
}

export async function uploadVideo(sessionId, file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/uploads/video?session_id=${encodeURIComponent(sessionId)}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    throw new Error("Failed to upload video");
  }
  return res.json();
}

export async function createSampleClip(sessionId, durationSeconds = 10) {
  const res = await fetch(`${API_BASE}/uploads/sample`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, duration_seconds: durationSeconds }),
  });
  if (!res.ok) {
    throw new Error("Failed to create sample clip");
  }
  return res.json();
}

export async function updateConfig(payload) {
  const res = await fetch(`${API_BASE}/config`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error("Failed to update config");
  }
  return res.json();
}

export async function extractFrames(sessionId, quality = 2, startTime = null, endTime = null) {
  const payload = { session_id: sessionId, quality };
  if (startTime !== null && startTime !== "") payload.start_time = parseFloat(startTime);
  if (endTime !== null && endTime !== "") payload.end_time = parseFloat(endTime);
  const res = await fetch(`${API_BASE}/frames/extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error("Failed to extract frames");
  }
  return res.json();
}

export async function listFrames(sessionId) {
  const res = await fetch(`${API_BASE}/frames/list/${sessionId}`);
  if (!res.ok) {
    throw new Error("Failed to load frames");
  }
  return res.json();
}

export async function detectGreyStart(sessionId, maxFrames = 300) {
  const res = await fetch(`${API_BASE}/frames/detect-grey-start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, max_frames: maxFrames }),
  });
  if (!res.ok) {
    throw new Error("Failed to detect grey frames");
  }
  return res.json();
}

export async function submitAnnotation(payload) {
  const res = await fetch(`${API_BASE}/annotation/points`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const detail = await getErrorDetail(res, "Failed to save annotation points");
    throw new Error(detail);
  }
  return res.json();
}

export async function testMask(payload) {
  const res = await fetch(`${API_BASE}/annotation/test-mask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const detail = await getErrorDetail(res, "Failed to test mask");
    throw new Error(detail);
  }
  return res.json();
}

export async function deleteAnnotationObject(sessionId, objectName) {
  const res = await fetch(`${API_BASE}/annotation/object/${sessionId}/${encodeURIComponent(objectName)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const detail = await getErrorDetail(res, "Failed to delete object");
    throw new Error(detail);
  }
  return res.json();
}

export async function fetchFrameAnnotations(sessionId, frameIndex) {
  const res = await fetch(`${API_BASE}/annotation/frames/${sessionId}/${frameIndex}`);
  if (!res.ok) {
    throw new Error("Failed to load annotations");
  }
  return res.json();
}

export async function fetchSessionObjects(sessionId) {
  const res = await fetch(`${API_BASE}/annotation/objects/${sessionId}`);
  if (!res.ok) {
    throw new Error("Failed to load object list");
  }
  return res.json();
}

export async function startProcessing(sessionId) {
  const res = await fetch(`${API_BASE}/processing/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) {
    const detail = await getErrorDetail(res, "Failed to start processing");
    throw new Error(detail);
  }
  return res.json();
}

export async function fetchProcessingStatus(sessionId) {
  const res = await fetch(`${API_BASE}/processing/status/${sessionId}`);
  if (!res.ok) {
    throw new Error("Failed to fetch processing status");
  }
  return res.json();
}

export function getProcessingPreviewUrl(sessionId) {
  return `${API_BASE}/processing/preview/${sessionId}`;
}

export async function fetchResults(sessionId) {
  const res = await fetch(`${API_BASE}/results/${sessionId}`);
  if (!res.ok) {
    throw new Error("Failed to fetch results");
  }
  return res.json();
}

export async function fetchModels() {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) {
    throw new Error("Failed to fetch models");
  }
  return res.json();
}

export function getResultDownloadUrl(sessionId, kind) {
  const params = new URLSearchParams({ kind });
  return `${API_BASE}/results/download/${sessionId}?${params.toString()}`;
}

export function getResultDownloadAllUrl(sessionId) {
  return `${API_BASE}/results/download-all/${sessionId}`;
}

export async function runDiagnostics() {
  const res = await fetch(`${API_BASE}/diagnostics`);
  if (!res.ok) {
    throw new Error("Failed to run diagnostics");
  }
  return res.json();
}

export async function suggestFrames(sessionId, topK = 7, useDinov2 = true) {
  const params = new URLSearchParams({ top_k: topK, use_dinov2: useDinov2 });
  const res = await fetch(`${API_BASE}/frames/suggest/${sessionId}?${params.toString()}`);
  if (!res.ok) {
    const detail = await getErrorDetail(res, "Failed to suggest frames");
    throw new Error(detail);
  }
  return res.json();
}
