import { Router } from "express";
import { modelStatuses, settingsView, userSettings } from "../store.js";

export const settingsRouter = Router();
export const modelsRouter = Router();

const FORBIDDEN_SETTINGS_KEYS = new Set(["gemini_model", "has_api_key", "fonts", "api_key", "fastapi_connected"]);
const ALLOWED_SETTINGS_KEYS = new Set(Object.keys(userSettings));

settingsRouter.get("/", (_req, res) => {
  res.json(settingsView());
});

settingsRouter.put("/", (req, res) => {
  const patch = req.body ?? {};
  if (!patch || typeof patch !== "object" || Array.isArray(patch)) {
    res.status(422).json({ detail: "Body must be a JSON object" });
    return;
  }
  const forbidden = Object.keys(patch).filter((k) => FORBIDDEN_SETTINGS_KEYS.has(k));
  if (forbidden.length > 0) {
    res.status(422).json({ detail: `Cannot set read-only fields: ${forbidden.sort().join(", ")}` });
    return;
  }
  const unknown = Object.keys(patch).filter((k) => !ALLOWED_SETTINGS_KEYS.has(k));
  if (unknown.length > 0) {
    res.status(422).json({ detail: `Unknown settings fields: ${unknown.join(", ")}` });
    return;
  }
  Object.assign(userSettings, patch);
  res.json(settingsView());
});

// ---------------------------------------------------------------------------
// Models API (mirrors FastAPI contract; ML downloads/loads run on Cloud Run Python backend)
// ---------------------------------------------------------------------------
modelsRouter.get("/status", (_req, res) => {
  res.json({ models: modelStatuses });
});

modelsRouter.post("/download", (req, res) => {
  const id = String(req.body?.id ?? "");
  if (!modelStatuses.some((m) => m.id === id)) {
    res.status(404).json({ detail: `Unknown model: ${id}` });
    return;
  }
  res.status(501).json({
    detail: "Model weight downloads require FASTAPI_BACKEND_URL pointing to the Python Cloud Run backend.",
  });
});

modelsRouter.post("/load", (req, res) => {
  const id = String(req.body?.id ?? "");
  if (!modelStatuses.some((m) => m.id === id)) {
    res.status(404).json({ detail: `Unknown model: ${id}` });
    return;
  }
  res.status(501).json({
    detail: "PyTorch model loading requires FASTAPI_BACKEND_URL pointing to the Python Cloud Run backend.",
  });
});
