import crypto from "node:crypto";
import type { NextFunction, Request, Response } from "express";
import type { WebSocket } from "ws";
import {
  DEFAULT_GEMINI_MODEL,
  FALLBACK_GEMINI_MODEL,
  getServerConfig,
  PROJECT_OVERRIDABLE,
} from "./config.js";
import type {
  Box,
  Dialogue,
  JobPageState,
  JobRecord,
  JobStatus,
  ModelStatusItem,
  Page,
  Project,
  UserSettings,
} from "./types.js";

export class DomainError extends Error {
  status: number;
  constructor(message: string, status = 400) {
    super(message);
    this.status = status;
  }
}

export function newId(): string {
  return crypto.randomBytes(16).toString("hex");
}

export function nowIso(): string {
  return new Date().toISOString();
}

export function naturalSortCompare(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
}

// ---------------------------------------------------------------------------
// Auth & Security
// ---------------------------------------------------------------------------
export function isAuthRequired(): boolean {
  return Boolean(getServerConfig().apiToken);
}

export function checkToken(candidate: string | undefined): boolean {
  const configured = getServerConfig().apiToken;
  if (!configured) return true;
  if (!candidate) return false;
  const expected = Buffer.from(configured, "utf8");
  const actual = Buffer.from(candidate, "utf8");
  if (expected.length !== actual.length) return false;
  return crypto.timingSafeEqual(expected, actual);
}

export function requireToken(req: Request, res: Response, next: NextFunction): void {
  if (!isAuthRequired()) {
    next();
    return;
  }
  const auth = req.header("authorization");
  let candidate = req.header("x-api-token") ?? undefined;
  if (auth && auth.toLowerCase().startsWith("bearer ")) {
    candidate = auth.slice(7).trim();
  }
  if (!checkToken(candidate)) {
    res.setHeader("WWW-Authenticate", "Bearer");
    res.status(401).json({ detail: "Missing or invalid API token" });
    return;
  }
  next();
}

const BAD_FILENAME_CHARS = /[\x00-\x1f<>:"|?*]/g;
const WINDOWS_RESERVED = new Set([
  "CON",
  "PRN",
  "AUX",
  "NUL",
  ...Array.from({ length: 9 }, (_, i) => `COM${i + 1}`),
  ...Array.from({ length: 9 }, (_, i) => `LPT${i + 1}`),
]);

export function safeFilename(name: string, maxLen = 120): string {
  let base = (name || "").replace(/\\/g, "/").split("/").pop()?.trim().replace(/^\.+|\.+$/g, "") ?? "";
  base = base.replace(BAD_FILENAME_CHARS, "_");
  if (!base || base === "." || base === "..") {
    throw new DomainError("Invalid filename", 400);
  }
  if (base.length > maxLen) {
    const dotIdx = base.lastIndexOf(".");
    if (dotIdx > 0) {
      const ext = base.slice(dotIdx + 1);
      const stem = base.slice(0, dotIdx);
      base = `${stem.slice(0, Math.max(1, maxLen - ext.length - 1))}.${ext}`;
    } else {
      base = base.slice(0, maxLen);
    }
  }
  const stemUpper = base.split(".")[0].toUpperCase();
  if (WINDOWS_RESERVED.has(stemUpper)) {
    base = `_${base}`;
  }
  return base;
}

export function inspectImageBuffer(buf: Buffer): { mime: string; width: number; height: number } {
  if (buf.length < 12) {
    throw new DomainError("Empty or corrupted image file", 400);
  }
  // PNG
  if (buf.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))) {
    if (buf.length < 24) throw new DomainError("Invalid PNG header", 400);
    return { mime: "image/png", width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) };
  }
  // JPEG
  if (buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff) {
    let offset = 2;
    while (offset + 4 < buf.length) {
      if (buf[offset] !== 0xff) break;
      const marker = buf[offset + 1];
      offset += 2;
      if (marker === 0xd9 || marker === 0xda) break;
      const segLen = buf.readUInt16BE(offset);
      if (marker >= 0xc0 && marker <= 0xc3 && offset + 7 < buf.length) {
        return {
          mime: "image/jpeg",
          height: buf.readUInt16BE(offset + 3),
          width: buf.readUInt16BE(offset + 5),
        };
      }
      offset += segLen;
    }
    return { mime: "image/jpeg", width: 0, height: 0 };
  }
  // WEBP
  if (buf.subarray(0, 4).toString("ascii") === "RIFF" && buf.subarray(8, 12).toString("ascii") === "WEBP") {
    let width = 0;
    let height = 0;
    if (buf.length >= 30 && buf.subarray(12, 16).toString("ascii") === "VP8X") {
      width = 1 + (buf[24] | (buf[25] << 8) | (buf[26] << 16));
      height = 1 + (buf[27] | (buf[28] << 8) | (buf[29] << 16));
    }
    return { mime: "image/webp", width, height };
  }
  // BMP
  if (buf[0] === 0x42 && buf[1] === 0x4d && buf.length >= 26) {
    const width = buf.readInt32LE(18);
    const height = Math.abs(buf.readInt32LE(22));
    return { mime: "image/bmp", width, height };
  }
  throw new DomainError("Unsupported image format (PNG, JPEG, WEBP, BMP only)", 400);
}

// ---------------------------------------------------------------------------
// In-memory store (used when FASTAPI_BACKEND_URL is not configured)
// ---------------------------------------------------------------------------
export const projectsMap = new Map<string, Project>();
export const pagesMap = new Map<string, Page>();
export const jobsMap = new Map<string, JobRecord>();

export const userSettings: UserSettings = {
  source_lang: "ja",
  target_lang: "vi",
  concurrency_limit: 4,
  translation_backend: "yolo_gemini_vision",
  sequential_context_mode: true,
  keep_honorifics: false,
  enable_stray_text_removal: false,
  enable_redraw: true,
  paint_auto_redraw: false,
  paint_direct: false,
  lama_device: "cpu",
  lama_threads: 0,
  redraw_font: "",
  redraw_text_color: "20,20,20",
  redraw_line_spacing: 1.06,
  theme: "system",
  use_koharu: false,
  koharu_imgsz: 0,
  koharu_erase_sfx: false,
  koharu_use_yolo: true,
  ocr_force_cpu: false,
  ocr_gemini_rescue: true,
  ocr_low_conf: 0.6,
  gemini_thinking_level: "low",
  custom_glossary: {},
};

export const modelStatuses: ModelStatusItem[] = [
  {
    id: "yolo-bubble",
    label: "YOLO Speech Bubble Detector",
    repo: "ogkalu/comic-speech-bubble-detector-yolov8m",
    state: "not_downloaded",
    progress: null,
    error: "",
    version: "",
    optional: false,
    file: "comic-speech-bubble-detector.pt",
  },
  {
    id: "lama-manga",
    label: "LaMa Manga Inpainting",
    repo: "mayocream/lama-manga",
    state: "not_downloaded",
    progress: null,
    error: "",
    version: "",
    optional: false,
    file: "lama-manga.safetensors",
  },
  {
    id: "manga-ocr",
    label: "Manga-OCR",
    repo: "kha-white/manga-ocr-base",
    state: "not_downloaded",
    progress: null,
    error: "",
    version: "",
    optional: false,
    file: "manga-ocr-base",
  },
  {
    id: "koharu-yolo26s",
    label: "Koharu Detector (Bubbles + Text + SFX)",
    repo: "mayocream/koharu-yolo26s",
    state: "not_downloaded",
    progress: null,
    error: "",
    version: "",
    optional: true,
    file: "koharu-yolo26s",
  },
];

export function getEffectiveApiKey(): string | null {
  const envKey = getServerConfig().geminiApiKey;
  return envKey || null;
}

export function settingsView(): Record<string, unknown> {
  const cfg = getServerConfig();
  return {
    ...userSettings,
    gemini_model: DEFAULT_GEMINI_MODEL,
    gemini_fallback_model: FALLBACK_GEMINI_MODEL,
    has_api_key: Boolean(getEffectiveApiKey()),
    fastapi_connected: Boolean(cfg.fastapiBackendUrl),
    fonts: ["Plus Jakarta Sans", "DM Sans", "WildWords Style (Web)", "Comic Neue", "Noto Sans"],
  };
}

export function effectiveProjectSettings(project: Project): UserSettings {
  const merged: Record<string, unknown> = { ...userSettings };
  for (const key of PROJECT_OVERRIDABLE) {
    const val = project.settings[key];
    if (val !== undefined && val !== null) {
      merged[key] = val;
    }
  }
  return merged as unknown as UserSettings;
}

// ---------------------------------------------------------------------------
// Serializers (compatible with Desktop JSON & FastAPI DTOs)
// ---------------------------------------------------------------------------
export function projectOut(p: Project) {
  return {
    id: p.id,
    name: p.name,
    source_lang: p.source_lang,
    target_lang: p.target_lang,
    work_title: p.work_title,
    page_count: p.page_ids.length,
    glossary: p.glossary,
    settings: p.settings,
    created_at: p.created_at,
    updated_at: p.updated_at,
  };
}

export function dialogueOut(d: Dialogue) {
  return {
    ...d,
    display_original: d.original_text,
  };
}

export function pageSummary(p: Page) {
  return {
    id: p.id,
    page_name: p.page_name,
    status: p.status,
    error_message: p.error_message,
    width: p.width,
    height: p.height,
    dialogue_count: p.dialogues.length,
    edited_count: p.dialogues.filter((d) => d.edited).length,
    redraw_status: p.redraw_status,
    direct_edited: p.direct_edited,
  };
}

export function pageDetail(p: Page) {
  return {
    ...pageSummary(p),
    dialogues: p.dialogues.map(dialogueOut),
    strokes: p.strokes,
    fills: p.fills,
    redraw: {
      status: p.redraw_status,
      message: p.redraw_message,
      has_image: false,
      has_layers: false,
      layout_boxes: p.layout_boxes,
      layout_info: p.layout_info,
    },
  };
}

export function cleanBox(v: unknown, field = "box"): Box {
  if (!Array.isArray(v) || v.length !== 4) {
    throw new DomainError(`${field} must be [x, y, w, h]`, 400);
  }
  const nums = v.map((n) => Number(n));
  if (nums.some((n) => Number.isNaN(n))) {
    throw new DomainError(`${field} contains non-numeric values`, 400);
  }
  const [x, y, w, h] = nums;
  if (w <= 0 || h <= 0) {
    throw new DomainError(`${field}: width and height must be > 0`, 400);
  }
  return [Math.round(x), Math.round(y), Math.round(w), Math.round(h)];
}

// ---------------------------------------------------------------------------
// EventHub & JobManager (WebSocket broadcast)
// ---------------------------------------------------------------------------
interface BufferedEvent {
  seq: number;
  payload: Record<string, unknown>;
}

class EventHub {
  private seq = 0;
  private subscribers = new Map<string, Set<WebSocket>>();
  private history = new Map<string, BufferedEvent[]>();

  subscribe(projectId: string, ws: WebSocket, since = 0): void {
    let set = this.subscribers.get(projectId);
    if (!set) {
      set = new Set();
      this.subscribers.set(projectId, set);
    }
    set.add(ws);

    const recent = (this.history.get(projectId) || []).filter((e) => e.seq > since);
    for (const ev of recent) {
      if (ws.readyState === ws.OPEN) {
        ws.send(JSON.stringify({ ...ev.payload, seq: ev.seq }));
      }
    }
  }

  unsubscribe(projectId: string, ws: WebSocket): void {
    const set = this.subscribers.get(projectId);
    if (!set) return;
    set.delete(ws);
    if (set.size === 0) {
      this.subscribers.delete(projectId);
    }
  }

  publish(projectId: string, payload: Record<string, unknown>): void {
    this.seq += 1;
    const seq = this.seq;
    const buf = this.history.get(projectId) || [];
    buf.push({ seq, payload });
    if (buf.length > 200) buf.shift();
    this.history.set(projectId, buf);

    const msg = JSON.stringify({ ...payload, seq });
    const targets = [
      ...(this.subscribers.get(projectId) || []),
      ...(projectId === "*" ? [] : this.subscribers.get("*") || []),
    ];
    for (const ws of targets) {
      if (ws.readyState === ws.OPEN) {
        ws.send(msg);
      }
    }
  }
}

export const eventHub = new EventHub();

const FINAL_JOB_STATUSES = new Set<JobStatus>(["completed", "failed", "cancelled"]);

export const jobManager = {
  create(
    projectId: string | null,
    kind: string,
    total = 0,
    params: Record<string, unknown> = {},
    pageIds: string[] = []
  ): JobRecord {
    const pages: Record<string, JobPageState> = {};
    for (const pid of pageIds) {
      pages[pid] = { status: "pending", message: "" };
    }
    const job: JobRecord = {
      id: newId(),
      project_id: projectId,
      kind,
      status: "pending",
      total,
      done: 0,
      failed: 0,
      message: "",
      pages,
      params,
      created_at: nowIso(),
      finished_at: null,
    };
    jobsMap.set(job.id, job);
    eventHub.publish(projectId || "*", { type: "job", job: { ...job } });
    return job;
  },

  get(jobId: string): JobRecord | undefined {
    return jobsMap.get(jobId);
  },

  list(projectId?: string, limit = 50): JobRecord[] {
    let items = Array.from(jobsMap.values());
    if (projectId) {
      items = items.filter((j) => j.project_id === projectId);
    }
    items.sort((a, b) => b.created_at.localeCompare(a.created_at));
    return items.slice(0, limit);
  },

  start(jobId: string, message = ""): void {
    const job = jobsMap.get(jobId);
    if (!job || FINAL_JOB_STATUSES.has(job.status)) return;
    job.status = "processing";
    if (message) job.message = message;
    eventHub.publish(job.project_id || "*", { type: "job", job: { ...job } });
  },

  updatePage(jobId: string, pageId: string, status: JobStatus, message = ""): void {
    const job = jobsMap.get(jobId);
    if (!job || FINAL_JOB_STATUSES.has(job.status)) return;
    const prev = job.pages[pageId]?.status;
    job.pages[pageId] = { status, message };
    if ((status === "completed" || status === "failed") && prev !== "completed" && prev !== "failed") {
      job.done += 1;
      if (status === "failed") job.failed += 1;
    }
    eventHub.publish(job.project_id || "*", { type: "job", job: { ...job } });
  },

  finish(jobId: string, status: JobStatus = "completed", message = ""): void {
    const job = jobsMap.get(jobId);
    if (!job || FINAL_JOB_STATUSES.has(job.status)) return;
    job.status = status;
    if (message) job.message = message;
    job.finished_at = nowIso();
    eventHub.publish(job.project_id || "*", { type: "job", job: { ...job } });
  },

  cancel(jobId: string): boolean {
    const job = jobsMap.get(jobId);
    if (!job || FINAL_JOB_STATUSES.has(job.status)) return false;
    job.status = "cancelled";
    job.finished_at = nowIso();
    for (const [pid, state] of Object.entries(job.pages)) {
      if (state.status !== "completed" && state.status !== "failed") {
        job.pages[pid] = { ...state, status: "cancelled" };
      }
    }
    eventHub.publish(job.project_id || "*", { type: "job", job: { ...job } });
    return true;
  },
};
