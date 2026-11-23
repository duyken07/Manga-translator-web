import type {
  Box,
  Dialogue,
  FillItem,
  GrammarAnalysisResult,
  JobRecord,
  ModelStatusItem,
  PageDetail,
  PageSummary,
  ProjectSummary,
  RuntimeInfo,
  StrokeItem,
  UserSettings,
} from "./types";

// Held strictly in JS memory (never written to localStorage/sessionStorage)
let memoryToken = "";

export function setApiToken(token: string) {
  memoryToken = token.trim();
}

export function getApiToken(): string {
  return memoryToken;
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (memoryToken) {
    headers.set("Authorization", `Bearer ${memoryToken}`);
  }
  if (init.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(path, { ...init, headers });
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      message = data.detail || data.error || message;
    } catch {
      // ignore non-json error body
    }
    throw new ApiError(message, res.status);
  }
  if (res.status === 204) return undefined as T;
  const contentType = (res.headers.get("content-type") || "").toLowerCase();
  if (contentType.includes("application/json")) {
    return (await res.json()) as T;
  }
  const text = await res.text();
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new ApiError(
      `Expected JSON from ${path} but received ${contentType || "non-JSON"} response. Check FASTAPI_BACKEND_URL configuration.`,
      res.status || 502
    );
  }
}

export function buildAuthImageUrl(path: string): string {
  return path;
}

export const api = {
  runtime: () => apiFetch<RuntimeInfo>("/api/runtime"),
  health: () => apiFetch<{ status: string; proxy?: boolean }>("/api/health"),

  // Projects
  listProjects: async () => {
    const data = await apiFetch<ProjectSummary[]>("/api/projects");
    return Array.isArray(data) ? data : [];
  },
  createProject: (body: { name: string; source_lang: string; target_lang: string; work_title?: string }) =>
    apiFetch<ProjectSummary>("/api/projects", { method: "POST", body: JSON.stringify(body) }),
  getProject: (id: string) => apiFetch<ProjectSummary>(`/api/projects/${id}`),
  updateProject: (
    id: string,
    body: Partial<Pick<ProjectSummary, "name" | "source_lang" | "target_lang" | "work_title" | "glossary" | "settings">>
  ) => apiFetch<ProjectSummary>(`/api/projects/${id}`, { method: "PUT", body: JSON.stringify(body) }),
  deleteProject: (id: string) => apiFetch<void>(`/api/projects/${id}`, { method: "DELETE" }),
  findReplace: (
    projectId: string,
    body: { find: string; replace: string; field: "translated_text" | "original_text"; case_sensitive: boolean; dry_run?: boolean }
  ) =>
    apiFetch<{ matches: number; updated_dialogues: number; affected_page_ids: string[]; dry_run: boolean }>(
      `/api/projects/${projectId}/find-replace`,
      { method: "POST", body: JSON.stringify(body) }
    ),
  chat: (projectId: string, body: { message: string; character: string }) =>
    apiFetch<{ reply: string }>(`/api/projects/${projectId}/chat`, { method: "POST", body: JSON.stringify(body) }),

  // Pages
  listPages: async (projectId: string) => {
    const data = await apiFetch<PageSummary[]>(`/api/projects/${projectId}/pages`);
    return Array.isArray(data) ? data : [];
  },
  uploadPages: (projectId: string, files: File[]) => {
    const form = new FormData();
    for (const f of files) {
      form.append("files", f, f.name);
    }
    return apiFetch<{
      added: number;
      page_ids: string[];
      skipped_duplicates: string[];
      rejected: Array<{ file: string; reason: string }>;
    }>(`/api/projects/${projectId}/pages/upload`, { method: "POST", body: form });
  },
  getPage: (pageId: string) => apiFetch<PageDetail>(`/api/pages/${pageId}`),
  deletePage: (pageId: string) => apiFetch<void>(`/api/pages/${pageId}`, { method: "DELETE" }),
  saveStrokes: (pageId: string, strokes: StrokeItem[]) =>
    apiFetch<PageDetail>(`/api/pages/${pageId}/strokes`, { method: "PUT", body: JSON.stringify({ strokes }) }),
  saveFills: (pageId: string, fills: FillItem[]) =>
    apiFetch<PageDetail>(`/api/pages/${pageId}/fills`, { method: "PUT", body: JSON.stringify({ fills }) }),

  // Dialogues
  createDialogue: (
    pageId: string,
    body: { box_location: Box; original_text?: string; translated_text?: string; romanization?: string; redraw_only?: boolean; no_redraw?: boolean }
  ) => apiFetch<Dialogue>(`/api/pages/${pageId}/dialogues`, { method: "POST", body: JSON.stringify(body) }),
  patchDialogue: (pageId: string, dialogueId: number, body: Partial<Dialogue>) =>
    apiFetch<Dialogue>(`/api/pages/${pageId}/dialogues/${dialogueId}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  deleteDialogue: (pageId: string, dialogueId: number) =>
    apiFetch<void>(`/api/pages/${pageId}/dialogues/${dialogueId}`, { method: "DELETE" }),
  reorderDialogues: (pageId: string, order: number[]) =>
    apiFetch<Dialogue[]>(`/api/pages/${pageId}/dialogues/order`, {
      method: "PUT",
      body: JSON.stringify({ order }),
    }),

  // Translation & AI
  translateProject: (projectId: string, body: { page_ids?: string[]; keep_manual?: boolean | null } = {}) =>
    apiFetch<{ job_id: string }>(`/api/projects/${projectId}/translate`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  translatePage: (pageId: string, body: { keep_manual?: boolean | null } = {}) =>
    apiFetch<{ job_id: string }>(`/api/pages/${pageId}/translate`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  translateRegion: (pageId: string, body: { box: Box; add?: boolean }) =>
    apiFetch<Dialogue>(`/api/pages/${pageId}/translate-region`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  detectRegions: (pageId: string) =>
    apiFetch<{ detected: number; dialogues: Dialogue[] }>(`/api/pages/${pageId}/detect-regions`, {
      method: "POST",
    }),
  analyzeGrammar: (pageId: string, text: string) =>
    apiFetch<{ cached: boolean; analysis: GrammarAnalysisResult }>(`/api/pages/${pageId}/grammar`, {
      method: "POST",
      body: JSON.stringify({ text }),
    }),

  // Redraw (delegates to Cloud Run FastAPI when configured)
  requestRedraw: (pageId: string, body: { force_full?: boolean; relayout?: boolean } = {}) =>
    apiFetch<{ queued: boolean }>(`/api/pages/${pageId}/redraw`, { method: "POST", body: JSON.stringify(body) }),

  // Jobs
  listJobs: async (projectId: string) => {
    const data = await apiFetch<JobRecord[]>(`/api/projects/${projectId}/jobs`);
    return Array.isArray(data) ? data : [];
  },
  cancelJob: (jobId: string) => apiFetch<JobRecord>(`/api/jobs/${jobId}/cancel`, { method: "POST" }),
  retryJob: (jobId: string) => apiFetch<{ job_id: string }>(`/api/jobs/${jobId}/retry-failed`, { method: "POST" }),

  // Models & Settings
  getModelsStatus: async () => {
    const data = await apiFetch<{ models: ModelStatusItem[] }>("/api/models/status");
    return { models: Array.isArray(data?.models) ? data.models : [] };
  },
  downloadModel: (id: string) =>
    apiFetch<{ models: ModelStatusItem[] }>("/api/models/download", {
      method: "POST",
      body: JSON.stringify({ id }),
    }),
  loadModel: (id: string) =>
    apiFetch<{ job_id: string }>("/api/models/load", {
      method: "POST",
      body: JSON.stringify({ id }),
    }),
  getSettings: () => apiFetch<UserSettings>("/api/settings"),
  updateSettings: (patch: Partial<UserSettings>) =>
    apiFetch<UserSettings>("/api/settings", { method: "PUT", body: JSON.stringify(patch) }),

  downloadProjectExport: async (projectId: string, format: "json" | "txt", projectName: string) => {
    const headers = new Headers();
    if (memoryToken) {
      headers.set("Authorization", `Bearer ${memoryToken}`);
    }
    const res = await fetch(`/api/projects/${projectId}/export?format=${format}`, { headers });
    if (!res.ok) {
      throw new ApiError(`${res.status} ${res.statusText}`, res.status);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const stem = projectName.replace(/[^a-zA-Z0-9._-]/g, "_") || "project";
    a.download = `${stem}.${format}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  },
};
