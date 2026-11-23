export const DEFAULT_GEMINI_MODEL = "gemini-4-hydrogen";
export const FALLBACK_GEMINI_MODEL = "gemini-3.1-flash-lite";

export const PROJECT_OVERRIDABLE = [
  "source_lang",
  "target_lang",
  "translation_backend",
  "sequential_context_mode",
  "keep_honorifics",
  "enable_redraw",
] as const;

export interface ServerConfig {
  port: number;
  host: string;
  apiToken: string;
  fastapiBackendUrl: string;
  fastapiWrapperUrlDetected: boolean;
  corsOrigins: string[];
  maxUploadMb: number;
  maxImagePixels: number;
  translateConcurrency: number;
  geminiApiKey: string;
}

export function getServerConfig(): ServerConfig {
  const rawOrigins = process.env.MT_CORS_ORIGINS ?? "";
  const corsOrigins = rawOrigins
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  const rawBackend = (process.env.FASTAPI_BACKEND_URL || "").trim().replace(/\/+$/, "");
  let fastapiBackendUrl = "";
  let fastapiWrapperUrlDetected = false;
  if (rawBackend) {
    try {
      const parsed = new URL(rawBackend);
      if (parsed.protocol === "http:" || parsed.protocol === "https:") {
        if (
          (parsed.hostname === "huggingface.co" || parsed.hostname === "www.huggingface.co") &&
          parsed.pathname.startsWith("/spaces/")
        ) {
          fastapiWrapperUrlDetected = true;
        } else {
          fastapiBackendUrl = rawBackend;
        }
      }
    } catch {
      // Malformed/non-URL value ignored; standalone Node + Gemini mode remains active
    }
  }

  return {
    port: Number(process.env.PORT || process.env.MT_PORT || 3000),
    host: process.env.MT_HOST?.trim() || "0.0.0.0",
    apiToken: (process.env.MT_API_TOKEN || "").trim(),
    fastapiBackendUrl,
    fastapiWrapperUrlDetected,
    corsOrigins,
    maxUploadMb: Math.max(1, Number(process.env.MT_MAX_UPLOAD_MB || 40)),
    maxImagePixels: Math.max(1, Number(process.env.MT_MAX_IMAGE_PIXELS || 120_000_000)),
    translateConcurrency: Math.max(1, Number(process.env.MT_TRANSLATE_CONCURRENCY || 4)),
    geminiApiKey: (process.env.GEMINI_API_KEY || "").trim(),
  };
}
