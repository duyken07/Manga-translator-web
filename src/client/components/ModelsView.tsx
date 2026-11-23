import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Cpu, Download, HardDrive, Play } from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import type { ModelStatusItem } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Button } from "./ui/button";

export function ModelsView() {
  const qc = useQueryClient();

  const runtimeQuery = useQuery({
    queryKey: ["runtime"],
    queryFn: api.runtime,
  });

  const modelsQuery = useQuery({
    queryKey: ["models"],
    queryFn: api.getModelsStatus,
    refetchInterval: 4000,
  });

  const downloadMutation = useMutation({
    mutationFn: (id: string) => api.downloadModel(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["models"] });
      toast.success("Model download started");
    },
    onError: (err) => toast.error(err.message),
  });

  const loadMutation = useMutation({
    mutationFn: (id: string) => api.loadModel(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["models"] });
      toast.success("Model load queued");
    },
    onError: (err) => toast.error(err.message),
  });

  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    enabled: Boolean(runtimeQuery.data?.fastapi_configured),
    retry: false,
  });

  const models: ModelStatusItem[] = Array.isArray(modelsQuery.data?.models) ? modelsQuery.data.models : [];
  const fastapiConnected = runtimeQuery.data?.fastapi_configured ?? false;
  const fastapiUrlInvalid = runtimeQuery.data?.fastapi_url_invalid ?? false;
  const fastapiHfWrapper = runtimeQuery.data?.fastapi_hf_wrapper_url ?? false;

  return (
    <div className="mx-auto flex h-full w-full max-w-5xl flex-col gap-6 overflow-y-auto px-4 py-6 sm:px-6">
      <div className="border-b border-zinc-200 pb-5 dark:border-zinc-800">
        <h1 className="font-display text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
          AI Model Manager
        </h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          Hugging Face model weights for bubble detection, Manga-OCR, Koharu segmentation, and LaMa inpainting.
        </p>
      </div>

      {fastapiHfWrapper && (
        <div className="rounded-xl border border-amber-300 bg-amber-50/70 p-4 text-sm text-amber-950 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-100">
          <div className="font-semibold">
            Hugging Face Space Web Wrapper URL Detected
          </div>
          <p className="mt-1 text-xs text-amber-900/90 dark:text-amber-200/90">
            <code className="font-mono">FASTAPI_BACKEND_URL</code> points to a{" "}
            <code className="font-mono">huggingface.co/spaces/...</code> web page, which returns HTML instead of FastAPI JSON. Use your Space&apos;s direct API URL instead (for example{" "}
            <code className="font-mono">https://&lt;username&gt;-&lt;space-name&gt;.hf.space</code> from Space menu{" "}
            <code className="font-mono">⋮ → Embed this Space</code>). Node.js is running safely in standalone Gemini mode meanwhile.
          </p>
        </div>
      )}

      {fastapiUrlInvalid && (
        <div className="rounded-xl border border-amber-300 bg-amber-50/70 p-4 text-sm text-amber-950 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-100">
          <div className="font-semibold">
            Invalid FASTAPI_BACKEND_URL Value Detected
          </div>
          <p className="mt-1 text-xs text-amber-900/90 dark:text-amber-200/90">
            <code className="font-mono">FASTAPI_BACKEND_URL</code> is currently set to a value that is not a valid{" "}
            <code className="font-mono">http://</code> or <code className="font-mono">https://</code> URL (check that you did not paste an API token into this field). Node.js has safely fallen back to standalone Gemini Vision mode. Set{" "}
            <code className="font-mono">FASTAPI_BACKEND_URL</code> to your deployed Cloud Run or <code className="font-mono">*.hf.space</code> URL and restart the server.
          </p>
        </div>
      )}

      {!fastapiConnected && !fastapiUrlInvalid && !fastapiHfWrapper && (
        <div className="rounded-xl border border-zinc-200 bg-white p-4 text-sm text-zinc-700 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300">
          <div className="font-semibold text-zinc-900 dark:text-zinc-100">
            Python Cloud Run Backend Not Connected
          </div>
          <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
            Node.js is running in standalone mode using server-side Gemini Vision translation, Konva canvas editing, and multi-layer PSD/PNG/JPG/ZIP export. Deploy the Python FastAPI service (<code className="font-mono">backend/</code> from <code className="font-mono">duyken07/Manga-translator-web</code>) to Google Cloud Run and set{" "}
            <code className="font-mono">FASTAPI_BACKEND_URL</code> to enable local Hugging Face weights (YOLO, Koharu, Manga-OCR) and LaMa inpainting.
          </p>
        </div>
      )}

      {fastapiConnected && healthQuery.isError && (
        <div className="rounded-xl border border-rose-300 bg-rose-50/70 p-4 text-sm text-rose-900 dark:border-rose-800 dark:bg-rose-950/30 dark:text-rose-100">
          <div className="font-semibold">
            Cloud Run Backend Configured But Unreachable
          </div>
          <p className="mt-1 text-xs">
            {healthQuery.error instanceof Error ? healthQuery.error.message : "Failed to reach upstream FastAPI /api/health."}
          </p>
        </div>
      )}

      <div className="divide-y divide-zinc-200 rounded-xl border border-zinc-200 bg-white dark:divide-zinc-800 dark:border-zinc-800 dark:bg-zinc-900">
        {models.map((m) => (
          <div
            key={m.id}
            className="flex flex-col gap-4 p-5 sm:flex-row sm:items-center sm:justify-between"
          >
            <div className="min-w-0 space-y-1">
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4 text-zinc-400 shrink-0" />
                <span className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                  {m.label}
                </span>
                {m.optional && (
                  <span className="text-xs text-zinc-500 dark:text-zinc-400">· Optional</span>
                )}
              </div>
              <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-500 dark:text-zinc-400">
                <span className="font-mono">{m.repo}</span>
                <span aria-hidden="true">·</span>
                <span className="font-mono">{m.file}</span>
                <span aria-hidden="true">·</span>
                <span
                  className={cn(
                    "font-medium",
                    m.state === "ready" && "text-emerald-600 dark:text-emerald-400",
                    m.state === "error" && "text-rose-600 dark:text-rose-400",
                    (m.state === "downloading" || m.state === "loading") &&
                      "text-blue-600 dark:text-blue-400"
                  )}
                >
                  {m.state === "not_downloaded"
                    ? "Not downloaded"
                    : m.state === "downloaded"
                      ? "Downloaded"
                      : m.state === "downloading"
                        ? `Downloading ${m.progress !== null ? `${Math.round(m.progress * 100)}%` : ""}`
                        : m.state === "loading"
                          ? "Loading"
                          : m.state === "ready"
                            ? "Ready"
                            : "Error"}
                </span>
              </div>
              {m.error && (
                <p className="text-xs text-rose-600 dark:text-rose-400">{m.error}</p>
              )}
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <Button
                size="sm"
                variant="outline"
                disabled={
                  !fastapiConnected ||
                  m.state === "downloading" ||
                  m.state === "downloaded" ||
                  m.state === "ready" ||
                  downloadMutation.isPending
                }
                onClick={() => downloadMutation.mutate(m.id)}
              >
                <Download className="h-3.5 w-3.5" />
                Download
              </Button>
              <Button
                size="sm"
                disabled={
                  !fastapiConnected ||
                  (m.state !== "downloaded" && m.state !== "error") ||
                  loadMutation.isPending
                }
                onClick={() => loadMutation.mutate(m.id)}
              >
                <Play className="h-3.5 w-3.5" />
                Load into RAM
              </Button>
            </div>
          </div>
        ))}

        {models.length === 0 && (
          <div className="flex flex-col items-center justify-center p-10 text-center text-sm text-zinc-500">
            <HardDrive className="h-7 w-7 text-zinc-400" />
            <span className="mt-2">Loading model registry...</span>
          </div>
        )}
      </div>
    </div>
  );
}
