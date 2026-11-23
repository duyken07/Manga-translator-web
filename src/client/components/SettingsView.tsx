import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { api } from "@/lib/api";
import type { UserSettings } from "@/lib/types";
import { cn } from "@/lib/utils";
import { type ThemeMode, useEditorStore } from "@/store/editorStore";
import { Input } from "./ui/input";
import { Select } from "./ui/select";
import { Switch } from "./ui/switch";

export function SettingsView() {
  const qc = useQueryClient();
  const { theme, setTheme } = useEditorStore();

  const runtimeQuery = useQuery({
    queryKey: ["runtime"],
    queryFn: api.runtime,
  });

  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  });

  const updateMutation = useMutation({
    mutationFn: (patch: Partial<UserSettings>) => api.updateSettings(patch),
    onSuccess: (updated) => {
      qc.setQueryData(["settings"], updated);
      toast.success("Settings saved");
    },
    onError: (err) => toast.error(err.message),
  });

  const s = settingsQuery.data;
  const runtime = runtimeQuery.data;

  const updateField = <K extends keyof UserSettings>(key: K, value: UserSettings[K]) => {
    updateMutation.mutate({ [key]: value });
  };

  return (
    <div className="mx-auto flex h-full w-full max-w-4xl flex-col gap-6 overflow-y-auto px-4 py-6 sm:px-6">
      <div className="border-b border-zinc-200 pb-5 dark:border-zinc-800">
        <h1 className="font-display text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
          Settings
        </h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          Configure appearance, translation pipeline behavior, OCR thresholds, and redraw defaults.
        </p>
      </div>

      {/* Theme Selector */}
      <div className="rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">Appearance</h2>
        <div className="mt-3 grid grid-cols-3 gap-2 rounded-lg bg-zinc-100 p-1 dark:bg-zinc-950">
          {(
            [
              { id: "system", label: "System" },
              { id: "light", label: "Light" },
              { id: "dark", label: "Dark" },
            ] as const
          ).map((opt) => (
            <button
              key={opt.id}
              type="button"
              onClick={() => {
                setTheme(opt.id as ThemeMode);
                if (s) updateField("theme", opt.id);
              }}
              className={cn(
                "rounded-md py-2 text-xs font-medium transition-colors",
                theme === opt.id
                  ? "bg-white text-zinc-900 shadow-xs dark:bg-zinc-800 dark:text-zinc-100"
                  : "text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
              )}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Runtime Integration Status (read-only, no secret inputs) */}
      {runtime && (
        <div className="rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
            Server Integration Status
          </h2>
          <div className="mt-3 grid grid-cols-1 gap-3 text-xs sm:grid-cols-3">
            <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <div className="text-zinc-500 dark:text-zinc-400">Execution Mode</div>
              <div className="mt-1 font-medium text-zinc-900 dark:text-zinc-100">
                {runtime.fastapi_configured
                  ? "Cloud Run / HF FastAPI Proxy"
                  : runtime.fastapi_hf_wrapper_url
                    ? "Node.js Standalone (Use *.hf.space URL)"
                    : runtime.fastapi_url_invalid
                      ? "Node.js Standalone (Invalid Proxy URL)"
                      : "Node.js + Gemini Standalone"}
              </div>
            </div>
            <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <div className="text-zinc-500 dark:text-zinc-400">Server Gemini Key</div>
              <div className="mt-1 font-medium text-zinc-900 dark:text-zinc-100">
                {runtime.has_gemini_key ? "Configured (Server Secret)" : "Not Configured"}
              </div>
            </div>
            <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <div className="text-zinc-500 dark:text-zinc-400">API Auth</div>
              <div className="mt-1 font-medium text-zinc-900 dark:text-zinc-100">
                {runtime.auth_required ? "Bearer Token Enforced" : "Open Local Preview"}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Pipeline & AI Settings */}
      {s && (
        <div className="space-y-6 rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
            Translation &amp; OCR Pipeline
          </h2>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <label className="mb-1.5 block text-xs font-medium text-zinc-500">
                Default Backend Mode
              </label>
              <Select
                value={s.translation_backend ?? "yolo_gemini_vision"}
                onChange={(e) =>
                  updateField(
                    "translation_backend",
                    e.target.value as UserSettings["translation_backend"]
                  )
                }
              >
                <option value="yolo_ocr_gemini">Mode A · YOLO + Manga-OCR + Gemini</option>
                <option value="yolo_gemini_vision">Mode B · YOLO + Gemini Vision</option>
              </Select>
            </div>

            <div>
              <label className="mb-1.5 block text-xs font-medium text-zinc-500">
                Gemini Thinking Level
              </label>
              <Select
                value={s.gemini_thinking_level ?? "low"}
                onChange={(e) =>
                  updateField(
                    "gemini_thinking_level",
                    e.target.value as UserSettings["gemini_thinking_level"]
                  )
                }
              >
                <option value="minimal">Minimal</option>
                <option value="low">Low (Recommended)</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </Select>
            </div>

            <div>
              <label className="mb-1.5 block text-xs font-medium text-zinc-500">
                Concurrency Limit
              </label>
              <Input
                type="number"
                min={1}
                max={16}
                value={s.concurrency_limit ?? 4}
                onChange={(e) =>
                  updateField("concurrency_limit", Math.max(1, Number(e.target.value) || 4))
                }
              />
            </div>

            <div>
              <label className="mb-1.5 block text-xs font-medium text-zinc-500">
                OCR Low-Confidence Rescue Threshold
              </label>
              <Input
                type="number"
                step="0.05"
                min={0}
                max={1}
                value={s.ocr_low_conf ?? 0.6}
                onChange={(e) => updateField("ocr_low_conf", Number(e.target.value) || 0.6)}
              />
            </div>

            <div>
              <label className="mb-1.5 block text-xs font-medium text-zinc-500">
                LaMa Device (FastAPI Backend)
              </label>
              <Select
                value={s.lama_device ?? "cpu"}
                onChange={(e) =>
                  updateField("lama_device", e.target.value as UserSettings["lama_device"])
                }
              >
                <option value="cpu">CPU (Default)</option>
                <option value="cuda">CUDA GPU</option>
                <option value="mps">Apple MPS</option>
              </Select>
            </div>

            <div>
              <label className="mb-1.5 block text-xs font-medium text-zinc-500">
                Default Line Spacing
              </label>
              <Input
                type="number"
                step="0.05"
                min={0.7}
                max={2.5}
                value={s.redraw_line_spacing ?? 1.06}
                onChange={(e) =>
                  updateField("redraw_line_spacing", Number(e.target.value) || 1.06)
                }
              />
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3 border-t border-zinc-100 pt-4 dark:border-zinc-800 sm:grid-cols-2">
            {(
              [
                ["sequential_context_mode", "Sequential Context Mode"],
                ["keep_honorifics", "Preserve Honorifics (-san, -kun, etc.)"],
                ["enable_redraw", "Enable Automatic Redraw"],
                ["use_koharu", "Use Koharu Detector (Bubbles + SFX)"],
                ["koharu_erase_sfx", "Erase Detected SFX Regions"],
                ["ocr_gemini_rescue", "Gemini Vision OCR Rescue"],
              ] as const
            ).map(([key, label]) => (
              <label
                key={key}
                className="flex items-center justify-between rounded-lg border border-zinc-200 px-3 py-2.5 text-xs font-medium text-zinc-700 dark:border-zinc-800 dark:text-zinc-300"
              >
                <span>{label}</span>
                <Switch
                  checked={Boolean(s[key])}
                  onCheckedChange={(checked) => updateField(key, checked)}
                />
              </label>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
