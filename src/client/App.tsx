import { QueryClient, QueryClientProvider, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  BookOpenText,
  Cpu,
  FolderKanban,
  KeyRound,
  Layers,
  Moon,
  Settings,
  Sun,
} from "lucide-react";
import * as React from "react";
import { Toaster } from "sonner";
import { ModelsView } from "@/components/ModelsView";
import { ProjectsDashboard } from "@/components/ProjectsDashboard";
import { SettingsView } from "@/components/SettingsView";
import { WorkspaceView } from "@/components/WorkspaceView";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ApiError, api, getApiToken, setApiToken } from "@/lib/api";
import type { JobRecord } from "@/lib/types";
import { cn } from "@/lib/utils";
import { type NavTab, useEditorStore } from "@/store/editorStore";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: (failureCount, error) => {
        if (error instanceof ApiError && (error.status === 401 || error.status === 404)) {
          return false;
        }
        return failureCount < 1;
      },
      staleTime: 10_000,
    },
  },
});

const NAV_ITEMS: Array<{
  id: NavTab;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}> = [
  { id: "projects", label: "Projects", icon: FolderKanban },
  { id: "workspace", label: "Workspace", icon: Layers },
  { id: "models", label: "Models", icon: Cpu },
  { id: "settings", label: "Settings", icon: Settings },
];

function AppShell() {
  const qc = useQueryClient();
  const {
    activeTab,
    setActiveTab,
    activeProjectId,
    theme,
    setTheme,
    setLiveJobs,
    upsertLiveJob,
    setWsConnected,
  } = useEditorStore();

  const [tokenInput, setTokenInput] = React.useState("");
  const [authError, setAuthError] = React.useState<string | null>(null);
  const [tokenUnlocked, setTokenUnlocked] = React.useState(false);

  const runtimeQuery = useQuery({
    queryKey: ["runtime"],
    queryFn: api.runtime,
  });

  const authProbeQuery = useQuery({
    queryKey: ["projects", "auth-probe"],
    queryFn: api.listProjects,
    enabled: Boolean(runtimeQuery.data?.auth_required) && !tokenUnlocked,
  });

  const needsAuth =
    Boolean(runtimeQuery.data?.auth_required) &&
    !tokenUnlocked &&
    (authProbeQuery.error instanceof ApiError
      ? authProbeQuery.error.status === 401
      : !getApiToken());

  // Sync Tailwind dark class on documentElement
  React.useEffect(() => {
    const root = document.documentElement;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () => {
      const dark = theme === "dark" || (theme === "system" && media.matches);
      root.classList.toggle("dark", dark);
    };
    apply();
    media.addEventListener("change", apply);
    return () => media.removeEventListener("change", apply);
  }, [theme]);

  // Real-time WebSocket progress channel for the active project
  React.useEffect(() => {
    if (!activeProjectId || needsAuth) {
      setWsConnected(false);
      setLiveJobs([]);
      return;
    }

    let cancelled = false;
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (cancelled) return;
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const token = getApiToken();
      const qs = token ? `?token=${encodeURIComponent(token)}` : "";
      const url = `${proto}//${window.location.host}/ws/projects/${encodeURIComponent(activeProjectId)}/progress${qs}`;

      ws = new WebSocket(url);

      ws.onopen = () => {
        if (!cancelled) setWsConnected(true);
      };

      ws.onmessage = (evt) => {
        if (cancelled) return;
        try {
          const msg = JSON.parse(String(evt.data)) as {
            type?: string;
            jobs?: JobRecord[];
            job?: JobRecord;
            page_id?: string;
          };

          if (msg.type === "hello" && Array.isArray(msg.jobs)) {
            setLiveJobs(msg.jobs);
          } else if (msg.type === "job" && msg.job) {
            upsertLiveJob(msg.job);
            if (msg.job.status === "completed" || msg.job.status === "failed" || msg.job.status === "cancelled") {
              qc.invalidateQueries({ queryKey: ["pages", activeProjectId] });
              qc.invalidateQueries({ queryKey: ["projects"] });
            }
          } else if (msg.type === "page_changed") {
            if (msg.page_id) {
              qc.invalidateQueries({ queryKey: ["page", msg.page_id] });
            }
            qc.invalidateQueries({ queryKey: ["pages", activeProjectId] });
          } else if (msg.type === "project") {
            qc.invalidateQueries({ queryKey: ["project", activeProjectId] });
            qc.invalidateQueries({ queryKey: ["projects"] });
          }
        } catch {
          // ignore malformed non-JSON frame
        }
      };

      ws.onclose = (ev) => {
        if (cancelled) return;
        setWsConnected(false);
        if (ev.code !== 4401 && ev.code !== 4404) {
          reconnectTimer = setTimeout(connect, 2500);
        }
      };

      ws.onerror = () => {
        ws?.close();
      };
    };

    connect();

    return () => {
      cancelled = true;
      setWsConnected(false);
      if (reconnectTimer) clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, [activeProjectId, needsAuth, qc, setLiveJobs, setWsConnected, upsertLiveJob]);

  const handleUnlock = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError(null);
    setApiToken(tokenInput);
    try {
      await api.listProjects();
      setTokenUnlocked(true);
      qc.invalidateQueries();
    } catch (err) {
      setApiToken("");
      setAuthError(err instanceof Error ? err.message : "Invalid API token");
    }
  };

  if (needsAuth) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-zinc-50 p-4 dark:bg-zinc-950">
        <div className="w-full max-w-md rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <div className="flex items-center gap-2.5 text-zinc-900 dark:text-zinc-100">
            <KeyRound className="h-5 w-5 text-blue-600 dark:text-blue-400" />
            <h1 className="font-display text-lg font-semibold">API Access Required</h1>
          </div>
          <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">
            This server enforces Bearer token authentication (<code className="font-mono">MT_API_TOKEN</code>).
            Enter your access token to unlock the workspace. It stays in memory for this session only.
          </p>
          <form className="mt-4 space-y-3" onSubmit={handleUnlock}>
            <Input
              type="password"
              value={tokenInput}
              onChange={(e) => setTokenInput(e.target.value)}
              placeholder="Enter Bearer token"
              autoFocus
              required
            />
            {authError && <p className="text-xs text-rose-600 dark:text-rose-400">{authError}</p>}
            <Button type="submit" className="w-full">
              Unlock Workspace
            </Button>
          </form>
        </div>
      </div>
    );
  }

  const isDarkActive =
    theme === "dark" ||
    (theme === "system" &&
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches);

  return (
    <div className="flex h-dvh w-full flex-col overflow-hidden bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      {/* Desktop 3-Zone Top Bar */}
      <header className="flex h-12 shrink-0 items-center justify-between border-b border-zinc-200 bg-white px-4 dark:border-zinc-800 dark:bg-zinc-900 md:h-14 md:px-6">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600 text-white dark:bg-blue-500 dark:text-zinc-950">
            <BookOpenText className="h-4 w-4" />
          </div>
          <span className="font-display text-base font-semibold tracking-tight whitespace-nowrap">
            Manga Translator
          </span>
        </div>

        {/* Center Navigation Links (Desktop) */}
        <nav className="hidden items-center gap-1 md:flex" aria-label="Primary">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => setActiveTab(item.id)}
                className={cn(
                  "flex h-9 items-center gap-2 rounded-lg px-3 text-sm font-medium whitespace-nowrap transition-colors",
                  active
                    ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-950"
                    : "text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-100"
                )}
              >
                <Icon className="h-4 w-4" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Right Primary Contextual Action + Theme Switch */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            aria-label="Toggle theme"
            onClick={() => setTheme(isDarkActive ? "light" : "dark")}
            className="flex h-9 w-9 items-center justify-center rounded-lg text-zinc-600 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800"
          >
            {isDarkActive ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>

          <Button
            size="sm"
            className="hidden md:inline-flex"
            onClick={() => setActiveTab(activeTab === "workspace" ? "projects" : "workspace")}
          >
            {activeTab === "workspace" ? "All Projects" : "Open Workspace"}
          </Button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="min-h-0 flex-1 overflow-hidden">
        {activeTab === "projects" && <ProjectsDashboard />}
        {activeTab === "workspace" && <WorkspaceView />}
        {activeTab === "models" && <ModelsView />}
        {activeTab === "settings" && <SettingsView />}
      </main>

      {/* Mobile Bottom Tab Navigation */}
      <nav
        aria-label="Mobile Navigation"
        className="grid h-14 shrink-0 grid-cols-4 border-t border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900 md:hidden"
      >
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const active = activeTab === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => setActiveTab(item.id)}
              className={cn(
                "flex flex-col items-center justify-center gap-0.5 text-[11px] font-medium transition-colors",
                active
                  ? "text-blue-600 dark:text-blue-400"
                  : "text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
              )}
            >
              <Icon className="h-4 w-4" />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppShell />
      <Toaster position="bottom-right" richColors />
    </QueryClientProvider>
  );
}
