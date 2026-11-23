import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  FolderOpen,
  ImagePlus,
  Languages,
  Pencil,
  Play,
  Plus,
  Trash2,
  Upload,
} from "lucide-react";
import * as React from "react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import type { PageSummary, ProjectSummary } from "@/lib/types";
import { cn, formatShortDate } from "@/lib/utils";
import { useEditorStore } from "@/store/editorStore";
import { AuthImage } from "./AuthImage";
import { Button } from "./ui/button";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "./ui/dialog";
import { Input } from "./ui/input";
import { Select } from "./ui/select";
import { Switch } from "./ui/switch";

const SOURCE_LANGS = [
  { value: "ja", label: "Japanese (JA)" },
  { value: "ko", label: "Korean (KO)" },
  { value: "zh", label: "Chinese (ZH)" },
  { value: "en", label: "English (EN)" },
];

const TARGET_LANGS = [
  { value: "vi", label: "Vietnamese (VI)" },
  { value: "en", label: "English (EN)" },
  { value: "ja", label: "Japanese (JA)" },
];

export function ProjectsDashboard() {
  const qc = useQueryClient();
  const {
    activeProjectId,
    setActiveProjectId,
    setActivePageId,
    setActiveTab,
    liveJobs,
  } = useEditorStore();

  const [createOpen, setCreateOpen] = React.useState(false);
  const [newName, setNewName] = React.useState("");
  const [newWorkTitle, setNewWorkTitle] = React.useState("");
  const [newSourceLang, setNewSourceLang] = React.useState("ja");
  const [newTargetLang, setNewTargetLang] = React.useState("vi");

  const [renameOpen, setRenameOpen] = React.useState(false);
  const [renameValue, setRenameValue] = React.useState("");
  const [dragActive, setDragActive] = React.useState(false);

  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const projectsQuery = useQuery({
    queryKey: ["projects"],
    queryFn: api.listProjects,
  });

  const projects = Array.isArray(projectsQuery.data) ? projectsQuery.data : [];

  React.useEffect(() => {
    if (!activeProjectId && projects.length > 0) {
      setActiveProjectId(projects[0].id);
    }
  }, [activeProjectId, projects, setActiveProjectId]);

  const activeProject = projects.find((p) => p.id === activeProjectId) ?? null;

  const pagesQuery = useQuery({
    queryKey: ["pages", activeProjectId],
    queryFn: () => api.listPages(activeProjectId!),
    enabled: Boolean(activeProjectId),
  });

  const pages: PageSummary[] = Array.isArray(pagesQuery.data) ? pagesQuery.data : [];

  const createMutation = useMutation({
    mutationFn: () =>
      api.createProject({
        name: newName.trim(),
        source_lang: newSourceLang,
        target_lang: newTargetLang,
        work_title: newWorkTitle.trim(),
      }),
    onSuccess: (created) => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      setActiveProjectId(created.id);
      setCreateOpen(false);
      setNewName("");
      setNewWorkTitle("");
      toast.success(`Created project "${created.name}"`);
    },
    onError: (err) => toast.error(err.message),
  });

  const updateProjectMutation = useMutation({
    mutationFn: (patch: Parameters<typeof api.updateProject>[1]) =>
      api.updateProject(activeProject!.id, patch),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["projects"] });
    },
    onError: (err) => toast.error(err.message),
  });

  const deleteProjectMutation = useMutation({
    mutationFn: (id: string) => api.deleteProject(id),
    onSuccess: (_data, deletedId) => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      if (activeProjectId === deletedId) {
        setActiveProjectId(null);
      }
      toast.success("Project deleted");
    },
    onError: (err) => toast.error(err.message),
  });

  const uploadMutation = useMutation({
    mutationFn: async (files: File[]) => {
      let targetId = activeProjectId;
      if (!targetId) {
        const created = await api.createProject({
          name: `Manga Chapter ${new Date().toLocaleDateString()}`,
          source_lang: "ja",
          target_lang: "vi",
        });
        targetId = created.id;
        setActiveProjectId(targetId);
        await qc.invalidateQueries({ queryKey: ["projects"] });
      }
      return { res: await api.uploadPages(targetId, files), projectId: targetId };
    },
    onSuccess: ({ res, projectId }) => {
      qc.invalidateQueries({ queryKey: ["pages", projectId] });
      qc.invalidateQueries({ queryKey: ["projects"] });
      if (res.added > 0) {
        toast.success(`Added ${res.added} page(s)`);
      }
      if (res.skipped_duplicates.length > 0) {
        toast.info(`Skipped ${res.skipped_duplicates.length} duplicate file(s)`);
      }
      if (res.rejected.length > 0) {
        toast.error(`Rejected ${res.rejected.length} file(s): ${res.rejected[0].reason}`);
      }
    },
    onError: (err) => toast.error(err.message),
  });

  const deletePageMutation = useMutation({
    mutationFn: (pageId: string) => api.deletePage(pageId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["pages", activeProjectId] });
      qc.invalidateQueries({ queryKey: ["projects"] });
      toast.success("Page removed");
    },
    onError: (err) => toast.error(err.message),
  });

  const translateAllMutation = useMutation({
    mutationFn: (keepManual: boolean | null = null) =>
      api.translateProject(activeProject!.id, { keep_manual: keepManual }),
    onSuccess: () => {
      toast.success("Translation job queued");
      qc.invalidateQueries({ queryKey: ["jobs", activeProjectId] });
    },
    onError: (err) => toast.error(err.message),
  });

  const handleFiles = (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    uploadMutation.mutate(Array.from(fileList));
  };

  const activeJob = liveJobs.find(
    (j) =>
      j.project_id === activeProjectId &&
      j.kind === "translate" &&
      (j.status === "pending" || j.status === "processing")
  );

  const projectSettings = (activeProject?.settings ?? {}) as Record<string, unknown>;
  const translationBackend =
    (projectSettings.translation_backend as string) || "yolo_gemini_vision";
  const sequentialMode =
    projectSettings.sequential_context_mode !== undefined
      ? Boolean(projectSettings.sequential_context_mode)
      : true;
  const keepHonorifics = Boolean(projectSettings.keep_honorifics ?? false);

  return (
    <div className="mx-auto flex h-full w-full max-w-7xl flex-col gap-6 overflow-y-auto px-4 py-6 sm:px-6">
      {/* Top Split: Project Selector + Actions */}
      <div className="flex flex-col gap-4 border-b border-zinc-200 pb-5 dark:border-zinc-800 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="font-display text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
            Projects
          </h1>
          <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
            Create chapters, upload manga pages or ZIP archives, and launch translation pipelines.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2.5">
          <Button onClick={() => setCreateOpen(true)}>
            <Plus className="h-4 w-4" />
            New Project
          </Button>
          {activeProject && (
            <>
              <Button
                variant="outline"
                onClick={() => {
                  setRenameValue(activeProject.name);
                  setRenameOpen(true);
                }}
              >
                <Pencil className="h-4 w-4" />
                Rename
              </Button>
              <Button
                variant="outline"
                onClick={() => deleteProjectMutation.mutate(activeProject.id)}
                disabled={deleteProjectMutation.isPending}
              >
                <Trash2 className="h-4 w-4 text-rose-600 dark:text-rose-400" />
                Delete
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Main 2-Column Workbench */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Left Column: Project List */}
        <div className="space-y-3 lg:col-span-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
              All Projects
            </h2>
            <span className="font-mono text-xs text-zinc-500 tabular-nums">
              {projects.length} total
            </span>
          </div>

          {projectsQuery.isError ? (
            <div className="rounded-xl border border-rose-300 bg-rose-50/70 p-4 text-xs text-rose-900 dark:border-rose-800 dark:bg-rose-950/30 dark:text-rose-100">
              <p className="font-semibold">Failed to load projects</p>
              <p className="mt-1">
                {projectsQuery.error instanceof Error ? projectsQuery.error.message : "Backend request failed"}
              </p>
              <Button size="sm" variant="outline" className="mt-3" onClick={() => projectsQuery.refetch()}>
                Retry
              </Button>
            </div>
          ) : projectsQuery.isLoading ? (
            <div className="space-y-2">
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="h-20 animate-pulse rounded-xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900"
                />
              ))}
            </div>
          ) : projects.length === 0 ? (
            <div className="rounded-xl border border-dashed border-zinc-300 bg-white p-6 text-center dark:border-zinc-800 dark:bg-zinc-900">
              <FolderOpen className="mx-auto h-7 w-7 text-zinc-400" />
              <p className="mt-2 text-sm font-medium text-zinc-900 dark:text-zinc-100">
                No projects yet
              </p>
              <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                Create a project or drop manga images on the right to start.
              </p>
              <Button size="sm" className="mt-4" onClick={() => setCreateOpen(true)}>
                Create Project
              </Button>
            </div>
          ) : (
            <div className="space-y-2">
              {projects.map((p: ProjectSummary) => {
                const selected = p.id === activeProjectId;
                return (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => setActiveProjectId(p.id)}
                    className={cn(
                      "w-full rounded-xl border p-4 text-left transition-colors",
                      selected
                        ? "border-blue-600 bg-blue-50/40 dark:border-blue-500 dark:bg-blue-950/20"
                        : "border-zinc-200 bg-white hover:border-zinc-300 dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-zinc-700"
                    )}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <span className="truncate text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                        {p.name}
                      </span>
                      <span className="font-mono text-xs text-zinc-500 tabular-nums shrink-0">
                        {p.page_count}p
                      </span>
                    </div>
                    <div className="mt-1.5 flex items-center gap-1.5 text-xs text-zinc-500 dark:text-zinc-400">
                      <span className="uppercase">{p.source_lang}</span>
                      <span aria-hidden="true">→</span>
                      <span className="uppercase">{p.target_lang}</span>
                      <span aria-hidden="true">·</span>
                      <span className="truncate">{formatShortDate(p.updated_at)}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Right Column: Upload, Configuration & Pages */}
        <div className="space-y-5 lg:col-span-8">
          {/* Drag and Drop Upload Zone */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragActive(false);
              handleFiles(e.dataTransfer.files);
            }}
            className={cn(
              "rounded-xl border-2 border-dashed p-6 text-center transition-colors",
              dragActive
                ? "border-blue-500 bg-blue-50/50 dark:bg-blue-950/20"
                : "border-zinc-300 bg-white dark:border-zinc-800 dark:bg-zinc-900"
            )}
          >
            <Upload className="mx-auto h-7 w-7 text-zinc-400" />
            <h3 className="mt-2 text-sm font-semibold text-zinc-900 dark:text-zinc-100">
              Drop manga pages or ZIP archives here
            </h3>
            <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
              Supports PNG, JPG, WEBP, BMP, and ZIP collections. Pages sort naturally by filename.
            </p>
            <div className="mt-4 flex flex-wrap items-center justify-center gap-2.5">
              <Button
                variant="default"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploadMutation.isPending}
              >
                <ImagePlus className="h-4 w-4" />
                {uploadMutation.isPending ? "Uploading..." : "Select Images / ZIP"}
              </Button>
              {activeProject && pages.length > 0 && (
                <Button
                  variant="secondary"
                  onClick={() => {
                    setActivePageId(pages[0].id);
                    setActiveTab("workspace");
                  }}
                >
                  Open Workspace
                </Button>
              )}
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp,image/bmp,.zip,application/zip"
              multiple
              className="hidden"
              onChange={(e) => {
                handleFiles(e.target.files);
                e.target.value = "";
              }}
            />
          </div>

          {/* Active Project Pipeline Settings */}
          {activeProject && (
            <div className="rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                <div>
                  <label className="mb-1.5 block text-xs font-medium text-zinc-500 dark:text-zinc-400">
                    Source Language
                  </label>
                  <Select
                    value={activeProject.source_lang ?? "ja"}
                    onChange={(e) =>
                      updateProjectMutation.mutate({ source_lang: e.target.value })
                    }
                  >
                    {SOURCE_LANGS.map((l) => (
                      <option key={l.value} value={l.value}>
                        {l.label}
                      </option>
                    ))}
                  </Select>
                </div>

                <div>
                  <label className="mb-1.5 block text-xs font-medium text-zinc-500 dark:text-zinc-400">
                    Target Language
                  </label>
                  <Select
                    value={activeProject.target_lang ?? "vi"}
                    onChange={(e) =>
                      updateProjectMutation.mutate({ target_lang: e.target.value })
                    }
                  >
                    {TARGET_LANGS.map((l) => (
                      <option key={l.value} value={l.value}>
                        {l.label}
                      </option>
                    ))}
                  </Select>
                </div>

                <div>
                  <label className="mb-1.5 block text-xs font-medium text-zinc-500 dark:text-zinc-400">
                    Translation Mode
                  </label>
                  <Select
                    value={translationBackend || "yolo_gemini_vision"}
                    onChange={(e) =>
                      updateProjectMutation.mutate({
                        settings: {
                          ...projectSettings,
                          translation_backend: e.target.value,
                        },
                      })
                    }
                  >
                    <option value="yolo_ocr_gemini">Mode A · YOLO + Manga-OCR + Gemini</option>
                    <option value="yolo_gemini_vision">Mode B · YOLO + Gemini Vision</option>
                  </Select>
                </div>
              </div>

              <div className="mt-4 flex flex-wrap items-center justify-between gap-4 border-t border-zinc-100 pt-4 dark:border-zinc-800">
                <div className="flex flex-wrap items-center gap-6">
                  <label className="flex items-center gap-2 text-xs font-medium text-zinc-700 dark:text-zinc-300">
                    <Switch
                      checked={sequentialMode}
                      onCheckedChange={(checked) =>
                        updateProjectMutation.mutate({
                          settings: {
                            ...projectSettings,
                            sequential_context_mode: checked,
                          },
                        })
                      }
                    />
                    Sequential Context
                  </label>

                  <label className="flex items-center gap-2 text-xs font-medium text-zinc-700 dark:text-zinc-300">
                    <Switch
                      checked={keepHonorifics}
                      onCheckedChange={(checked) =>
                        updateProjectMutation.mutate({
                          settings: {
                            ...projectSettings,
                            keep_honorifics: checked,
                          },
                        })
                      }
                    />
                    Keep Honorifics
                  </label>
                </div>

                <div className="flex items-center gap-2">
                  <Button
                    onClick={() => translateAllMutation.mutate(true)}
                    disabled={pages.length === 0 || translateAllMutation.isPending || Boolean(activeJob)}
                  >
                    <Play className="h-4 w-4" />
                    {activeJob
                      ? `Translating (${activeJob.done}/${activeJob.total})`
                      : "Translate All Pages"}
                  </Button>
                </div>
              </div>

              {activeJob && (
                <div className="mt-4 space-y-1.5">
                  <div className="flex items-center justify-between text-xs text-zinc-600 dark:text-zinc-400">
                    <span>{activeJob.message || "Processing..."}</span>
                    <span className="font-mono tabular-nums">
                      {activeJob.done}/{activeJob.total}
                    </span>
                  </div>
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
                    <div
                      className="h-full bg-blue-600 transition-all dark:bg-blue-500"
                      style={{
                        width: `${activeJob.total ? Math.round((activeJob.done / activeJob.total) * 100) : 0}%`,
                      }}
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Pages List */}
          {activeProject && (
            <div className="rounded-xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
              <div className="flex items-center justify-between border-b border-zinc-100 px-4 py-3 dark:border-zinc-800">
                <span className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                  Pages
                </span>
                <span className="font-mono text-xs text-zinc-500 tabular-nums">
                  {pages.length} pages
                </span>
              </div>

              {pages.length === 0 ? (
                <div className="p-8 text-center text-sm text-zinc-500 dark:text-zinc-400">
                  No pages uploaded to this project yet.
                </div>
              ) : (
                <div className="divide-y divide-zinc-100 dark:divide-zinc-800">
                  {pages.map((page) => (
                    <div
                      key={page.id}
                      className="flex items-center gap-3.5 px-4 py-3 hover:bg-zinc-50 dark:hover:bg-zinc-800/50"
                    >
                      <button
                        type="button"
                        onClick={() => {
                          setActivePageId(page.id);
                          setActiveTab("workspace");
                        }}
                        className="flex flex-1 min-w-0 items-center gap-3.5 text-left"
                      >
                        <AuthImage
                          src={`/api/pages/${page.id}/thumbnail`}
                          alt={page.page_name}
                          referrerPolicy="no-referrer"
                          className="h-14 w-11 shrink-0 rounded-md border border-zinc-200 bg-zinc-100 object-cover dark:border-zinc-700 dark:bg-zinc-800"
                        />
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-sm font-medium text-zinc-900 dark:text-zinc-100">
                            {page.page_name}
                          </div>
                          <div className="mt-0.5 flex flex-wrap items-center gap-1.5 text-xs text-zinc-500 dark:text-zinc-400">
                            <span className="font-mono tabular-nums">
                              {page.width}×{page.height}
                            </span>
                            <span aria-hidden="true">·</span>
                            <span className="font-mono tabular-nums">
                              {page.dialogue_count} dialogues
                            </span>
                            <span aria-hidden="true">·</span>
                            <span
                              className={cn(
                                "font-medium",
                                page.status === "done" && "text-emerald-600 dark:text-emerald-400",
                                page.status === "error" && "text-rose-600 dark:text-rose-400",
                                page.status === "translating" && "text-blue-600 dark:text-blue-400"
                              )}
                            >
                              {page.status === "done"
                                ? "Translated"
                                : page.status === "translating"
                                  ? "Translating"
                                  : page.status === "error"
                                    ? "Error"
                                    : "Pending"}
                            </span>
                          </div>
                          {page.error_message && (
                            <p className="mt-1 truncate text-xs text-rose-600 dark:text-rose-400">
                              {page.error_message}
                            </p>
                          )}
                        </div>
                      </button>

                      <div className="flex items-center gap-1.5 shrink-0">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => {
                            setActivePageId(page.id);
                            setActiveTab("workspace");
                          }}
                        >
                          <Languages className="h-3.5 w-3.5" />
                          Edit
                        </Button>
                        <Button
                          size="icon"
                          variant="ghost"
                          aria-label="Delete page"
                          onClick={() => deletePageMutation.mutate(page.id)}
                        >
                          <Trash2 className="h-4 w-4 text-zinc-500 hover:text-rose-600" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Create Project Dialog */}
      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent>
          <DialogTitle>New Translation Project</DialogTitle>
          <DialogDescription>
            Configure project title and language pair.
          </DialogDescription>
          <form
            className="mt-4 space-y-4"
            onSubmit={(e) => {
              e.preventDefault();
              if (newName.trim()) createMutation.mutate();
            }}
          >
            <div>
              <label className="mb-1.5 block text-xs font-medium text-zinc-600 dark:text-zinc-400">
                Project Name
              </label>
              <Input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Chapter 01"
                required
              />
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-medium text-zinc-600 dark:text-zinc-400">
                Series / Work Title (optional, improves Gemini context)
              </label>
              <Input
                value={newWorkTitle}
                onChange={(e) => setNewWorkTitle(e.target.value)}
                placeholder="Series title"
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="mb-1.5 block text-xs font-medium text-zinc-600 dark:text-zinc-400">
                  Source
                </label>
                <Select value={newSourceLang} onChange={(e) => setNewSourceLang(e.target.value)}>
                  {SOURCE_LANGS.map((l) => (
                    <option key={l.value} value={l.value}>
                      {l.label}
                    </option>
                  ))}
                </Select>
              </div>
              <div>
                <label className="mb-1.5 block text-xs font-medium text-zinc-600 dark:text-zinc-400">
                  Target
                </label>
                <Select value={newTargetLang} onChange={(e) => setNewTargetLang(e.target.value)}>
                  {TARGET_LANGS.map((l) => (
                    <option key={l.value} value={l.value}>
                      {l.label}
                    </option>
                  ))}
                </Select>
              </div>
            </div>
            <div className="flex justify-end gap-2 pt-2">
              <Button type="button" variant="outline" onClick={() => setCreateOpen(false)}>
                Cancel
              </Button>
              <Button type="submit" disabled={createMutation.isPending}>
                Create Project
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>

      {/* Rename Project Dialog */}
      <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
        <DialogContent>
          <DialogTitle>Rename Project</DialogTitle>
          <form
            className="mt-4 space-y-4"
            onSubmit={(e) => {
              e.preventDefault();
              if (renameValue.trim() && activeProject) {
                updateProjectMutation.mutate({ name: renameValue.trim() });
                setRenameOpen(false);
                toast.success("Project renamed");
              }
            }}
          >
            <Input
              value={renameValue}
              onChange={(e) => setRenameValue(e.target.value)}
              required
            />
            <div className="flex justify-end gap-2">
              <Button type="button" variant="outline" onClick={() => setRenameOpen(false)}>
                Cancel
              </Button>
              <Button type="submit">Save</Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
}
