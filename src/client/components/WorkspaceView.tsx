import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlignCenter,
  AlignLeft,
  AlignRight,
  ArrowDown,
  ArrowUp,
  Bold,
  BookOpen,
  Crop,
  Download,
  Eraser,
  Eye,
  EyeOff,
  FileArchive,
  FileText,
  Hand,
  Highlighter,
  Layers,
  Maximize2,
  MessageSquare,
  MousePointer2,
  PaintBucket,
  Pen,
  PlusSquare,
  Redo2,
  RefreshCw,
  RotateCcw,
  ScanSearch,
  Search,
  Sparkles,
  Square,
  Trash2,
  Type as TypeIcon,
  Undo2,
  Wand2,
} from "lucide-react";
import * as React from "react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { exportPagePsd, exportPageRaster, exportProjectZip } from "@/lib/export";
import type {
  Box,
  Dialogue,
  FillItem,
  GrammarAnalysisResult,
  PageDetail,
  StrokeItem,
} from "@/lib/types";
import { cn } from "@/lib/utils";
import { type CanvasTool, useEditorStore } from "@/store/editorStore";
import { AuthImage } from "./AuthImage";
import { KonvaStage } from "./KonvaStage";
import { Button } from "./ui/button";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "./ui/dialog";
import { Input } from "./ui/input";
import { Select } from "./ui/select";
import { Switch } from "./ui/switch";
import { Textarea } from "./ui/textarea";

const TOOL_BUTTONS: Array<{ id: CanvasTool; label: string; icon: React.ComponentType<{ className?: string }> }> = [
  { id: "select", label: "Select", icon: MousePointer2 },
  { id: "pan", label: "Pan", icon: Hand },
  { id: "add_box", label: "Add Box", icon: PlusSquare },
  { id: "translate_region", label: "Crop & Translate", icon: Crop },
  { id: "pen", label: "Pen", icon: Pen },
  { id: "highlighter", label: "Highlighter", icon: Highlighter },
  { id: "eraser", label: "Eraser", icon: Eraser },
  { id: "fill_brush", label: "Fill Brush", icon: PaintBucket },
  { id: "magic_wand", label: "Magic Wand", icon: Wand2 },
];

export function WorkspaceView() {
  const qc = useQueryClient();
  const {
    activeProjectId,
    activePageId,
    setActivePageId,
    setActiveTab,
    selectedDialogueId,
    setSelectedDialogueId,
    tool,
    setTool,
    inspectorTab,
    setInspectorTab,
    brushColor,
    setBrushColor,
    brushSize,
    setBrushSize,
    fillColor,
    setFillColor,
    fillMethod,
    setFillMethod,
    wandTolerance,
    setWandTolerance,
    showTranslated,
    showRedraw,
    showBoxes,
    showStrokes,
    toggleLayer,
    triggerStageCommand,
    undoStack,
    redoStack,
    popUndo,
    popRedo,
    liveJobs,
    mobileSheetOpen,
    setMobileSheetOpen,
  } = useEditorStore();

  const [grammarDialog, setGrammarDialog] = React.useState<{
    open: boolean;
    loading: boolean;
    data: GrammarAnalysisResult | null;
    error: string | null;
  }>({ open: false, loading: false, data: null, error: null });

  const [findText, setFindText] = React.useState("");
  const [replaceText, setReplaceText] = React.useState("");
  const [findCase, setFindCase] = React.useState(false);

  const [glossarySrc, setGlossarySrc] = React.useState("");
  const [glossaryDst, setGlossaryDst] = React.useState("");

  const [chatCharacter, setChatCharacter] = React.useState<"assistant" | "sengoku_yuno">("assistant");
  const [chatInput, setChatInput] = React.useState("");
  const [chatHistory, setChatHistory] = React.useState<Array<{ role: "user" | "bot"; text: string }>>([]);

  const [exporting, setExporting] = React.useState(false);

  const projectQuery = useQuery({
    queryKey: ["project", activeProjectId],
    queryFn: () => api.getProject(activeProjectId!),
    enabled: Boolean(activeProjectId),
  });

  const pagesQuery = useQuery({
    queryKey: ["pages", activeProjectId],
    queryFn: () => api.listPages(activeProjectId!),
    enabled: Boolean(activeProjectId),
  });

  const pages = Array.isArray(pagesQuery.data) ? pagesQuery.data : [];

  React.useEffect(() => {
    if (!activePageId && pages.length > 0) {
      setActivePageId(pages[0].id);
    }
  }, [activePageId, pages, setActivePageId]);

  const pageQuery = useQuery({
    queryKey: ["page", activePageId],
    queryFn: () => api.getPage(activePageId!),
    enabled: Boolean(activePageId),
  });

  const page = pageQuery.data ?? null;
  const project = projectQuery.data ?? null;

  const patchDialogueMutation = useMutation({
    mutationFn: ({ dialogueId, patch }: { dialogueId: number; patch: Partial<Dialogue> }) =>
      api.patchDialogue(activePageId!, dialogueId, patch),
    onMutate: async ({ dialogueId, patch }) => {
      await qc.cancelQueries({ queryKey: ["page", activePageId] });
      const prev = qc.getQueryData<PageDetail>(["page", activePageId]);
      if (prev) {
        qc.setQueryData<PageDetail>(["page", activePageId], {
          ...prev,
          dialogues: prev.dialogues.map((d) =>
            d.dialogue_id === dialogueId ? { ...d, ...patch, edited: true } : d
          ),
        });
      }
      return { prev };
    },
    onError: (err, _vars, ctx) => {
      if (ctx?.prev) qc.setQueryData(["page", activePageId], ctx.prev);
      toast.error(err.message);
    },
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ["page", activePageId] });
      qc.invalidateQueries({ queryKey: ["pages", activeProjectId] });
    },
  });

  const createDialogueMutation = useMutation({
    mutationFn: (box: Box) =>
      api.createDialogue(activePageId!, {
        box_location: box,
        original_text: "",
        translated_text: "New dialogue",
      }),
    onSuccess: (created) => {
      qc.invalidateQueries({ queryKey: ["page", activePageId] });
      setSelectedDialogueId(created.dialogue_id);
      setTool("select");
      toast.success("Added dialogue box");
    },
    onError: (err) => toast.error(err.message),
  });

  const deleteDialogueMutation = useMutation({
    mutationFn: (dialogueId: number) => api.deleteDialogue(activePageId!, dialogueId),
    onSuccess: (_data, dialogueId) => {
      if (selectedDialogueId === dialogueId) setSelectedDialogueId(null);
      qc.invalidateQueries({ queryKey: ["page", activePageId] });
      qc.invalidateQueries({ queryKey: ["pages", activeProjectId] });
    },
    onError: (err) => toast.error(err.message),
  });

  const reorderMutation = useMutation({
    mutationFn: (order: number[]) => api.reorderDialogues(activePageId!, order),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["page", activePageId] }),
    onError: (err) => toast.error(err.message),
  });

  const saveStrokesMutation = useMutation({
    mutationFn: (strokes: StrokeItem[]) => api.saveStrokes(activePageId!, strokes),
    onSuccess: (updated) => qc.setQueryData(["page", activePageId], updated),
    onError: (err) => toast.error(err.message),
  });

  const saveFillsMutation = useMutation({
    mutationFn: (fills: FillItem[]) => api.saveFills(activePageId!, fills),
    onSuccess: (updated) => qc.setQueryData(["page", activePageId], updated),
    onError: (err) => toast.error(err.message),
  });

  const translateRegionMutation = useMutation({
    mutationFn: (box: Box) => api.translateRegion(activePageId!, { box, add: true }),
    onSuccess: (created) => {
      qc.invalidateQueries({ queryKey: ["page", activePageId] });
      setSelectedDialogueId(created.dialogue_id);
      setTool("select");
      toast.success("Region translated");
    },
    onError: (err) => toast.error(err.message),
  });

  const translatePageMutation = useMutation({
    mutationFn: () => api.translatePage(activePageId!, { keep_manual: true }),
    onSuccess: () => {
      toast.success("Page translation started");
      qc.invalidateQueries({ queryKey: ["jobs", activeProjectId] });
    },
    onError: (err) => toast.error(err.message),
  });

  const detectRegionsMutation = useMutation({
    mutationFn: () => api.detectRegions(activePageId!),
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ["page", activePageId] });
      toast.success(`Detected ${res.detected} speech region(s)`);
    },
    onError: (err) => toast.error(err.message),
  });

  const redrawMutation = useMutation({
    mutationFn: () => api.requestRedraw(activePageId!, { force_full: false }),
    onSuccess: () => toast.success("Redraw queued on FastAPI backend"),
    onError: (err) => toast.error(err.message),
  });

  const findReplaceMutation = useMutation({
    mutationFn: () =>
      api.findReplace(activeProjectId!, {
        find: findText,
        replace: replaceText,
        field: "translated_text",
        case_sensitive: findCase,
      }),
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ["page", activePageId] });
      toast.success(`Replaced ${res.matches} match(es) across ${res.updated_dialogues} dialogue(s)`);
    },
    onError: (err) => toast.error(err.message),
  });

  const chatMutation = useMutation({
    mutationFn: (msg: string) => api.chat(activeProjectId!, { message: msg, character: chatCharacter }),
    onSuccess: (res) => {
      setChatHistory((h) => [...h, { role: "bot", text: res.reply }]);
    },
    onError: (err) => toast.error(err.message),
  });

  const handleUndo = React.useCallback(() => {
    if (!page) return;
    const prev = popUndo({ strokes: page.strokes, fills: page.fills });
    if (!prev) return;
    saveStrokesMutation.mutate(prev.strokes);
    saveFillsMutation.mutate(prev.fills);
  }, [page, popUndo, saveStrokesMutation, saveFillsMutation]);

  const handleRedo = React.useCallback(() => {
    if (!page) return;
    const next = popRedo({ strokes: page.strokes, fills: page.fills });
    if (!next) return;
    saveStrokesMutation.mutate(next.strokes);
    saveFillsMutation.mutate(next.fills);
  }, [page, popRedo, saveStrokesMutation, saveFillsMutation]);

  // Keyboard shortcuts
  React.useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const inInput = target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
      if (inInput) return;

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey) handleRedo();
        else handleUndo();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") {
        e.preventDefault();
        handleRedo();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "f") {
        e.preventDefault();
        setInspectorTab("tools");
      } else if (e.key === "Delete" && selectedDialogueId !== null) {
        deleteDialogueMutation.mutate(selectedDialogueId);
      } else if (e.key.toLowerCase() === "f") {
        triggerStageCommand("fit");
      } else if (e.key === "1") {
        triggerStageCommand("one");
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [handleUndo, handleRedo, selectedDialogueId, deleteDialogueMutation, setInspectorTab, triggerStageCommand]);

  const selectedDialogue =
    page?.dialogues.find((d) => d.dialogue_id === selectedDialogueId) ?? null;

  const moveDialogue = (idx: number, dir: -1 | 1) => {
    if (!page) return;
    const target = idx + dir;
    if (target < 0 || target >= page.dialogues.length) return;
    const order = page.dialogues.map((d) => d.dialogue_id);
    [order[idx], order[target]] = [order[target], order[idx]];
    reorderMutation.mutate(order);
  };

  const applyFormattingToAll = async () => {
    if (!page || !selectedDialogue) return;
    const ov = selectedDialogue.overrides || {};
    await Promise.all(
      page.dialogues.map((d) =>
        api.patchDialogue(page.id, d.dialogue_id, {
          overrides: {
            ...d.overrides,
            font_path: ov.font_path,
            font_size: ov.font_size,
            color: ov.color,
            bold: ov.bold,
            line_spacing: ov.line_spacing,
            align: ov.align,
          },
        })
      )
    );
    qc.invalidateQueries({ queryKey: ["page", page.id] });
    toast.success("Applied formatting to all dialogues on this page");
  };

  const openGrammar = async (text: string) => {
    if (!page || !text.trim()) {
      toast.info("Dialogue has no source text to analyze");
      return;
    }
    setGrammarDialog({ open: true, loading: true, data: null, error: null });
    try {
      const res = await api.analyzeGrammar(page.id, text);
      setGrammarDialog({ open: true, loading: false, data: res.analysis, error: null });
    } catch (err) {
      setGrammarDialog({
        open: true,
        loading: false,
        data: null,
        error: err instanceof Error ? err.message : "Failed to analyze grammar",
      });
    }
  };

  const handleBatchExportZip = async (fmt: "png" | "jpg" | "psd") => {
    if (!project || pages.length === 0) return;
    setExporting(true);
    try {
      const details = await Promise.all(pages.map((p) => api.getPage(p.id)));
      await exportProjectZip(project, details, fmt);
      toast.success(`Exported ${fmt.toUpperCase()} ZIP archive`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Export failed");
    } finally {
      setExporting(false);
    }
  };

  if (!activeProjectId || pages.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center p-6 text-center">
        <Layers className="h-10 w-10 text-zinc-400" />
        <h2 className="mt-3 font-display text-lg font-semibold text-zinc-900 dark:text-zinc-100">
          No active pages in workspace
        </h2>
        <p className="mt-1 max-w-sm text-sm text-zinc-500 dark:text-zinc-400">
          Create a project and upload manga images or a ZIP archive in Projects first.
        </p>
        <Button className="mt-4" onClick={() => setActiveTab("projects")}>
          Go to Projects
        </Button>
      </div>
    );
  }

  const activeJob = liveJobs.find(
    (j) =>
      j.project_id === activeProjectId &&
      (j.status === "pending" || j.status === "processing")
  );

  const isPaintTool =
    tool === "pen" ||
    tool === "highlighter" ||
    tool === "eraser" ||
    tool === "fill_brush" ||
    tool === "magic_wand";

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Workspace Top Contextual Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-200 bg-white px-3 py-2 dark:border-zinc-800 dark:bg-zinc-900">
        {/* Tool Selector */}
        <div className="flex items-center gap-1 overflow-x-auto">
          {TOOL_BUTTONS.map((t) => {
            const Icon = t.icon;
            const active = tool === t.id;
            return (
              <button
                key={t.id}
                type="button"
                title={t.label}
                onClick={() => setTool(t.id)}
                className={cn(
                  "flex h-9 items-center gap-1.5 rounded-lg px-2.5 text-xs font-medium whitespace-nowrap shrink-0 transition-colors",
                  active
                    ? "bg-blue-600 text-white dark:bg-blue-500 dark:text-zinc-950"
                    : "text-zinc-700 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-800"
                )}
              >
                <Icon className="h-4 w-4" />
                <span className="hidden xl:inline">{t.label}</span>
              </button>
            );
          })}

          <div className="mx-1 h-5 w-px bg-zinc-200 dark:bg-zinc-800" />

          <Button
            size="icon"
            variant="ghost"
            title="Undo (Ctrl+Z)"
            disabled={undoStack.length === 0}
            onClick={handleUndo}
          >
            <Undo2 className="h-4 w-4" />
          </Button>
          <Button
            size="icon"
            variant="ghost"
            title="Redo (Ctrl+Y)"
            disabled={redoStack.length === 0}
            onClick={handleRedo}
          >
            <Redo2 className="h-4 w-4" />
          </Button>
        </div>

        {/* Layer Visibility & AI Actions */}
        <div className="flex items-center gap-1.5 overflow-x-auto">
          <Button
            size="sm"
            variant={showTranslated ? "secondary" : "ghost"}
            onClick={() => toggleLayer("showTranslated")}
          >
            <TypeIcon className="h-3.5 w-3.5" />
            Text
          </Button>
          <Button
            size="sm"
            variant={showRedraw ? "secondary" : "ghost"}
            onClick={() => toggleLayer("showRedraw")}
          >
            <Layers className="h-3.5 w-3.5" />
            Redraw
          </Button>
          <Button
            size="sm"
            variant={showBoxes ? "secondary" : "ghost"}
            onClick={() => toggleLayer("showBoxes")}
          >
            <Square className="h-3.5 w-3.5" />
            Boxes
          </Button>

          <div className="mx-1 h-5 w-px bg-zinc-200 dark:bg-zinc-800" />

          <Button size="icon" variant="ghost" title="Fit to screen (F)" onClick={() => triggerStageCommand("fit")}>
            <Maximize2 className="h-4 w-4" />
          </Button>
          <Button size="sm" variant="ghost" title="100% scale (1)" onClick={() => triggerStageCommand("one")}>
            1:1
          </Button>

          <div className="mx-1 h-5 w-px bg-zinc-200 dark:bg-zinc-800" />

          <Button
            size="sm"
            variant="outline"
            disabled={!page || detectRegionsMutation.isPending}
            onClick={() => detectRegionsMutation.mutate()}
          >
            <ScanSearch className="h-3.5 w-3.5" />
            Detect
          </Button>
          <Button
            size="sm"
            disabled={!page || translatePageMutation.isPending}
            onClick={() => translatePageMutation.mutate()}
          >
            <Sparkles className="h-3.5 w-3.5" />
            Translate Page
          </Button>
        </div>
      </div>

      {/* Contextual Paint / Wand Tool Options Bar */}
      {isPaintTool && page && (
        <div className="flex flex-wrap items-center gap-4 border-b border-zinc-200 bg-zinc-50 px-4 py-2 text-xs dark:border-zinc-800 dark:bg-zinc-950">
          {(tool === "pen" || tool === "highlighter") && (
            <label className="flex items-center gap-2">
              <span className="text-zinc-500">Color</span>
              <input
                type="color"
                value={brushColor}
                onChange={(e) => setBrushColor(e.target.value)}
                className="h-6 w-8 cursor-pointer rounded border border-zinc-300 bg-transparent"
              />
            </label>
          )}
          {tool === "fill_brush" && (
            <>
              <label className="flex items-center gap-2">
                <span className="text-zinc-500">Fill</span>
                <input
                  type="color"
                  value={fillColor}
                  onChange={(e) => setFillColor(e.target.value)}
                  className="h-6 w-8 cursor-pointer rounded border border-zinc-300 bg-transparent"
                />
              </label>
              <label className="flex items-center gap-2">
                <span className="text-zinc-500">Method</span>
                <Select
                  className="h-7 w-28 py-0 text-xs"
                  value={fillMethod}
                  onChange={(e) => setFillMethod(e.target.value as typeof fillMethod)}
                >
                  <option value="auto">Auto</option>
                  <option value="lama">LaMa</option>
                  <option value="texture">Texture</option>
                  <option value="opencv">OpenCV</option>
                </Select>
              </label>
            </>
          )}
          {tool !== "magic_wand" ? (
            <label className="flex items-center gap-2">
              <span className="text-zinc-500">Size</span>
              <input
                type="range"
                min={2}
                max={64}
                value={brushSize}
                onChange={(e) => setBrushSize(Number(e.target.value))}
              />
              <span className="font-mono w-7 tabular-nums">{brushSize}</span>
            </label>
          ) : (
            <label className="flex items-center gap-2">
              <span className="text-zinc-500">Tolerance</span>
              <input
                type="range"
                min={4}
                max={80}
                value={wandTolerance}
                onChange={(e) => setWandTolerance(Number(e.target.value))}
              />
              <span className="font-mono w-7 tabular-nums">{wandTolerance}</span>
            </label>
          )}

          <div className="ml-auto flex items-center gap-2">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => toggleLayer("showStrokes")}
            >
              {showStrokes ? "Hide Strokes" : "Show Strokes"}
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                saveStrokesMutation.mutate([]);
                saveFillsMutation.mutate([]);
              }}
            >
              Clear Paint
            </Button>
          </div>
        </div>
      )}

      {/* Mobile Horizontal Page Strip */}
      <div className="flex items-center gap-2 overflow-x-auto border-b border-zinc-200 bg-white px-3 py-2 dark:border-zinc-800 dark:bg-zinc-900 lg:hidden">
        {pages.map((p, idx) => (
          <button
            key={p.id}
            type="button"
            onClick={() => setActivePageId(p.id)}
            className={cn(
              "relative shrink-0 overflow-hidden rounded-md border-2 transition-opacity",
              p.id === activePageId
                ? "border-blue-600 opacity-100"
                : "border-transparent opacity-70"
            )}
          >
            <AuthImage
              src={`/api/pages/${p.id}/thumbnail`}
              alt={p.page_name}
              referrerPolicy="no-referrer"
              className="h-12 w-9 object-cover"
            />
            <span className="sr-only">Page {idx + 1}</span>
          </button>
        ))}
      </div>

      {/* 3-Column Studio Body */}
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        {/* Desktop Left Rail: Page Thumbnails */}
        <aside className="hidden w-56 shrink-0 flex-col border-r border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900 lg:flex">
          <div className="flex items-center justify-between border-b border-zinc-100 px-3 py-2.5 dark:border-zinc-800">
            <span className="text-xs font-semibold text-zinc-700 dark:text-zinc-300">
              Pages ({pages.length})
            </span>
          </div>
          <div className="flex-1 space-y-1.5 overflow-y-auto p-2">
            {pages.map((p, idx) => {
              const active = p.id === activePageId;
              return (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => setActivePageId(p.id)}
                  className={cn(
                    "flex w-full items-center gap-2.5 rounded-lg border p-2 text-left transition-colors",
                    active
                      ? "border-blue-600 bg-blue-50/50 dark:border-blue-500 dark:bg-blue-950/20"
                      : "border-transparent hover:bg-zinc-100 dark:hover:bg-zinc-800/60"
                  )}
                >
                  <AuthImage
                    src={`/api/pages/${p.id}/thumbnail`}
                    alt={p.page_name}
                    referrerPolicy="no-referrer"
                    className="h-12 w-9 shrink-0 rounded border border-zinc-200 object-cover dark:border-zinc-700"
                  />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-xs font-medium text-zinc-900 dark:text-zinc-100">
                      {idx + 1}. {p.page_name}
                    </div>
                    <div className="mt-0.5 flex items-center gap-1 text-[11px] text-zinc-500 dark:text-zinc-400">
                      <span className="font-mono tabular-nums">{p.dialogue_count}</span>
                      <span aria-hidden="true">·</span>
                      <span>{p.status}</span>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </aside>

        {/* Center: Konva Image Viewer */}
        <div className="relative min-h-[260px] flex-1">
          {page ? (
            <KonvaStage
              page={page}
              onPatchDialogue={(dialogueId, patch) =>
                patchDialogueMutation.mutate({ dialogueId, patch })
              }
              onCreateBox={(box) => createDialogueMutation.mutate(box)}
              onTranslateRegion={(box) => translateRegionMutation.mutate(box)}
              onSaveStrokes={(strokes) => saveStrokesMutation.mutate(strokes)}
              onSaveFills={(fills) => saveFillsMutation.mutate(fills)}
            />
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-zinc-500">
              Loading page...
            </div>
          )}

          {/* Mobile sheet toggle button */}
          <button
            type="button"
            onClick={() => setMobileSheetOpen(!mobileSheetOpen)}
            className="absolute bottom-3 right-3 rounded-lg border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium text-zinc-800 shadow-sm dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100 lg:hidden"
          >
            {mobileSheetOpen ? "Hide Inspector" : `Dialogues (${page?.dialogues.length ?? 0})`}
          </button>
        </div>

        {/* Right Inspector Panel (Collapsible bottom sheet on mobile, 360px rail on desktop) */}
        <aside
          className={cn(
            "flex flex-col border-t border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900 lg:w-96 lg:shrink-0 lg:border-l lg:border-t-0",
            mobileSheetOpen ? "h-[46%] lg:h-auto" : "hidden lg:flex"
          )}
        >
          {/* Inspector Segmented Tabs */}
          <div className="grid grid-cols-3 gap-1 border-b border-zinc-200 p-2 dark:border-zinc-800">
            {(
              [
                { id: "dialogues", label: `Dialogues (${page?.dialogues.length ?? 0})` },
                { id: "typeset", label: "Typeset" },
                { id: "tools", label: "Tools & Export" },
              ] as const
            ).map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setInspectorTab(tab.id)}
                className={cn(
                  "rounded-lg px-2 py-1.5 text-xs font-medium whitespace-nowrap transition-colors",
                  inspectorTab === tab.id
                    ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-950"
                    : "text-zinc-600 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800"
                )}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab 1: Dialogues Editor */}
          {inspectorTab === "dialogues" && page && (
            <div className="flex-1 space-y-3 overflow-y-auto p-3">
              {page.dialogues.length === 0 ? (
                <div className="rounded-xl border border-dashed border-zinc-300 p-6 text-center text-xs text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
                  No dialogues yet. Click <strong>Translate Page</strong> or use{" "}
                  <strong>Add Box</strong> to draw a speech region.
                </div>
              ) : (
                page.dialogues.map((d, idx) => {
                  const selected = d.dialogue_id === selectedDialogueId;
                  return (
                    <div
                      key={d.dialogue_id}
                      onClick={() => setSelectedDialogueId(d.dialogue_id)}
                      className={cn(
                        "rounded-xl border p-3 transition-colors",
                        selected
                          ? "border-blue-600 bg-blue-50/30 dark:border-blue-500 dark:bg-blue-950/20"
                          : "border-zinc-200 bg-zinc-50/60 dark:border-zinc-800 dark:bg-zinc-950/40",
                        (d.hide_text || d.hide_redraw) && "opacity-65"
                      )}
                    >
                      <div className="mb-2 flex items-center justify-between gap-2 text-xs text-zinc-500 dark:text-zinc-400">
                        <div className="flex items-center gap-1.5">
                          <span className="font-mono font-semibold text-zinc-900 dark:text-zinc-100 tabular-nums">
                            #{d.dialogue_id}
                          </span>
                          {d.edited && <span className="text-amber-600 dark:text-amber-400">· edited</span>}
                          {typeof d.confidence === "number" && (
                            <span className="font-mono tabular-nums">
                              · {Math.round(d.confidence * 100)}%
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-1">
                          <button
                            type="button"
                            title="Move up"
                            disabled={idx === 0}
                            onClick={(e) => {
                              e.stopPropagation();
                              moveDialogue(idx, -1);
                            }}
                            className="rounded p-1 hover:bg-zinc-200 disabled:opacity-30 dark:hover:bg-zinc-800"
                          >
                            <ArrowUp className="h-3.5 w-3.5" />
                          </button>
                          <button
                            type="button"
                            title="Move down"
                            disabled={idx === page.dialogues.length - 1}
                            onClick={(e) => {
                              e.stopPropagation();
                              moveDialogue(idx, 1);
                            }}
                            className="rounded p-1 hover:bg-zinc-200 disabled:opacity-30 dark:hover:bg-zinc-800"
                          >
                            <ArrowDown className="h-3.5 w-3.5" />
                          </button>
                          <button
                            type="button"
                            title="Grammar analysis"
                            onClick={(e) => {
                              e.stopPropagation();
                              openGrammar(d.original_text);
                            }}
                            className="rounded p-1 hover:bg-zinc-200 dark:hover:bg-zinc-800"
                          >
                            <BookOpen className="h-3.5 w-3.5" />
                          </button>
                          <button
                            type="button"
                            title="Retranslate region"
                            onClick={(e) => {
                              e.stopPropagation();
                              translateRegionMutation.mutate(d.box_location);
                            }}
                            className="rounded p-1 hover:bg-zinc-200 dark:hover:bg-zinc-800"
                          >
                            <RefreshCw className="h-3.5 w-3.5" />
                          </button>
                          <button
                            type="button"
                            title={d.hide_text ? "Show text" : "Hide text"}
                            onClick={(e) => {
                              e.stopPropagation();
                              patchDialogueMutation.mutate({
                                dialogueId: d.dialogue_id,
                                patch: { hide_text: !d.hide_text },
                              });
                            }}
                            className="rounded p-1 hover:bg-zinc-200 dark:hover:bg-zinc-800"
                          >
                            {d.hide_text ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                          </button>
                          <button
                            type="button"
                            title="Delete dialogue"
                            onClick={(e) => {
                              e.stopPropagation();
                              deleteDialogueMutation.mutate(d.dialogue_id);
                            }}
                            className="rounded p-1 text-rose-600 hover:bg-rose-50 dark:text-rose-400 dark:hover:bg-rose-950/30"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <Input
                          value={d.original_text ?? ""}
                          placeholder="Original text"
                          className="h-8 text-xs"
                          onChange={(e) =>
                            patchDialogueMutation.mutate({
                              dialogueId: d.dialogue_id,
                              patch: { original_text: e.target.value },
                            })
                          }
                        />
                        {d.romanization && (
                          <div className="px-1 text-[11px] italic text-zinc-500 dark:text-zinc-400">
                            {d.romanization}
                          </div>
                        )}
                        <Textarea
                          rows={2}
                          value={d.translated_text ?? ""}
                          placeholder="Translated text"
                          className="min-h-[56px] text-sm"
                          onChange={(e) =>
                            patchDialogueMutation.mutate({
                              dialogueId: d.dialogue_id,
                              patch: { translated_text: e.target.value },
                            })
                          }
                        />
                        <div className="flex items-center justify-between pt-1 text-[11px] text-zinc-500">
                          <span className="font-mono tabular-nums">
                            [{d.box_location.join(", ")}]
                          </span>
                          <div className="flex items-center gap-3">
                            <label className="flex items-center gap-1">
                              <input
                                type="checkbox"
                                checked={Boolean(d.hide_redraw)}
                                onChange={(e) =>
                                  patchDialogueMutation.mutate({
                                    dialogueId: d.dialogue_id,
                                    patch: { hide_redraw: e.target.checked },
                                  })
                                }
                              />
                              Hide Redraw
                            </label>
                            <label className="flex items-center gap-1">
                              <input
                                type="checkbox"
                                checked={Boolean(d.no_redraw)}
                                onChange={(e) =>
                                  patchDialogueMutation.mutate({
                                    dialogueId: d.dialogue_id,
                                    patch: { no_redraw: e.target.checked },
                                  })
                                }
                              />
                              No Redraw
                            </label>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          )}

          {/* Tab 2: Typesetting Inspector */}
          {inspectorTab === "typeset" && page && (
            <div className="flex-1 space-y-4 overflow-y-auto p-4">
              {!selectedDialogue ? (
                <div className="rounded-xl border border-dashed border-zinc-300 p-6 text-center text-xs text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
                  Select a dialogue box on the canvas or in the Dialogues tab to edit typesetting.
                </div>
              ) : (
                <>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                      Dialogue #{selectedDialogue.dialogue_id} Typography
                    </span>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() =>
                        patchDialogueMutation.mutate({
                          dialogueId: selectedDialogue.dialogue_id,
                          patch: { text_frame: null, text_rotation: 0 },
                        })
                      }
                    >
                      <RotateCcw className="h-3.5 w-3.5" />
                      Reset Frame
                    </Button>
                  </div>

                  <div>
                    <label className="mb-1 block text-xs text-zinc-500">Font Family</label>
                    <Select
                      value={selectedDialogue.overrides?.font_path || "Plus Jakarta Sans, sans-serif"}
                      onChange={(e) =>
                        patchDialogueMutation.mutate({
                          dialogueId: selectedDialogue.dialogue_id,
                          patch: {
                            overrides: { ...selectedDialogue.overrides, font_path: e.target.value },
                          },
                        })
                      }
                    >
                      <option value="Plus Jakarta Sans, sans-serif">Plus Jakarta Sans</option>
                      <option value="DM Sans, sans-serif">DM Sans</option>
                      <option value="Comic Sans MS, cursive">Comic Style</option>
                      <option value="Georgia, serif">Editorial Serif</option>
                    </Select>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="mb-1 block text-xs text-zinc-500">Font Size (px)</label>
                      <Input
                        type="number"
                        min={8}
                        max={96}
                        value={selectedDialogue.overrides?.font_size ?? 18}
                        onChange={(e) =>
                          patchDialogueMutation.mutate({
                            dialogueId: selectedDialogue.dialogue_id,
                            patch: {
                              overrides: {
                                ...selectedDialogue.overrides,
                                font_size: Number(e.target.value) || 18,
                              },
                            },
                          })
                        }
                      />
                    </div>
                    <div>
                      <label className="mb-1 block text-xs text-zinc-500">Text Color</label>
                      <input
                        type="color"
                        value={selectedDialogue.overrides?.color || "#141414"}
                        onChange={(e) =>
                          patchDialogueMutation.mutate({
                            dialogueId: selectedDialogue.dialogue_id,
                            patch: {
                              overrides: { ...selectedDialogue.overrides, color: e.target.value },
                            },
                          })
                        }
                        className="h-10 w-full cursor-pointer rounded-lg border border-zinc-200 bg-white p-1 dark:border-zinc-800 dark:bg-zinc-950"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="mb-1 block text-xs text-zinc-500">Line Spacing</label>
                      <Input
                        type="number"
                        step="0.05"
                        min={0.7}
                        max={2.5}
                        value={selectedDialogue.overrides?.line_spacing ?? 1.1}
                        onChange={(e) =>
                          patchDialogueMutation.mutate({
                            dialogueId: selectedDialogue.dialogue_id,
                            patch: {
                              overrides: {
                                ...selectedDialogue.overrides,
                                line_spacing: Number(e.target.value) || 1.1,
                              },
                            },
                          })
                        }
                      />
                    </div>
                    <div>
                      <label className="mb-1 block text-xs text-zinc-500">Rotation (deg)</label>
                      <Input
                        type="number"
                        min={-180}
                        max={180}
                        value={selectedDialogue.text_rotation ?? 0}
                        onChange={(e) =>
                          patchDialogueMutation.mutate({
                            dialogueId: selectedDialogue.dialogue_id,
                            patch: { text_rotation: Number(e.target.value) || 0 },
                          })
                        }
                      />
                    </div>
                  </div>

                  <div className="flex items-center justify-between gap-2 pt-1">
                    <div className="flex items-center gap-1">
                      {(["left", "center", "right"] as const).map((al) => (
                        <Button
                          key={al}
                          size="icon"
                          variant={
                            (selectedDialogue.overrides?.align || "center") === al
                              ? "default"
                              : "outline"
                          }
                          onClick={() =>
                            patchDialogueMutation.mutate({
                              dialogueId: selectedDialogue.dialogue_id,
                              patch: {
                                overrides: { ...selectedDialogue.overrides, align: al },
                              },
                            })
                          }
                        >
                          {al === "left" ? (
                            <AlignLeft className="h-4 w-4" />
                          ) : al === "center" ? (
                            <AlignCenter className="h-4 w-4" />
                          ) : (
                            <AlignRight className="h-4 w-4" />
                          )}
                        </Button>
                      ))}
                    </div>
                    <Button
                      size="sm"
                      variant={selectedDialogue.overrides?.bold ? "default" : "outline"}
                      onClick={() =>
                        patchDialogueMutation.mutate({
                          dialogueId: selectedDialogue.dialogue_id,
                          patch: {
                            overrides: {
                              ...selectedDialogue.overrides,
                              bold: !selectedDialogue.overrides?.bold,
                            },
                          },
                        })
                      }
                    >
                      <Bold className="h-4 w-4" />
                      Bold
                    </Button>
                  </div>

                  <div className="border-t border-zinc-200 pt-4 dark:border-zinc-800">
                    <Button variant="secondary" className="w-full" onClick={applyFormattingToAll}>
                      Apply Style to All Dialogues
                    </Button>
                    <Button
                      variant="outline"
                      className="mt-2 w-full"
                      disabled={redrawMutation.isPending}
                      onClick={() => redrawMutation.mutate()}
                    >
                      Run Server LaMa Redraw
                    </Button>
                  </div>
                </>
              )}
            </div>
          )}

          {/* Tab 3: Tools (Export, Find & Replace, Glossary, Assistant Chat) */}
          {inspectorTab === "tools" && page && project && (
            <div className="flex-1 space-y-6 overflow-y-auto p-4">
              {/* Export Section */}
              <div className="space-y-2.5">
                <h3 className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                  Export Current Page &amp; Project
                </h3>
                <div className="grid grid-cols-3 gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => exportPageRaster(page, "png")}
                  >
                    <Download className="h-3.5 w-3.5" />
                    PNG
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => exportPageRaster(page, "jpg")}
                  >
                    <Download className="h-3.5 w-3.5" />
                    JPG
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => exportPagePsd(page)}
                  >
                    <Layers className="h-3.5 w-3.5" />
                    PSD
                  </Button>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() =>
                      api.downloadProjectExport(project.id, "json", project.name).catch((e) => toast.error(e.message))
                    }
                  >
                    <FileText className="h-3.5 w-3.5" />
                    Project JSON
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() =>
                      api.downloadProjectExport(project.id, "txt", project.name).catch((e) => toast.error(e.message))
                    }
                  >
                    <FileText className="h-3.5 w-3.5" />
                    Script TXT
                  </Button>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  <Button
                    size="sm"
                    variant="secondary"
                    disabled={exporting}
                    onClick={() => handleBatchExportZip("png")}
                  >
                    <FileArchive className="h-3.5 w-3.5" />
                    ZIP PNG
                  </Button>
                  <Button
                    size="sm"
                    variant="secondary"
                    disabled={exporting}
                    onClick={() => handleBatchExportZip("jpg")}
                  >
                    <FileArchive className="h-3.5 w-3.5" />
                    ZIP JPG
                  </Button>
                  <Button
                    size="sm"
                    variant="secondary"
                    disabled={exporting}
                    onClick={() => handleBatchExportZip("psd")}
                  >
                    <FileArchive className="h-3.5 w-3.5" />
                    ZIP PSD
                  </Button>
                </div>
              </div>

              {/* Find & Replace */}
              <div className="space-y-2 border-t border-zinc-200 pt-4 dark:border-zinc-800">
                <h3 className="flex items-center gap-1.5 text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                  <Search className="h-3.5 w-3.5" />
                  Find &amp; Replace (Project)
                </h3>
                <Input
                  value={findText}
                  onChange={(e) => setFindText(e.target.value)}
                  placeholder="Find text..."
                  className="h-8 text-xs"
                />
                <Input
                  value={replaceText}
                  onChange={(e) => setReplaceText(e.target.value)}
                  placeholder="Replace with..."
                  className="h-8 text-xs"
                />
                <div className="flex items-center justify-between">
                  <label className="flex items-center gap-1.5 text-xs text-zinc-500">
                    <input
                      type="checkbox"
                      checked={findCase}
                      onChange={(e) => setFindCase(e.target.checked)}
                    />
                    Case sensitive
                  </label>
                  <Button
                    size="sm"
                    disabled={!findText || findReplaceMutation.isPending}
                    onClick={() => findReplaceMutation.mutate()}
                  >
                    Replace All
                  </Button>
                </div>
              </div>

              {/* Project Glossary */}
              <div className="space-y-2 border-t border-zinc-200 pt-4 dark:border-zinc-800">
                <h3 className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                  Glossary ({Object.keys(project.glossary).length})
                </h3>
                <div className="flex gap-1.5">
                  <Input
                    value={glossarySrc}
                    onChange={(e) => setGlossarySrc(e.target.value)}
                    placeholder="Source term"
                    className="h-8 text-xs"
                  />
                  <Input
                    value={glossaryDst}
                    onChange={(e) => setGlossaryDst(e.target.value)}
                    placeholder="Target"
                    className="h-8 text-xs"
                  />
                  <Button
                    size="sm"
                    onClick={async () => {
                      if (!glossarySrc.trim()) return;
                      const next = { ...project.glossary, [glossarySrc.trim()]: glossaryDst.trim() };
                      await api.updateProject(project.id, { glossary: next });
                      qc.invalidateQueries({ queryKey: ["project", project.id] });
                      setGlossarySrc("");
                      setGlossaryDst("");
                      toast.success("Glossary updated");
                    }}
                  >
                    Add
                  </Button>
                </div>
                {Object.entries(project.glossary).length > 0 && (
                  <div className="max-h-28 space-y-1 overflow-y-auto text-xs">
                    {Object.entries(project.glossary).map(([k, v]) => (
                      <div
                        key={k}
                        className="flex items-center justify-between rounded bg-zinc-100 px-2 py-1 dark:bg-zinc-800"
                      >
                        <span className="truncate">
                          {k} → {v}
                        </span>
                        <button
                          type="button"
                          onClick={async () => {
                            const next = { ...project.glossary };
                            delete next[k];
                            await api.updateProject(project.id, { glossary: next });
                            qc.invalidateQueries({ queryKey: ["project", project.id] });
                          }}
                          className="text-rose-600 dark:text-rose-400"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Translation Assistant Chat */}
              <div className="space-y-2 border-t border-zinc-200 pt-4 dark:border-zinc-800">
                <div className="flex items-center justify-between">
                  <h3 className="flex items-center gap-1.5 text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                    <MessageSquare className="h-3.5 w-3.5" />
                    Translation Chat
                  </h3>
                  <Select
                    className="h-7 w-36 py-0 text-xs"
                    value={chatCharacter}
                    onChange={(e) =>
                      setChatCharacter(e.target.value as "assistant" | "sengoku_yuno")
                    }
                  >
                    <option value="assistant">Assistant</option>
                    <option value="sengoku_yuno">Sengoku Yuno</option>
                  </Select>
                </div>
                {chatHistory.length > 0 && (
                  <div className="max-h-36 space-y-2 overflow-y-auto rounded-lg border border-zinc-200 bg-zinc-50 p-2 text-xs dark:border-zinc-800 dark:bg-zinc-950">
                    {chatHistory.map((m, i) => (
                      <div key={i} className={m.role === "user" ? "text-right font-medium" : "text-zinc-700 dark:text-zinc-300"}>
                        {m.text}
                      </div>
                    ))}
                  </div>
                )}
                <form
                  className="flex gap-1.5"
                  onSubmit={(e) => {
                    e.preventDefault();
                    if (!chatInput.trim() || chatMutation.isPending) return;
                    const msg = chatInput.trim();
                    setChatHistory((h) => [...h, { role: "user", text: msg }]);
                    setChatInput("");
                    chatMutation.mutate(msg);
                  }}
                >
                  <Input
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Ask about a phrase or SFX..."
                    className="h-8 text-xs"
                  />
                  <Button size="sm" type="submit" disabled={chatMutation.isPending}>
                    Send
                  </Button>
                </form>
              </div>
            </div>
          )}
        </aside>
      </div>

      {/* Bottom Live Processing Bar (real WebSocket job progress) */}
      {activeJob && (
        <div className="flex items-center justify-between gap-4 border-t border-zinc-200 bg-white px-4 py-2 text-xs dark:border-zinc-800 dark:bg-zinc-900">
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <span className="font-medium text-zinc-900 dark:text-zinc-100 truncate">
              {activeJob.message || activeJob.kind}
            </span>
            <span className="font-mono text-zinc-500 tabular-nums shrink-0">
              {activeJob.done}/{activeJob.total}
            </span>
            <div className="h-1.5 flex-1 max-w-xs overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
              <div
                className="h-full bg-blue-600 transition-all"
                style={{
                  width: `${activeJob.total ? Math.round((activeJob.done / activeJob.total) * 100) : 0}%`,
                }}
              />
            </div>
          </div>
          <Button
            size="sm"
            variant="outline"
            onClick={() => api.cancelJob(activeJob.id)}
          >
            Cancel
          </Button>
        </div>
      )}

      {/* Grammar Analysis Dialog */}
      <Dialog
        open={grammarDialog.open}
        onOpenChange={(open) => setGrammarDialog((s) => ({ ...s, open }))}
      >
        <DialogContent className="max-h-[85vh] overflow-y-auto">
          <DialogTitle>Grammar Breakdown</DialogTitle>
          <DialogDescription>
            Tokenized reading and grammar notes generated on the server.
          </DialogDescription>
          {grammarDialog.loading ? (
            <div className="py-8 text-center text-sm text-zinc-500">Analyzing dialogue...</div>
          ) : grammarDialog.error ? (
            <div className="mt-4 rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700 dark:border-rose-900 dark:bg-rose-950/30 dark:text-rose-300">
              {grammarDialog.error}
            </div>
          ) : grammarDialog.data ? (
            <div className="mt-4 space-y-4 text-sm">
              <div className="rounded-lg bg-zinc-100 p-3 dark:bg-zinc-800">
                <div className="font-medium">{grammarDialog.data.source_text}</div>
                <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-400">
                  {grammarDialog.data.summary}
                </p>
              </div>
              <div className="space-y-1.5">
                <h4 className="text-xs font-semibold text-zinc-500">Tokens</h4>
                <div className="divide-y divide-zinc-200 rounded-lg border border-zinc-200 dark:divide-zinc-800 dark:border-zinc-800">
                  {grammarDialog.data.tokens.map((t, i) => (
                    <div key={i} className="flex items-center justify-between gap-2 px-3 py-2 text-xs">
                      <div>
                        <span className="font-semibold">{t.surface}</span>{" "}
                        <span className="text-zinc-500">({t.reading})</span>
                      </div>
                      <div className="text-right text-zinc-600 dark:text-zinc-400">
                        {t.part_of_speech} · {t.meaning}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              {grammarDialog.data.grammar_points.length > 0 && (
                <div className="space-y-1.5">
                  <h4 className="text-xs font-semibold text-zinc-500">Grammar Points</h4>
                  {grammarDialog.data.grammar_points.map((g, i) => (
                    <div key={i} className="rounded-lg border border-zinc-200 p-2.5 text-xs dark:border-zinc-800">
                      <div className="font-semibold">{g.pattern}</div>
                      <div className="mt-0.5 text-zinc-600 dark:text-zinc-400">{g.explanation}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : null}
        </DialogContent>
      </Dialog>
    </div>
  );
}
