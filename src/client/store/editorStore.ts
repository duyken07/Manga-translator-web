import { create } from "zustand";
import type { FillItem, JobRecord, StrokeItem } from "../lib/types";

export type NavTab = "projects" | "workspace" | "models" | "settings";
export type CanvasTool =
  | "select"
  | "pan"
  | "add_box"
  | "translate_region"
  | "pen"
  | "highlighter"
  | "eraser"
  | "fill_brush"
  | "magic_wand";

export type InspectorTab = "dialogues" | "typeset" | "tools";
export type ThemeMode = "system" | "light" | "dark";

interface HistorySnapshot {
  strokes: StrokeItem[];
  fills: FillItem[];
}

interface EditorState {
  activeTab: NavTab;
  activeProjectId: string | null;
  activePageId: string | null;
  selectedDialogueId: number | null;
  tool: CanvasTool;
  inspectorTab: InspectorTab;

  // Brush & wand parameters
  brushColor: string;
  brushSize: number;
  fillColor: string;
  fillMethod: "auto" | "lama" | "texture" | "opencv";
  wandTolerance: number;

  // Viewport & layers
  zoom: number;
  stageCommandSeq: number;
  stageCommand: "fit" | "one" | null;
  showTranslated: boolean;
  showRedraw: boolean;
  showBoxes: boolean;
  showStrokes: boolean;
  showFills: boolean;

  // Undo/redo for strokes & fills
  undoStack: HistorySnapshot[];
  redoStack: HistorySnapshot[];

  // Live WS job state
  liveJobs: JobRecord[];
  wsConnected: boolean;

  // UI theme & mobile sheet
  theme: ThemeMode;
  mobileSheetOpen: boolean;

  setActiveTab: (tab: NavTab) => void;
  setActiveProjectId: (id: string | null) => void;
  setActivePageId: (id: string | null) => void;
  setSelectedDialogueId: (id: number | null) => void;
  setTool: (tool: CanvasTool) => void;
  setInspectorTab: (tab: InspectorTab) => void;
  setBrushColor: (color: string) => void;
  setBrushSize: (size: number) => void;
  setFillColor: (color: string) => void;
  setFillMethod: (method: "auto" | "lama" | "texture" | "opencv") => void;
  setWandTolerance: (tolerance: number) => void;
  setZoom: (zoom: number) => void;
  triggerStageCommand: (cmd: "fit" | "one") => void;
  toggleLayer: (layer: "showTranslated" | "showRedraw" | "showBoxes" | "showStrokes" | "showFills") => void;
  pushHistory: (snap: HistorySnapshot) => void;
  popUndo: (current: HistorySnapshot) => HistorySnapshot | null;
  popRedo: (current: HistorySnapshot) => HistorySnapshot | null;
  clearHistory: () => void;
  setLiveJobs: (jobs: JobRecord[]) => void;
  upsertLiveJob: (job: JobRecord) => void;
  setWsConnected: (connected: boolean) => void;
  setTheme: (theme: ThemeMode) => void;
  setMobileSheetOpen: (open: boolean) => void;
}

export const useEditorStore = create<EditorState>((set, get) => ({
  activeTab: "projects",
  activeProjectId: null,
  activePageId: null,
  selectedDialogueId: null,
  tool: "select",
  inspectorTab: "dialogues",

  brushColor: "#141414",
  brushSize: 12,
  fillColor: "#ffffff",
  fillMethod: "auto",
  wandTolerance: 24,

  zoom: 1,
  stageCommandSeq: 0,
  stageCommand: null,
  showTranslated: true,
  showRedraw: true,
  showBoxes: true,
  showStrokes: true,
  showFills: true,

  undoStack: [],
  redoStack: [],

  liveJobs: [],
  wsConnected: false,

  theme: "system",
  mobileSheetOpen: true,

  setActiveTab: (activeTab) => set({ activeTab }),
  setActiveProjectId: (activeProjectId) =>
    set({ activeProjectId, activePageId: null, selectedDialogueId: null, undoStack: [], redoStack: [] }),
  setActivePageId: (activePageId) => set({ activePageId, selectedDialogueId: null, undoStack: [], redoStack: [] }),
  setSelectedDialogueId: (selectedDialogueId) => set({ selectedDialogueId }),
  setTool: (tool) => set({ tool }),
  setInspectorTab: (inspectorTab) => set({ inspectorTab }),
  setBrushColor: (brushColor) => set({ brushColor }),
  setBrushSize: (brushSize) => set({ brushSize }),
  setFillColor: (fillColor) => set({ fillColor }),
  setFillMethod: (fillMethod) => set({ fillMethod }),
  setWandTolerance: (wandTolerance) => set({ wandTolerance }),
  setZoom: (zoom) => set({ zoom }),
  triggerStageCommand: (stageCommand) => set((s) => ({ stageCommand, stageCommandSeq: s.stageCommandSeq + 1 })),
  toggleLayer: (layer) => set((s) => ({ [layer]: !s[layer] })),

  pushHistory: (snap) =>
    set((s) => ({
      undoStack: [...s.undoStack.slice(-24), snap],
      redoStack: [],
    })),

  popUndo: (current) => {
    const { undoStack, redoStack } = get();
    if (undoStack.length === 0) return null;
    const prev = undoStack[undoStack.length - 1];
    set({
      undoStack: undoStack.slice(0, -1),
      redoStack: [...redoStack, current],
    });
    return prev;
  },

  popRedo: (current) => {
    const { undoStack, redoStack } = get();
    if (redoStack.length === 0) return null;
    const next = redoStack[redoStack.length - 1];
    set({
      redoStack: redoStack.slice(0, -1),
      undoStack: [...undoStack, current],
    });
    return next;
  },

  clearHistory: () => set({ undoStack: [], redoStack: [] }),
  setLiveJobs: (liveJobs) => set({ liveJobs }),
  upsertLiveJob: (job) =>
    set((s) => {
      const idx = s.liveJobs.findIndex((j) => j.id === job.id);
      if (idx === -1) return { liveJobs: [job, ...s.liveJobs].slice(0, 20) };
      const copy = [...s.liveJobs];
      copy[idx] = job;
      return { liveJobs: copy };
    }),
  setWsConnected: (wsConnected) => set({ wsConnected }),
  setTheme: (theme) => set({ theme }),
  setMobileSheetOpen: (mobileSheetOpen) => set({ mobileSheetOpen }),
}));
