export type Box = [number, number, number, number];

export interface DialogueOverrides {
  font_path?: string;
  font_size?: number;
  color?: string;
  bold?: boolean;
  line_spacing?: number;
  align?: "left" | "center" | "right";
}

export interface Dialogue {
  dialogue_id: number;
  box_location: Box;
  original_text: string;
  translated_text: string;
  romanization: string;
  confidence?: number;
  edited: boolean;
  text_frame: Box | null;
  text_rotation: number;
  overrides: DialogueOverrides & Record<string, unknown>;
  hide_text: boolean;
  hide_redraw: boolean;
  redraw_only: boolean;
  no_redraw: boolean;
}

export interface StrokeItem {
  id: string;
  tool: "pen" | "highlighter" | "eraser";
  color: string;
  size: number;
  opacity: number;
  points: number[];
  hidden?: boolean;
}

export interface FillItem {
  id: string;
  method: "auto" | "lama" | "texture" | "opencv";
  color?: string;
  box?: Box;
  points?: number[];
  size?: number;
  hidden?: boolean;
}

export interface Page {
  id: string;
  project_id: string;
  page_name: string;
  width: number;
  height: number;
  mime_type: string;
  image_buffer: Buffer;
  status: "pending" | "translating" | "done" | "error";
  error_message: string | null;
  dialogues: Dialogue[];
  strokes: StrokeItem[];
  fills: FillItem[];
  direct_edited: boolean;
  redraw_status: string;
  redraw_message: string;
  layout_boxes: Record<string, unknown>;
  layout_info: Record<string, unknown>;
  next_dialogue_id: number;
}

export interface Project {
  id: string;
  owner_id: string;
  name: string;
  source_lang: string;
  target_lang: string;
  work_title: string;
  glossary: Record<string, string>;
  grammar_cache: Record<string, GrammarAnalysisResult>;
  settings: Record<string, unknown>;
  page_ids: string[];
  created_at: string;
  updated_at: string;
}

export interface GrammarToken {
  surface: string;
  reading: string;
  part_of_speech: string;
  meaning: string;
  note?: string;
}

export interface GrammarAnalysisResult {
  source_text: string;
  summary: string;
  tokens: GrammarToken[];
  grammar_points: Array<{ pattern: string; explanation: string }>;
}

export type JobStatus = "pending" | "processing" | "completed" | "failed" | "cancelled";

export interface JobPageState {
  status: JobStatus;
  message: string;
}

export interface JobRecord {
  id: string;
  project_id: string | null;
  kind: string;
  status: JobStatus;
  total: number;
  done: number;
  failed: number;
  message: string;
  pages: Record<string, JobPageState>;
  params: Record<string, unknown>;
  created_at: string;
  finished_at: string | null;
}

export interface UserSettings {
  source_lang: string;
  target_lang: string;
  concurrency_limit: number;
  translation_backend: "yolo_ocr_gemini" | "yolo_gemini_vision";
  sequential_context_mode: boolean;
  keep_honorifics: boolean;
  enable_stray_text_removal: boolean;
  enable_redraw: boolean;
  paint_auto_redraw: boolean;
  paint_direct: boolean;
  lama_device: "cpu" | "cuda" | "mps";
  lama_threads: number;
  redraw_font: string;
  redraw_text_color: string;
  redraw_line_spacing: number;
  theme: "light" | "dark" | "system";
  use_koharu: boolean;
  koharu_imgsz: number;
  koharu_erase_sfx: boolean;
  koharu_use_yolo: boolean;
  ocr_force_cpu: boolean;
  ocr_gemini_rescue: boolean;
  ocr_low_conf: number;
  gemini_thinking_level: "minimal" | "low" | "medium" | "high";
  custom_glossary: Record<string, string>;
}

export interface ModelStatusItem {
  id: string;
  label: string;
  repo: string;
  state: "not_downloaded" | "downloading" | "downloaded" | "loading" | "ready" | "error";
  progress: number | null;
  error: string;
  version: string;
  optional: boolean;
  file: string;
}
