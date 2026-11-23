import { GoogleGenAI, ThinkingLevel, Type } from "@google/genai";
import { Router } from "express";
import { DEFAULT_GEMINI_MODEL } from "../config.js";
import {
  cleanBox,
  dialogueOut,
  effectiveProjectSettings,
  eventHub,
  getEffectiveApiKey,
  jobManager,
  pagesMap,
  projectsMap,
} from "../store.js";
import type { Box, Dialogue, GrammarAnalysisResult, Page, Project, UserSettings } from "../types.js";

export const translationRouter = Router();
export const jobsRouter = Router();
export const redrawRouter = Router();

function createAiClient(apiKey: string) {
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

function mapThinkingLevel(level: UserSettings["gemini_thinking_level"]): ThinkingLevel {
  switch (level) {
    case "high":
      return ThinkingLevel.HIGH;
    case "medium":
      return ThinkingLevel.MEDIUM;
    default:
      return ThinkingLevel.LOW;
  }
}

interface RawGeminiDialogue {
  box_location?: number[];
  original_text?: string;
  romanization?: string;
  translated_text?: string;
  confidence?: number;
}

const DIALOGUE_ARRAY_SCHEMA = {
  type: Type.ARRAY,
  items: {
    type: Type.OBJECT,
    properties: {
      box_location: {
        type: Type.ARRAY,
        items: { type: Type.NUMBER },
        description: "Bounding box [x, y, width, height] in original image pixels.",
      },
      original_text: {
        type: Type.STRING,
        description: "Original source text extracted from the speech bubble.",
      },
      romanization: {
        type: Type.STRING,
        description: "Romanization (romaji/pinyin/romanized reading) of the original text.",
      },
      translated_text: {
        type: Type.STRING,
        description: "Natural translation in the target language.",
      },
      confidence: {
        type: Type.NUMBER,
        description: "Recognition confidence between 0 and 1.",
      },
    },
    required: ["box_location", "original_text", "romanization", "translated_text"],
  },
};

async function translatePageWithGemini(
  apiKey: string,
  page: Page,
  project: Project,
  settings: UserSettings,
  previousContext: string[],
  regionBox?: Box
): Promise<Dialogue[]> {
  const ai = createAiClient(apiKey);
  const glossaryEntries = Object.entries({ ...settings.custom_glossary, ...project.glossary });
  const glossaryBlock = glossaryEntries
    .slice(0, 60)
    .map(([src, dst]) => `- ${src} => ${dst}`)
    .join("\n");

  const modeNote =
    settings.translation_backend === "yolo_ocr_gemini"
      ? "Mode A (OCR + Context Rescue): Transcribe every character faithfully with confidence score, rescue unclear vertical manga text, and translate."
      : "Mode B (Vision Direct): Detect speech bubbles in manga reading order (right-to-left, top-to-bottom for Japanese) and read + translate each bubble directly.";

  const regionInstruction = regionBox
    ? `Detect and translate ONLY the speech bubble / text inside pixel box [x=${regionBox[0]}, y=${regionBox[1]}, w=${regionBox[2]}, h=${regionBox[3]}] of this ${page.width}x${page.height} image.`
    : `Detect all speech bubbles and dialogue boxes across the full ${page.width}x${page.height} manga page in proper manga reading order.`;

  const prompt = [
    `You are an expert manga translator. Translate from "${settings.source_lang}" to "${settings.target_lang}".`,
    modeNote,
    settings.keep_honorifics
      ? "Preserve source honorifics (-san, -kun, -chan, -sama, -senpai, -sensei, etc.) in the translation."
      : "Adapt honorifics naturally into the target language.",
    project.work_title ? `Series title: ${project.work_title}` : "",
    glossaryBlock ? `Glossary (must enforce consistently):\n${glossaryBlock}` : "",
    previousContext.length > 0
      ? `Previous page dialogue context for continuity:\n${previousContext.slice(-6).join("\n")}`
      : "",
    regionInstruction,
    `Return exact pixel bounding boxes [x, y, width, height] inside [0..${page.width}, 0..${page.height}].`,
  ]
    .filter(Boolean)
    .join("\n\n");

  const response = await ai.models.generateContent({
    model: DEFAULT_GEMINI_MODEL,
    contents: {
      parts: [
        {
          inlineData: {
            mimeType: page.mime_type,
            data: page.image_buffer.toString("base64"),
          },
        },
        { text: prompt },
      ],
    },
    config: {
      responseMimeType: "application/json",
      responseSchema: DIALOGUE_ARRAY_SCHEMA,
      thinkingConfig: {
        thinkingLevel: mapThinkingLevel(settings.gemini_thinking_level),
      },
    },
  });

  const rawText = (response.text ?? "[]").trim();
  const parsed = JSON.parse(rawText) as RawGeminiDialogue[];
  const items = Array.isArray(parsed) ? parsed : [];

  return items.map((item) => {
    const rawBox =
      Array.isArray(item.box_location) && item.box_location.length === 4
        ? item.box_location
        : regionBox || [Math.round(page.width * 0.1), Math.round(page.height * 0.1), 160, 80];
    const x = Math.max(0, Math.min(page.width - 10, Math.round(Number(rawBox[0]) || 0)));
    const y = Math.max(0, Math.min(page.height - 10, Math.round(Number(rawBox[1]) || 0)));
    const w = Math.max(20, Math.min(page.width - x, Math.round(Number(rawBox[2]) || 120)));
    const h = Math.max(20, Math.min(page.height - y, Math.round(Number(rawBox[3]) || 60)));

    return {
      dialogue_id: page.next_dialogue_id++,
      box_location: [x, y, w, h],
      original_text: String(item.original_text ?? ""),
      romanization: String(item.romanization ?? ""),
      translated_text: String(item.translated_text ?? ""),
      confidence: typeof item.confidence === "number" ? item.confidence : undefined,
      edited: false,
      text_frame: null,
      text_rotation: 0,
      overrides: {},
      hide_text: false,
      hide_redraw: false,
      redraw_only: false,
      no_redraw: false,
    };
  });
}

function startTranslateJob(project: Project, pageIds: string[], keepManual: boolean | null): string {
  const apiKey = getEffectiveApiKey();
  if (!apiKey) {
    const err = new Error("GEMINI_API_KEY is not configured in server secrets") as Error & { status?: number };
    err.status = 400;
    throw err;
  }

  const job = jobManager.create(project.id, "translate", pageIds.length, { keep_manual: keepManual }, pageIds);
  const settings = effectiveProjectSettings(project);

  void (async () => {
    jobManager.start(job.id, "Translating pages via Gemini Vision");
    const rollingContext: string[] = [];

    for (const pid of pageIds) {
      const currentJob = jobManager.get(job.id);
      if (!currentJob || currentJob.status === "cancelled") return;

      const page = pagesMap.get(pid);
      if (!page) {
        jobManager.updatePage(job.id, pid, "failed", "Page not found");
        continue;
      }

      page.status = "translating";
      page.error_message = null;
      jobManager.updatePage(job.id, pid, "processing", `Translating ${page.page_name}`);
      eventHub.publish(project.id, { type: "page_changed", page_id: page.id });

      try {
        const translated = await translatePageWithGemini(
          apiKey,
          page,
          project,
          settings,
          settings.sequential_context_mode ? rollingContext : []
        );

        if (keepManual) {
          const manual = page.dialogues.filter((d) => d.edited);
          page.dialogues = [...manual, ...translated];
        } else {
          page.dialogues = translated;
        }
        page.status = "done";
        for (const d of page.dialogues.slice(-4)) {
          if (d.original_text || d.translated_text) {
            rollingContext.push(`${d.original_text} -> ${d.translated_text}`);
          }
        }
        jobManager.updatePage(job.id, pid, "completed", `${page.dialogues.length} dialogues`);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        page.status = "error";
        page.error_message = msg;
        jobManager.updatePage(job.id, pid, "failed", msg);
      }

      eventHub.publish(project.id, { type: "page_changed", page_id: page.id });
    }

    const latest = jobManager.get(job.id);
    if (latest && latest.status === "processing") {
      jobManager.finish(
        job.id,
        latest.failed > 0 && latest.done === latest.failed ? "failed" : "completed",
        latest.failed > 0 ? `Completed with ${latest.failed} failed page(s)` : "Translation complete"
      );
    }
  })();

  return job.id;
}

translationRouter.post("/projects/:project_id/translate", (req, res) => {
  const project = projectsMap.get(req.params.project_id);
  if (!project) {
    res.status(404).json({ detail: "Project not found" });
    return;
  }
  const body = req.body ?? {};
  const pageIds: string[] = Array.isArray(body.page_ids) ? body.page_ids : [...project.page_ids];
  if (pageIds.length === 0) {
    res.status(400).json({ detail: "Project has no pages to translate" });
    return;
  }
  const keepManual = body.keep_manual !== undefined && body.keep_manual !== null ? Boolean(body.keep_manual) : null;
  const jobId = startTranslateJob(project, pageIds, keepManual);
  res.status(202).json({ job_id: jobId });
});

translationRouter.post("/pages/:page_id/translate", (req, res) => {
  const page = pagesMap.get(req.params.page_id);
  if (!page) {
    res.status(404).json({ detail: "Page not found" });
    return;
  }
  const project = projectsMap.get(page.project_id);
  if (!project) {
    res.status(404).json({ detail: "Project not found" });
    return;
  }
  const body = req.body ?? {};
  const keepManual = body.keep_manual !== undefined && body.keep_manual !== null ? Boolean(body.keep_manual) : null;
  const jobId = startTranslateJob(project, [page.id], keepManual);
  res.status(202).json({ job_id: jobId });
});

translationRouter.post("/pages/:page_id/translate-region", async (req, res) => {
  const page = pagesMap.get(req.params.page_id);
  if (!page) {
    res.status(404).json({ detail: "Page not found" });
    return;
  }
  const project = projectsMap.get(page.project_id);
  if (!project) {
    res.status(404).json({ detail: "Project not found" });
    return;
  }
  const apiKey = getEffectiveApiKey();
  if (!apiKey) {
    res.status(400).json({ detail: "GEMINI_API_KEY is not configured in server secrets" });
    return;
  }
  const box = cleanBox(req.body?.box, "box");
  const add = req.body?.add !== undefined ? Boolean(req.body.add) : true;
  try {
    const settings = effectiveProjectSettings(project);
    const results = await translatePageWithGemini(apiKey, page, project, settings, [], box);
    const dialogue: Dialogue = results[0] ?? {
      dialogue_id: page.next_dialogue_id++,
      box_location: box,
      original_text: "",
      translated_text: "",
      romanization: "",
      edited: false,
      text_frame: null,
      text_rotation: 0,
      overrides: {},
      hide_text: false,
      hide_redraw: false,
      redraw_only: false,
      no_redraw: false,
    };
    if (add) {
      page.dialogues.push(dialogue);
      eventHub.publish(project.id, { type: "page_changed", page_id: page.id });
    }
    res.json(dialogueOut(dialogue));
  } catch (err) {
    res.status(502).json({ detail: `Region translation error: ${err instanceof Error ? err.message : String(err)}` });
  }
});

translationRouter.post("/pages/:page_id/detect-regions", async (req, res) => {
  const page = pagesMap.get(req.params.page_id);
  if (!page) {
    res.status(404).json({ detail: "Page not found" });
    return;
  }
  const apiKey = getEffectiveApiKey();
  if (!apiKey) {
    res.status(400).json({ detail: "GEMINI_API_KEY is not configured in server secrets" });
    return;
  }
  try {
    const ai = createAiClient(apiKey);
    const response = await ai.models.generateContent({
      model: DEFAULT_GEMINI_MODEL,
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: page.mime_type,
              data: page.image_buffer.toString("base64"),
            },
          },
          {
            text: `Detect all speech bubbles and text regions in this ${page.width}x${page.height} manga page. Return ONLY bounding boxes [x, y, width, height] in pixels and original_text.`,
          },
        ],
      },
      config: {
        responseMimeType: "application/json",
        responseSchema: DIALOGUE_ARRAY_SCHEMA,
      },
    });

    const items = JSON.parse((response.text ?? "[]").trim()) as RawGeminiDialogue[];
    const created: Dialogue[] = [];
    for (const item of Array.isArray(items) ? items : []) {
      const rawBox = Array.isArray(item.box_location) && item.box_location.length === 4 ? item.box_location : [40, 40, 120, 60];
      const d: Dialogue = {
        dialogue_id: page.next_dialogue_id++,
        box_location: [
          Math.max(0, Math.round(Number(rawBox[0]) || 0)),
          Math.max(0, Math.round(Number(rawBox[1]) || 0)),
          Math.max(20, Math.round(Number(rawBox[2]) || 120)),
          Math.max(20, Math.round(Number(rawBox[3]) || 60)),
        ],
        original_text: String(item.original_text ?? ""),
        romanization: String(item.romanization ?? ""),
        translated_text: String(item.translated_text ?? ""),
        edited: false,
        text_frame: null,
        text_rotation: 0,
        overrides: {},
        hide_text: false,
        hide_redraw: false,
        redraw_only: false,
        no_redraw: false,
      };
      page.dialogues.push(d);
      created.push(d);
    }
    eventHub.publish(page.project_id, { type: "page_changed", page_id: page.id });
    res.json({ detected: created.length, dialogues: created.map(dialogueOut) });
  } catch (err) {
    res.status(502).json({ detail: `Region detection failed: ${err instanceof Error ? err.message : String(err)}` });
  }
});

translationRouter.post("/pages/:page_id/grammar", async (req, res) => {
  const page = pagesMap.get(req.params.page_id);
  if (!page) {
    res.status(404).json({ detail: "Page not found" });
    return;
  }
  const project = projectsMap.get(page.project_id);
  if (!project) {
    res.status(404).json({ detail: "Project not found" });
    return;
  }
  const text = String(req.body?.text ?? "").trim();
  if (!text) {
    res.status(422).json({ detail: "text is required for grammar analysis" });
    return;
  }
  const cacheKey = `${project.source_lang}:${project.target_lang}:${text}`;
  if (project.grammar_cache[cacheKey]) {
    res.json({ cached: true, analysis: project.grammar_cache[cacheKey] });
    return;
  }
  const apiKey = getEffectiveApiKey();
  if (!apiKey) {
    res.status(400).json({ detail: "GEMINI_API_KEY is not configured in server secrets" });
    return;
  }
  try {
    const ai = createAiClient(apiKey);
    const response = await ai.models.generateContent({
      model: DEFAULT_GEMINI_MODEL,
      contents: `Analyze this ${project.source_lang} manga line and explain it in ${project.target_lang}:\n"${text}"`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            source_text: { type: Type.STRING },
            summary: { type: Type.STRING },
            tokens: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  surface: { type: Type.STRING },
                  reading: { type: Type.STRING },
                  part_of_speech: { type: Type.STRING },
                  meaning: { type: Type.STRING },
                },
                required: ["surface", "reading", "part_of_speech", "meaning"],
              },
            },
            grammar_points: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  pattern: { type: Type.STRING },
                  explanation: { type: Type.STRING },
                },
                required: ["pattern", "explanation"],
              },
            },
          },
          required: ["source_text", "summary", "tokens", "grammar_points"],
        },
      },
    });
    const analysis = JSON.parse((response.text ?? "{}").trim()) as GrammarAnalysisResult;
    project.grammar_cache[cacheKey] = analysis;
    res.json({ cached: false, analysis });
  } catch (err) {
    res.status(502).json({ detail: `Grammar analysis failed: ${err instanceof Error ? err.message : String(err)}` });
  }
});

// ---------------------------------------------------------------------------
// Jobs API
// ---------------------------------------------------------------------------
jobsRouter.get("/projects/:project_id/jobs", (req, res) => {
  res.json(jobManager.list(req.params.project_id));
});

jobsRouter.get("/jobs/:job_id", (req, res) => {
  const j = jobManager.get(req.params.job_id);
  if (!j) {
    res.status(404).json({ detail: "Job not found" });
    return;
  }
  res.json(j);
});

jobsRouter.post("/jobs/:job_id/cancel", (req, res) => {
  if (!jobManager.cancel(req.params.job_id)) {
    res.status(409).json({ detail: "Job not found or already finished" });
    return;
  }
  res.json(jobManager.get(req.params.job_id));
});

jobsRouter.post("/jobs/:job_id/retry-failed", (req, res) => {
  const j = jobManager.get(req.params.job_id);
  if (!j || j.kind !== "translate" || !j.project_id) {
    res.status(404).json({ detail: "Translation job not found" });
    return;
  }
  const failed = Object.entries(j.pages)
    .filter(([, v]) => v.status === "failed")
    .map(([pid]) => pid);
  if (failed.length === 0) {
    res.status(409).json({ detail: "No failed pages to retry" });
    return;
  }
  const project = projectsMap.get(j.project_id);
  if (!project) {
    res.status(404).json({ detail: "Project not found" });
    return;
  }
  const keepManual = j.params.keep_manual !== undefined && j.params.keep_manual !== null ? Boolean(j.params.keep_manual) : null;
  const jobId = startTranslateJob(project, failed, keepManual);
  res.status(202).json({ job_id: jobId });
});

// ---------------------------------------------------------------------------
// Redraw API (requires FASTAPI_BACKEND_URL for LaMa/OpenCV; original preview works natively)
// ---------------------------------------------------------------------------
redrawRouter.post("/:page_id/redraw", (_req, res) => {
  res.status(501).json({
    detail: "LaMa / OpenCV inpainting requires FASTAPI_BACKEND_URL connected to the Python Cloud Run backend. Client-side text overlay & canvas clean-fill editing remains active.",
  });
});

redrawRouter.post("/:page_id/redraw-dialogue", (_req, res) => {
  res.status(501).json({
    detail: "LaMa / OpenCV region redraw requires FASTAPI_BACKEND_URL connected to the Python Cloud Run backend.",
  });
});

redrawRouter.delete("/:page_id/redraw", (_req, res) => {
  res.status(501).json({ detail: "Python FastAPI backend not connected" });
});

redrawRouter.get("/:page_id/redraw-preview", (req, res) => {
  const page = pagesMap.get(req.params.page_id);
  if (!page) {
    res.status(404).json({ detail: "Page not found" });
    return;
  }
  const kind = String(req.query.kind || "translated");
  if (kind === "original") {
    res.setHeader("Content-Type", page.mime_type);
    res.send(page.image_buffer);
    return;
  }
  res.status(501).json({
    detail: "Server-side OpenCV layered render requires FASTAPI_BACKEND_URL. Use client canvas composition.",
  });
});

redrawRouter.get("/:page_id/layers", (_req, res) => {
  res.status(501).json({ detail: "Server-side OpenCV layer extraction requires FASTAPI_BACKEND_URL." });
});

redrawRouter.get("/:page_id/layers/file/*splat", (_req, res) => {
  res.status(501).json({ detail: "Python FastAPI backend not connected" });
});
