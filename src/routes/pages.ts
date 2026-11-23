import { Router } from "express";
import JSZip from "jszip";
import multer from "multer";
import { getServerConfig } from "../config.js";
import {
  DomainError,
  eventHub,
  inspectImageBuffer,
  naturalSortCompare,
  newId,
  nowIso,
  pageDetail,
  pagesMap,
  pageSummary,
  projectsMap,
  safeFilename,
} from "../store.js";
import type { FillItem, Page, StrokeItem } from "../types.js";

export const pagesRouter = Router();

const cfg = getServerConfig();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: Math.max(cfg.maxUploadMb, 100) * 1024 * 1024,
    files: 500,
  },
});

const IMAGE_EXT_RE = /\.(png|jpe?g|webp|bmp)$/i;

pagesRouter.post("/projects/:project_id/pages/upload", upload.array("files", 500), async (req, res) => {
  const project = projectsMap.get(String(req.params.project_id));
  if (!project) {
    res.status(404).json({ detail: "Project not found" });
    return;
  }
  const files = (req.files as Express.Multer.File[]) || [];
  if (files.length === 0) {
    res.status(422).json({ detail: "No files uploaded" });
    return;
  }

  const existingNames = new Set(
    project.page_ids
      .map((id) => pagesMap.get(id)?.page_name)
      .filter((n): n is string => Boolean(n))
  );

  const candidates: Array<{ name: string; buffer: Buffer }> = [];
  const rejected: Array<{ file: string; reason: string }> = [];

  for (const f of files) {
    const isZip =
      f.originalname?.toLowerCase().endsWith(".zip") ||
      f.mimetype === "application/zip" ||
      f.mimetype === "application/x-zip-compressed";

    if (isZip) {
      try {
        const zip = await JSZip.loadAsync(f.buffer);
        const entries = Object.values(zip.files)
          .filter((entry) => !entry.dir && !entry.name.startsWith("__MACOSX") && IMAGE_EXT_RE.test(entry.name))
          .sort((a, b) => naturalSortCompare(a.name, b.name));

        for (const entry of entries) {
          const buf = await entry.async("nodebuffer");
          candidates.push({ name: entry.name, buffer: buf });
        }
      } catch (err) {
        rejected.push({
          file: f.originalname || "archive.zip",
          reason: err instanceof Error ? err.message : "Invalid ZIP archive",
        });
      }
    } else {
      candidates.push({ name: f.originalname || "image.png", buffer: f.buffer });
    }
  }

  candidates.sort((a, b) => naturalSortCompare(a.name, b.name));

  const addedIds: string[] = [];
  const skippedDuplicates: string[] = [];

  for (const item of candidates) {
    try {
      const safeName = safeFilename(item.name);
      if (existingNames.has(safeName)) {
        skippedDuplicates.push(item.name);
        continue;
      }
      if (item.buffer.length > cfg.maxUploadMb * 1024 * 1024) {
        throw new DomainError(`File exceeds ${cfg.maxUploadMb}MB limit`, 400);
      }
      const { mime, width, height } = inspectImageBuffer(item.buffer);
      if (width * height > cfg.maxImagePixels) {
        throw new DomainError("Image dimensions exceed max allowed pixels", 400);
      }
      const page: Page = {
        id: newId(),
        project_id: project.id,
        page_name: safeName,
        width,
        height,
        mime_type: mime,
        image_buffer: item.buffer,
        status: "pending",
        error_message: null,
        dialogues: [],
        strokes: [],
        fills: [],
        direct_edited: false,
        redraw_status: "none",
        redraw_message: "",
        layout_boxes: {},
        layout_info: {},
        next_dialogue_id: 1,
      };
      pagesMap.set(page.id, page);
      project.page_ids.push(page.id);
      existingNames.add(safeName);
      addedIds.push(page.id);
    } catch (err) {
      rejected.push({
        file: item.name,
        reason: err instanceof Error ? err.message : String(err),
      });
    }
  }

  if (addedIds.length > 0) {
    // Keep project pages sorted naturally by filename
    project.page_ids.sort((a, b) => {
      const pa = pagesMap.get(a)?.page_name ?? "";
      const pb = pagesMap.get(b)?.page_name ?? "";
      return naturalSortCompare(pa, pb);
    });
    project.updated_at = nowIso();
    eventHub.publish(project.id, { type: "project", project_id: project.id });
  }

  res.status(201).json({
    added: addedIds.length,
    page_ids: addedIds,
    skipped_duplicates: skippedDuplicates,
    rejected,
  });
});

pagesRouter.get("/projects/:project_id/pages", (req, res) => {
  const project = projectsMap.get(req.params.project_id);
  if (!project) {
    res.status(404).json({ detail: "Project not found" });
    return;
  }
  const items = project.page_ids
    .map((pid) => pagesMap.get(pid))
    .filter((p): p is Page => Boolean(p))
    .map(pageSummary);
  res.json(items);
});

pagesRouter.get("/pages/:page_id", (req, res) => {
  const page = pagesMap.get(req.params.page_id);
  if (!page) {
    res.status(404).json({ detail: "Page not found" });
    return;
  }
  res.json(pageDetail(page));
});

pagesRouter.delete("/pages/:page_id", (req, res) => {
  const page = pagesMap.get(req.params.page_id);
  if (!page) {
    res.status(404).json({ detail: "Page not found" });
    return;
  }
  const project = projectsMap.get(page.project_id);
  if (project) {
    project.page_ids = project.page_ids.filter((id) => id !== page.id);
    project.updated_at = nowIso();
    eventHub.publish(project.id, { type: "project", project_id: project.id });
  }
  pagesMap.delete(page.id);
  res.status(204).send();
});

pagesRouter.put("/pages/:page_id/strokes", (req, res) => {
  const page = pagesMap.get(req.params.page_id);
  if (!page) {
    res.status(404).json({ detail: "Page not found" });
    return;
  }
  const strokes = req.body?.strokes;
  if (!Array.isArray(strokes)) {
    res.status(422).json({ detail: "strokes must be an array" });
    return;
  }
  page.strokes = strokes as StrokeItem[];
  eventHub.publish(page.project_id, { type: "page_changed", page_id: page.id });
  res.json(pageDetail(page));
});

pagesRouter.put("/pages/:page_id/fills", (req, res) => {
  const page = pagesMap.get(req.params.page_id);
  if (!page) {
    res.status(404).json({ detail: "Page not found" });
    return;
  }
  const fills = req.body?.fills;
  if (!Array.isArray(fills)) {
    res.status(422).json({ detail: "fills must be an array" });
    return;
  }
  page.fills = fills as FillItem[];
  eventHub.publish(page.project_id, { type: "page_changed", page_id: page.id });
  res.json(pageDetail(page));
});

pagesRouter.get("/pages/:page_id/image", (req, res) => {
  const page = pagesMap.get(req.params.page_id);
  if (!page) {
    res.status(404).json({ detail: "Page not found" });
    return;
  }
  res.setHeader("Content-Type", page.mime_type);
  res.setHeader("Cache-Control", "private, max-age=0, must-revalidate");
  res.send(page.image_buffer);
});

pagesRouter.get("/pages/:page_id/thumbnail", (req, res) => {
  const page = pagesMap.get(req.params.page_id);
  if (!page) {
    res.status(404).json({ detail: "Page not found" });
    return;
  }
  res.setHeader("Content-Type", page.mime_type);
  res.setHeader("Cache-Control", "private, max-age=60");
  res.send(page.image_buffer);
});
