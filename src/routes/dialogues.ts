import { Router } from "express";
import { cleanBox, dialogueOut, DomainError, eventHub, nowIso, pagesMap, projectsMap } from "../store.js";
import type { Dialogue } from "../types.js";

export const dialoguesRouter = Router({ mergeParams: true });

const PATCH_ALLOWED = new Set([
  "original_text",
  "romanization",
  "translated_text",
  "box_location",
  "text_frame",
  "text_rotation",
  "overrides",
  "hide_text",
  "hide_redraw",
  "redraw_only",
  "no_redraw",
]);

dialoguesRouter.get("/", (req, res) => {
  const page = pagesMap.get((req.params as Record<string, string>).page_id);
  if (!page) {
    res.status(404).json({ detail: "Không tìm thấy trang" });
    return;
  }
  res.json(page.dialogues.map(dialogueOut));
});

dialoguesRouter.post("/", (req, res) => {
  const page = pagesMap.get((req.params as Record<string, string>).page_id);
  if (!page) {
    res.status(404).json({ detail: "Không tìm thấy trang" });
    return;
  }
  const body = req.body ?? {};
  const box = cleanBox(body.box_location, "box_location");
  const d: Dialogue = {
    dialogue_id: page.next_dialogue_id++,
    box_location: box,
    original_text: String(body.original_text ?? ""),
    translated_text: String(body.translated_text ?? ""),
    romanization: String(body.romanization ?? ""),
    edited: body.edited !== undefined ? Boolean(body.edited) : true,
    text_frame: null,
    text_rotation: 0,
    overrides: {},
    hide_text: false,
    hide_redraw: false,
    redraw_only: Boolean(body.redraw_only ?? false),
    no_redraw: Boolean(body.no_redraw ?? false),
  };
  page.dialogues.push(d);
  const proj = projectsMap.get(page.project_id);
  if (proj) proj.updated_at = nowIso();
  eventHub.publish(page.project_id, { type: "page_changed", page_id: page.id });
  res.status(201).json(dialogueOut(d));
});

dialoguesRouter.put("/order", (req, res) => {
  const page = pagesMap.get((req.params as Record<string, string>).page_id);
  if (!page) {
    res.status(404).json({ detail: "Không tìm thấy trang" });
    return;
  }
  const order = req.body?.order;
  if (!Array.isArray(order)) {
    res.status(422).json({ detail: "order phải là mảng dialogue_id" });
    return;
  }
  const byId = new Map(page.dialogues.map((d) => [d.dialogue_id, d]));
  const seen = new Set<number>();
  const reordered: Dialogue[] = [];
  for (const id of order) {
    const numId = Number(id);
    const d = byId.get(numId);
    if (d && !seen.has(numId)) {
      reordered.push(d);
      seen.add(numId);
    }
  }
  for (const d of page.dialogues) {
    if (!seen.has(d.dialogue_id)) reordered.push(d);
  }
  page.dialogues = reordered;
  eventHub.publish(page.project_id, { type: "page_changed", page_id: page.id });
  res.json(page.dialogues.map(dialogueOut));
});

dialoguesRouter.patch("/:dialogue_id", (req, res) => {
  const page = pagesMap.get((req.params as Record<string, string>).page_id);
  if (!page) {
    res.status(404).json({ detail: "Không tìm thấy trang" });
    return;
  }
  const did = Number(req.params.dialogue_id);
  const d = page.dialogues.find((item) => item.dialogue_id === did);
  if (!d) {
    res.status(404).json({ detail: "Không tìm thấy thoại" });
    return;
  }
  const body = req.body ?? {};
  const extraKeys = Object.keys(body).filter((k) => !PATCH_ALLOWED.has(k));
  if (extraKeys.length > 0) {
    res.status(422).json({ detail: `Trường không hợp lệ: ${extraKeys.join(", ")}` });
    return;
  }

  if (body.box_location !== undefined) {
    d.box_location = cleanBox(body.box_location, "box_location");
  }
  if (body.text_frame !== undefined) {
    d.text_frame = body.text_frame === null ? null : cleanBox(body.text_frame, "text_frame");
  }
  if (body.text_rotation !== undefined) {
    const rot = Number(body.text_rotation);
    if (Number.isNaN(rot)) throw new DomainError("text_rotation không hợp lệ", 422);
    d.text_rotation = rot;
  }
  if (body.overrides !== undefined && body.overrides && typeof body.overrides === "object") {
    d.overrides = { ...body.overrides };
  }
  if (body.translated_text !== undefined && String(body.translated_text) !== d.translated_text) {
    d.translated_text = String(body.translated_text);
    d.edited = true;
  }
  if (body.original_text !== undefined && String(body.original_text) !== d.original_text) {
    d.original_text = String(body.original_text);
    d.edited = true;
  }
  if (body.romanization !== undefined && String(body.romanization) !== d.romanization) {
    d.romanization = String(body.romanization);
    d.edited = true;
  }
  if (body.hide_text !== undefined) d.hide_text = Boolean(body.hide_text);
  if (body.hide_redraw !== undefined) d.hide_redraw = Boolean(body.hide_redraw);
  if (body.redraw_only !== undefined) d.redraw_only = Boolean(body.redraw_only);
  if (body.no_redraw !== undefined) d.no_redraw = Boolean(body.no_redraw);

  eventHub.publish(page.project_id, { type: "page_changed", page_id: page.id });
  res.json(dialogueOut(d));
});

dialoguesRouter.delete("/:dialogue_id", (req, res) => {
  const page = pagesMap.get((req.params as Record<string, string>).page_id);
  if (!page) {
    res.status(404).json({ detail: "Không tìm thấy trang" });
    return;
  }
  const did = Number(req.params.dialogue_id);
  const idx = page.dialogues.findIndex((item) => item.dialogue_id === did);
  if (idx === -1) {
    res.status(404).json({ detail: "Không tìm thấy thoại" });
    return;
  }
  page.dialogues.splice(idx, 1);
  eventHub.publish(page.project_id, { type: "page_changed", page_id: page.id });
  res.status(204).send();
});
