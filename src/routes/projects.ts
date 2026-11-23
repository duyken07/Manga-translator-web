import { Router } from "express";
import { PROJECT_OVERRIDABLE } from "../config.js";
import { eventHub, newId, nowIso, pagesMap, projectOut, projectsMap } from "../store.js";
import type { Page, Project } from "../types.js";

export const projectsRouter = Router();

const OVERRIDABLE_SET = new Set<string>(PROJECT_OVERRIDABLE);

projectsRouter.get("/", (_req, res) => {
  const list = Array.from(projectsMap.values())
    .sort((a, b) => b.updated_at.localeCompare(a.updated_at))
    .map(projectOut);
  res.json(list);
});

projectsRouter.post("/", (req, res) => {
  const body = req.body ?? {};
  const name = typeof body.name === "string" ? body.name.trim() : "";
  if (!name || name.length > 200) {
    res.status(422).json({ detail: "Project name must be 1..200 characters" });
    return;
  }
  const source_lang = typeof body.source_lang === "string" ? body.source_lang.slice(0, 16) : "ja";
  const target_lang = typeof body.target_lang === "string" ? body.target_lang.slice(0, 16) : "vi";
  const work_title = typeof body.work_title === "string" ? body.work_title.slice(0, 300) : "";
  const now = nowIso();
  const project: Project = {
    id: newId(),
    owner_id: "owner",
    name,
    source_lang,
    target_lang,
    work_title,
    glossary: {},
    grammar_cache: {},
    settings: {},
    page_ids: [],
    created_at: now,
    updated_at: now,
  };
  projectsMap.set(project.id, project);
  res.status(201).json(projectOut(project));
});

projectsRouter.get("/:project_id", (req, res) => {
  const p = projectsMap.get(req.params.project_id);
  if (!p) {
    res.status(404).json({ detail: "Project not found" });
    return;
  }
  res.json(projectOut(p));
});

projectsRouter.put("/:project_id", (req, res) => {
  const p = projectsMap.get(req.params.project_id);
  if (!p) {
    res.status(404).json({ detail: "Project not found" });
    return;
  }
  const body = req.body ?? {};
  if (body.name !== undefined) {
    const name = String(body.name).trim();
    if (!name || name.length > 200) {
      res.status(422).json({ detail: "Invalid project name" });
      return;
    }
    p.name = name;
  }
  if (body.source_lang !== undefined) p.source_lang = String(body.source_lang).slice(0, 16);
  if (body.target_lang !== undefined) p.target_lang = String(body.target_lang).slice(0, 16);
  if (body.work_title !== undefined) p.work_title = String(body.work_title).slice(0, 300);
  if (body.glossary !== undefined && body.glossary && typeof body.glossary === "object") {
    const clean: Record<string, string> = {};
    for (const [k, v] of Object.entries(body.glossary)) {
      if (String(k).trim()) clean[String(k).trim()] = String(v);
    }
    p.glossary = clean;
  }
  if (body.settings !== undefined) {
    if (!body.settings || typeof body.settings !== "object" || Array.isArray(body.settings)) {
      res.status(422).json({ detail: "settings must be an object" });
      return;
    }
    const bad = Object.keys(body.settings).filter((k) => !OVERRIDABLE_SET.has(k));
    if (bad.length > 0) {
      res.status(422).json({ detail: `Non-overridable project settings: ${bad.join(", ")}` });
      return;
    }
    p.settings = { ...body.settings };
  }
  p.updated_at = nowIso();
  res.json(projectOut(p));
});

projectsRouter.delete("/:project_id", (req, res) => {
  const p = projectsMap.get(req.params.project_id);
  if (!p) {
    res.status(404).json({ detail: "Project not found" });
    return;
  }
  for (const pid of p.page_ids) {
    pagesMap.delete(pid);
  }
  projectsMap.delete(p.id);
  res.status(204).send();
});

projectsRouter.post("/:project_id/find-replace", (req, res) => {
  const p = projectsMap.get(req.params.project_id);
  if (!p) {
    res.status(404).json({ detail: "Project not found" });
    return;
  }
  const find = typeof req.body?.find === "string" ? req.body.find : "";
  const replace = typeof req.body?.replace === "string" ? req.body.replace : "";
  const field = req.body?.field === "original_text" ? "original_text" : "translated_text";
  const caseSensitive = Boolean(req.body?.case_sensitive ?? false);
  const dryRun = Boolean(req.body?.dry_run ?? false);

  if (!find) {
    res.status(422).json({ detail: "find text cannot be empty" });
    return;
  }

  const escaped = find.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const regex = new RegExp(escaped, caseSensitive ? "g" : "gi");

  let matches = 0;
  let updatedDialogues = 0;
  const affectedPageIds: string[] = [];

  for (const pid of p.page_ids) {
    const page = pagesMap.get(pid);
    if (!page) continue;
    let pageChanged = false;
    for (const d of page.dialogues) {
      const current = d[field];
      const found = current.match(regex);
      if (found && found.length > 0) {
        matches += found.length;
        updatedDialogues += 1;
        pageChanged = true;
        if (!dryRun) {
          d[field] = current.replace(regex, replace);
          d.edited = true;
        }
      }
    }
    if (pageChanged) {
      affectedPageIds.push(page.id);
      if (!dryRun) {
        eventHub.publish(p.id, { type: "page_changed", page_id: page.id });
      }
    }
  }

  if (!dryRun && updatedDialogues > 0) {
    p.updated_at = nowIso();
  }

  res.json({ matches, updated_dialogues: updatedDialogues, affected_page_ids: affectedPageIds, dry_run: dryRun });
});

projectsRouter.get("/:project_id/export", (req, res) => {
  const p = projectsMap.get(req.params.project_id);
  if (!p) {
    res.status(404).json({ detail: "Project not found" });
    return;
  }
  const fmt = String(req.query.format || "json").toLowerCase();
  const pages = p.page_ids
    .map((pid) => pagesMap.get(pid))
    .filter((pg): pg is Page => Boolean(pg));

  if (fmt === "txt") {
    const lines: string[] = [];
    pages.forEach((pg, idx) => {
      lines.push(`=== Page ${idx + 1}: ${pg.page_name} ===`);
      for (const d of pg.dialogues) {
        if (d.hide_text) continue;
        lines.push(`#${d.dialogue_id} ${d.translated_text || d.original_text}`);
      }
      lines.push("");
    });
    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    res.setHeader("Content-Disposition", `attachment; filename="${p.name.replace(/[^a-zA-Z0-9._-]/g, "_")}.txt"`);
    res.send(lines.join("\n"));
    return;
  }

  // Desktop-compatible Project.to_json_dict schema
  const payload = {
    name: p.name,
    source_lang: p.source_lang,
    target_lang: p.target_lang,
    work_title: p.work_title,
    glossary: p.glossary,
    grammar_cache: p.grammar_cache,
    pages: pages.map((pg) => ({
      page_name: pg.page_name,
      status: pg.status,
      error_message: pg.error_message,
      direct_edited: pg.direct_edited,
      strokes: pg.strokes,
      fills: pg.fills,
      dialogues: pg.dialogues,
    })),
  };
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Content-Disposition", `attachment; filename="${p.name.replace(/[^a-zA-Z0-9._-]/g, "_")}.json"`);
  res.json(payload);
});
