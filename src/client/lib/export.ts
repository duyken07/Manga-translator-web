import { writePsd, type Layer } from "ag-psd";
import JSZip from "jszip";
import { getApiToken } from "./api";
import type { Dialogue, FillItem, PageDetail, ProjectSummary, StrokeItem } from "./types";

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function loadImage(src: string): Promise<HTMLImageElement> {
  const token = getApiToken();
  let finalSrc = src;
  let objectUrl: string | null = null;
  if (token) {
    const res = await fetch(src, { headers: { Authorization: `Bearer ${token}` } });
    if (!res.ok) throw new Error(`Failed to fetch image (${res.status}): ${src}`);
    const blob = await res.blob();
    objectUrl = URL.createObjectURL(blob);
    finalSrc = objectUrl;
  }

  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      resolve(img);
    };
    img.onerror = () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      reject(new Error(`Failed to load image: ${src}`));
    };
    img.src = finalSrc;
  });
}

function makeCanvas(w: number, h: number): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = Math.max(1, w);
  c.height = Math.max(1, h);
  return c;
}

function wrapLines(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
  const paragraphs = text.split(/\r?\n/);
  const lines: string[] = [];
  for (const para of paragraphs) {
    const words = para.split(/\s+/).filter(Boolean);
    if (words.length === 0) {
      lines.push("");
      continue;
    }
    let current = words[0];
    for (let i = 1; i < words.length; i++) {
      const next = `${current} ${words[i]}`;
      if (ctx.measureText(next).width <= maxWidth) {
        current = next;
      } else {
        lines.push(current);
        current = words[i];
      }
    }
    lines.push(current);
  }
  return lines;
}

function computeFontSize(ctx: CanvasRenderingContext2D, d: Dialogue, w: number, h: number): number {
  if (d.overrides?.font_size && d.overrides.font_size > 4) {
    return d.overrides.font_size;
  }
  const fontFamily = d.overrides?.font_path || "Plus Jakarta Sans, sans-serif";
  const weight = d.overrides?.bold ? "700" : "600";
  const spacing = d.overrides?.line_spacing || 1.1;

  for (let size = Math.min(42, Math.max(12, Math.floor(h * 0.4))); size >= 10; size -= 1) {
    ctx.font = `${weight} ${size}px ${fontFamily}`;
    const lines = wrapLines(ctx, d.translated_text, Math.max(20, w - 12));
    const totalHeight = lines.length * size * spacing;
    if (totalHeight <= h - 8) return size;
  }
  return 11;
}

export function drawStrokesToCanvas(ctx: CanvasRenderingContext2D, strokes: StrokeItem[]) {
  for (const s of strokes) {
    if (s.hidden || s.points.length < 2) continue;
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = s.size;
    if (s.tool === "eraser") {
      ctx.globalCompositeOperation = "destination-out";
      ctx.strokeStyle = "rgba(0,0,0,1)";
    } else {
      ctx.globalAlpha = s.tool === "highlighter" ? 0.35 : s.opacity ?? 1;
      ctx.strokeStyle = s.color || "#141414";
    }
    ctx.beginPath();
    ctx.moveTo(s.points[0], s.points[1]);
    for (let i = 2; i < s.points.length; i += 2) {
      ctx.lineTo(s.points[i], s.points[i + 1]);
    }
    ctx.stroke();
    ctx.restore();
  }
}

export function drawFillsToCanvas(ctx: CanvasRenderingContext2D, fills: FillItem[]) {
  for (const f of fills) {
    if (f.hidden) continue;
    ctx.save();
    ctx.fillStyle = f.color || "#ffffff";
    ctx.strokeStyle = f.color || "#ffffff";
    if (f.box) {
      const [x, y, w, h] = f.box;
      ctx.fillRect(x, y, w, h);
    } else if (f.points && f.points.length >= 2) {
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.lineWidth = f.size || 24;
      ctx.beginPath();
      ctx.moveTo(f.points[0], f.points[1]);
      for (let i = 2; i < f.points.length; i += 2) {
        ctx.lineTo(f.points[i], f.points[i + 1]);
      }
      ctx.stroke();
    }
    ctx.restore();
  }
}

export function drawDialogueText(ctx: CanvasRenderingContext2D, d: Dialogue) {
  const frame = d.text_frame ?? d.box_location;
  const [x, y, w, h] = frame;
  const fontSize = computeFontSize(ctx, d, w, h);
  const fontFamily = d.overrides?.font_path || "Plus Jakarta Sans, sans-serif";
  const weight = d.overrides?.bold ? "700" : "600";
  const spacing = d.overrides?.line_spacing || 1.1;
  const align = d.overrides?.align || "center";
  const color = d.overrides?.color || "#141414";

  ctx.save();
  const cx = x + w / 2;
  const cy = y + h / 2;
  if (d.text_rotation) {
    ctx.translate(cx, cy);
    ctx.rotate((d.text_rotation * Math.PI) / 180);
    ctx.translate(-cx, -cy);
  }

  ctx.font = `${weight} ${fontSize}px ${fontFamily}`;
  ctx.fillStyle = color;
  ctx.textBaseline = "middle";
  ctx.textAlign = align;

  const lines = wrapLines(ctx, d.translated_text, Math.max(20, w - 12));
  const lineH = fontSize * spacing;
  const startY = cy - ((lines.length - 1) * lineH) / 2;
  const textX = align === "left" ? x + 6 : align === "right" ? x + w - 6 : cx;

  lines.forEach((line, idx) => {
    ctx.fillText(line, textX, startY + idx * lineH);
  });
  ctx.restore();
}

export async function renderPageCanvas(
  page: PageDetail,
  imageSrc: string,
  options: { showRedraw?: boolean; showText?: boolean } = {}
): Promise<HTMLCanvasElement> {
  const showRedraw = options.showRedraw ?? true;
  const showText = options.showText ?? true;

  const img = await loadImage(imageSrc);
  const w = page.width || img.naturalWidth;
  const h = page.height || img.naturalHeight;
  const canvas = makeCanvas(w, h);
  const ctx = canvas.getContext("2d")!;

  ctx.drawImage(img, 0, 0, w, h);

  if (showRedraw) {
    for (const d of page.dialogues) {
      if (d.hide_redraw || d.no_redraw) continue;
      const [x, y, bw, bh] = d.box_location;
      ctx.save();
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.roundRect(x, y, bw, bh, Math.min(bw, bh) * 0.25);
      ctx.fill();
      ctx.restore();
    }
    drawFillsToCanvas(ctx, page.fills || []);
  }

  drawStrokesToCanvas(ctx, page.strokes || []);

  if (showText) {
    for (const d of page.dialogues) {
      if (d.hide_text || d.redraw_only || !d.translated_text) continue;
      drawDialogueText(ctx, d);
    }
  }

  return canvas;
}

export async function exportPageRaster(page: PageDetail, format: "png" | "jpg") {
  const canvas = await renderPageCanvas(page, `/api/pages/${page.id}/image`);
  const mime = format === "jpg" ? "image/jpeg" : "image/png";
  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error("Canvas export failed"))),
      mime,
      format === "jpg" ? 0.92 : undefined
    );
  });
  const stem = page.page_name.replace(/\.[^.]+$/, "");
  downloadBlob(blob, `${stem}.${format}`);
}

async function buildPagePsdBuffer(page: PageDetail): Promise<ArrayBuffer> {
  const img = await loadImage(`/api/pages/${page.id}/image`);
  const w = page.width || img.naturalWidth;
  const h = page.height || img.naturalHeight;

  const baseCanvas = makeCanvas(w, h);
  baseCanvas.getContext("2d")!.drawImage(img, 0, 0, w, h);

  const redrawCanvas = makeCanvas(w, h);
  const rCtx = redrawCanvas.getContext("2d")!;
  for (const d of page.dialogues) {
    if (d.no_redraw) continue;
    const [x, y, bw, bh] = d.box_location;
    rCtx.fillStyle = "#ffffff";
    rCtx.beginPath();
    rCtx.roundRect(x, y, bw, bh, Math.min(bw, bh) * 0.25);
    rCtx.fill();
  }

  const fillsCanvas = makeCanvas(w, h);
  drawFillsToCanvas(fillsCanvas.getContext("2d")!, page.fills || []);

  const strokesCanvas = makeCanvas(w, h);
  drawStrokesToCanvas(strokesCanvas.getContext("2d")!, page.strokes || []);

  const textLayers: Layer[] = page.dialogues
    .filter((d) => !d.redraw_only && d.translated_text)
    .map((d) => {
      const c = makeCanvas(w, h);
      drawDialogueText(c.getContext("2d")!, d);
      return {
        name: `Text_${d.dialogue_id}`,
        hidden: Boolean(d.hide_text),
        canvas: c,
      };
    });

  const compositeCanvas = await renderPageCanvas(page, `/api/pages/${page.id}/image`);

  return writePsd({
    width: w,
    height: h,
    canvas: compositeCanvas,
    children: [
      { name: "Anh goc", canvas: baseCanvas },
      { name: "Redraw", canvas: redrawCanvas },
      { name: "Fills", canvas: fillsCanvas },
      { name: "Strokes", canvas: strokesCanvas },
      ...textLayers,
    ],
  });
}

export async function exportPagePsd(page: PageDetail) {
  const psdBuffer = await buildPagePsdBuffer(page);
  const stem = page.page_name.replace(/\.[^.]+$/, "");
  downloadBlob(new Blob([psdBuffer], { type: "application/octet-stream" }), `${stem}.psd`);
}

export async function exportProjectZip(
  project: ProjectSummary,
  pages: PageDetail[],
  format: "png" | "jpg" | "psd"
) {
  const zip = new JSZip();
  for (const p of pages) {
    const stem = p.page_name.replace(/\.[^.]+$/, "");
    if (format === "psd") {
      const psdBuf = await buildPagePsdBuffer(p);
      zip.file(`${stem}.psd`, psdBuf);
    } else {
      const canvas = await renderPageCanvas(p, `/api/pages/${p.id}/image`);
      const mime = format === "jpg" ? "image/jpeg" : "image/png";
      const blob = await new Promise<Blob>((resolve, reject) =>
        canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("Render failed"))), mime, 0.92)
      );
      zip.file(`${stem}.${format}`, blob);
    }
  }
  const out = await zip.generateAsync({ type: "blob" });
  downloadBlob(out, `${project.name.replace(/[^a-zA-Z0-9._-]/g, "_")}_${format}.zip`);
}
