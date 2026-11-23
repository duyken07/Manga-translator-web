import type Konva from "konva";
import * as React from "react";
import { Group, Image as KonvaImage, Layer, Line, Rect, Stage, Text, Transformer } from "react-konva";
import type { Box, Dialogue, FillItem, PageDetail, StrokeItem } from "@/lib/types";
import { useEditorStore } from "@/store/editorStore";
import { useAuthImageUrl } from "./AuthImage";

interface KonvaStageProps {
  page: PageDetail;
  onPatchDialogue: (dialogueId: number, patch: Partial<Dialogue>) => void;
  onCreateBox: (box: Box) => void;
  onTranslateRegion: (box: Box) => void;
  onSaveStrokes: (strokes: StrokeItem[]) => void;
  onSaveFills: (fills: FillItem[]) => void;
}

export function KonvaStage({
  page,
  onPatchDialogue,
  onCreateBox,
  onTranslateRegion,
  onSaveStrokes,
  onSaveFills,
}: KonvaStageProps) {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const stageRef = React.useRef<Konva.Stage>(null);
  const trRef = React.useRef<Konva.Transformer>(null);
  const nodeMapRef = React.useRef<Map<number, Konva.Group>>(new Map());

  const [size, setSize] = React.useState({ width: 800, height: 600 });
  const [imageEl, setImageEl] = React.useState<HTMLImageElement | null>(null);
  const [stagePos, setStagePos] = React.useState({ x: 0, y: 0 });
  const [stageScale, setStageScale] = React.useState(1);

  const [draftRect, setDraftRect] = React.useState<Box | null>(null);
  const [draftStroke, setDraftStroke] = React.useState<StrokeItem | null>(null);
  const [draftFill, setDraftFill] = React.useState<FillItem | null>(null);
  const [inlineEdit, setInlineEdit] = React.useState<{
    dialogueId: number;
    value: string;
    frame: Box;
  } | null>(null);
  const drawStartRef = React.useRef<{ x: number; y: number } | null>(null);

  const {
    selectedDialogueId,
    setSelectedDialogueId,
    tool,
    brushColor,
    brushSize,
    fillColor,
    fillMethod,
    wandTolerance,
    showTranslated,
    showRedraw,
    showBoxes,
    showStrokes,
    showFills,
    stageCommand,
    stageCommandSeq,
    setZoom,
    pushHistory,
  } = useEditorStore();

  // Measure container cleanly
  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = Math.max(200, Math.floor(entry.contentRect.width));
        const h = Math.max(200, Math.floor(entry.contentRect.height));
        setSize({ width: w, height: h });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const resolvedImageUrl = useAuthImageUrl(`/api/pages/${page.id}/image`);

  // Load page image
  React.useEffect(() => {
    if (!resolvedImageUrl) {
      setImageEl(null);
      return;
    }
    let active = true;
    const img = new window.Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      if (!active) return;
      setImageEl(img);
    };
    img.src = resolvedImageUrl;
    return () => {
      active = false;
    };
  }, [resolvedImageUrl]);

  const imgW = page.width || imageEl?.naturalWidth || 800;
  const imgH = page.height || imageEl?.naturalHeight || 1100;

  const fitToContainer = React.useCallback(() => {
    const scale = Math.min(size.width / imgW, size.height / imgH) * 0.94;
    const clamped = Math.max(0.05, Math.min(4, scale));
    setStageScale(clamped);
    setStagePos({
      x: (size.width - imgW * clamped) / 2,
      y: (size.height - imgH * clamped) / 2,
    });
    setZoom(clamped);
  }, [imgW, imgH, size.width, size.height, setZoom]);

  // Fit on initial load or page switch
  React.useEffect(() => {
    if (imageEl) fitToContainer();
  }, [page.id, imageEl, fitToContainer]);

  // Handle toolbar zoom commands
  React.useEffect(() => {
    if (!stageCommandSeq) return;
    if (stageCommand === "fit") {
      fitToContainer();
    } else if (stageCommand === "one") {
      const s = 1;
      setStageScale(s);
      setStagePos({
        x: (size.width - imgW) / 2,
        y: (size.height - imgH) / 2,
      });
      setZoom(s);
    }
  }, [stageCommand, stageCommandSeq, fitToContainer, size.width, size.height, imgW, imgH, setZoom]);

  // Attach Transformer to selected dialogue box
  React.useEffect(() => {
    const tr = trRef.current;
    if (!tr) return;
    if (tool === "select" && selectedDialogueId !== null) {
      const node = nodeMapRef.current.get(selectedDialogueId);
      if (node) {
        tr.nodes([node]);
        tr.getLayer()?.batchDraw();
        return;
      }
    }
    tr.nodes([]);
    tr.getLayer()?.batchDraw();
  }, [selectedDialogueId, tool, page.dialogues]);

  const toImageCoords = (stage: Konva.Stage) => {
    const ptr = stage.getPointerPosition();
    if (!ptr) return null;
    return {
      x: (ptr.x - stagePos.x) / stageScale,
      y: (ptr.y - stagePos.y) / stageScale,
    };
  };

  const handleWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const stage = stageRef.current;
    if (!stage) return;
    const oldScale = stageScale;
    const pointer = stage.getPointerPosition();
    if (!pointer) return;

    const scaleBy = 1.08;
    const newScale = e.evt.deltaY < 0 ? oldScale * scaleBy : oldScale / scaleBy;
    const clamped = Math.max(0.08, Math.min(6, newScale));

    const mousePointTo = {
      x: (pointer.x - stagePos.x) / oldScale,
      y: (pointer.y - stagePos.y) / oldScale,
    };

    setStageScale(clamped);
    setStagePos({
      x: pointer.x - mousePointTo.x * clamped,
      y: pointer.y - mousePointTo.y * clamped,
    });
    setZoom(clamped);
  };

  // Magic Wand flood-box sampler in original pixel coordinates
  const runMagicWand = (ix: number, iy: number) => {
    if (!imageEl) return;
    const px = Math.floor(ix);
    const py = Math.floor(iy);
    if (px < 0 || py < 0 || px >= imgW || py >= imgH) return;

    const off = document.createElement("canvas");
    off.width = imgW;
    off.height = imgH;
    const ctx = off.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(imageEl, 0, 0, imgW, imgH);
    const data = ctx.getImageData(0, 0, imgW, imgH).data;

    const idx0 = (py * imgW + px) * 4;
    const r0 = data[idx0];
    const g0 = data[idx0 + 1];
    const b0 = data[idx0 + 2];

    const visited = new Uint8Array(imgW * imgH);
    const stack = [px, py];
    let minX = px;
    let maxX = px;
    let minY = py;
    let maxY = py;
    let count = 0;
    const maxPixels = 160_000;

    while (stack.length > 0 && count < maxPixels) {
      const cy = stack.pop()!;
      const cx = stack.pop()!;
      const pos = cy * imgW + cx;
      if (visited[pos]) continue;
      visited[pos] = 1;

      const idx = pos * 4;
      const diff =
        Math.abs(data[idx] - r0) + Math.abs(data[idx + 1] - g0) + Math.abs(data[idx + 2] - b0);
      if (diff > wandTolerance * 3) continue;

      count++;
      if (cx < minX) minX = cx;
      if (cx > maxX) maxX = cx;
      if (cy < minY) minY = cy;
      if (cy > maxY) maxY = cy;

      if (cx > 0) stack.push(cx - 1, cy);
      if (cx + 1 < imgW) stack.push(cx + 1, cy);
      if (cy > 0) stack.push(cx, cy - 1);
      if (cy + 1 < imgH) stack.push(cx, cy + 1);
    }

    if (count > 16) {
      pushHistory({ strokes: page.strokes, fills: page.fills });
      const newFill: FillItem = {
        id: `wand_${Date.now()}`,
        method: fillMethod,
        color: `rgb(${r0},${g0},${b0})`,
        box: [minX, minY, Math.max(8, maxX - minX + 1), Math.max(8, maxY - minY + 1)],
      };
      onSaveFills([...page.fills, newFill]);
    }
  };

  const handleMouseDown = (e: Konva.KonvaEventObject<MouseEvent | TouchEvent>) => {
    const stage = stageRef.current;
    if (!stage) return;

    if (tool === "select") {
      if (e.target === stage || e.target.attrs.name === "bg-image") {
        setSelectedDialogueId(null);
      }
      return;
    }

    if (tool === "pan") return;

    const pt = toImageCoords(stage);
    if (!pt) return;

    if (tool === "magic_wand") {
      runMagicWand(pt.x, pt.y);
      return;
    }

    if (tool === "add_box" || tool === "translate_region") {
      drawStartRef.current = pt;
      setDraftRect([Math.round(pt.x), Math.round(pt.y), 1, 1]);
      return;
    }

    if (tool === "pen" || tool === "highlighter" || tool === "eraser") {
      setDraftStroke({
        id: `stroke_${Date.now()}`,
        tool,
        color: brushColor,
        size: brushSize,
        opacity: tool === "highlighter" ? 0.35 : 1,
        points: [pt.x, pt.y],
      });
      return;
    }

    if (tool === "fill_brush") {
      setDraftFill({
        id: `fill_${Date.now()}`,
        method: fillMethod,
        color: fillColor,
        size: brushSize * 1.5,
        points: [pt.x, pt.y],
      });
    }
  };

  const handleMouseMove = () => {
    const stage = stageRef.current;
    if (!stage) return;
    const pt = toImageCoords(stage);
    if (!pt) return;

    if ((tool === "add_box" || tool === "translate_region") && drawStartRef.current) {
      const start = drawStartRef.current;
      const x = Math.min(start.x, pt.x);
      const y = Math.min(start.y, pt.y);
      const w = Math.abs(pt.x - start.x);
      const h = Math.abs(pt.y - start.y);
      setDraftRect([Math.round(x), Math.round(y), Math.round(w), Math.round(h)]);
      return;
    }

    if (draftStroke) {
      setDraftStroke((prev) =>
        prev ? { ...prev, points: [...prev.points, pt.x, pt.y] } : null
      );
      return;
    }

    if (draftFill && draftFill.points) {
      setDraftFill((prev) =>
        prev && prev.points ? { ...prev, points: [...prev.points, pt.x, pt.y] } : null
      );
    }
  };

  const handleMouseUp = () => {
    if ((tool === "add_box" || tool === "translate_region") && draftRect) {
      const [x, y, w, h] = draftRect;
      drawStartRef.current = null;
      setDraftRect(null);
      if (w >= 16 && h >= 16) {
        if (tool === "add_box") {
          onCreateBox([x, y, w, h]);
        } else {
          onTranslateRegion([x, y, w, h]);
        }
      }
      return;
    }

    if (draftStroke) {
      pushHistory({ strokes: page.strokes, fills: page.fills });
      const next = [...page.strokes, draftStroke];
      setDraftStroke(null);
      onSaveStrokes(next);
      return;
    }

    if (draftFill) {
      pushHistory({ strokes: page.strokes, fills: page.fills });
      const next = [...page.fills, draftFill];
      setDraftFill(null);
      onSaveFills(next);
    }
  };

  const cursorClass =
    tool === "pan"
      ? "cursor-grab active:cursor-grabbing"
      : tool === "select"
        ? "cursor-default"
        : "cursor-crosshair";

  return (
    <div
      ref={containerRef}
      className={`relative h-full w-full overflow-hidden bg-zinc-200/80 dark:bg-zinc-950 ${cursorClass}`}
    >
      <Stage
        ref={stageRef}
        width={size.width}
        height={size.height}
        x={stagePos.x}
        y={stagePos.y}
        scaleX={stageScale}
        scaleY={stageScale}
        draggable={tool === "pan"}
        onDragEnd={(e) => {
          if (e.target === stageRef.current) {
            setStagePos({ x: e.target.x(), y: e.target.y() });
          }
        }}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onTouchStart={handleMouseDown}
        onMouseMove={handleMouseMove}
        onTouchMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onTouchEnd={handleMouseUp}
      >
        {/* Base Image Layer */}
        <Layer>
          {imageEl && (
            <KonvaImage
              name="bg-image"
              image={imageEl}
              width={imgW}
              height={imgH}
            />
          )}
        </Layer>

        {/* Clean Redraw & Fill Layer */}
        {(showRedraw || showFills) && (
          <Layer>
            {showRedraw &&
              page.dialogues.map((d) => {
                if (d.hide_redraw || d.no_redraw) return null;
                const [x, y, w, h] = d.box_location;
                return (
                  <Rect
                    key={`clean-${d.dialogue_id}`}
                    x={x}
                    y={y}
                    width={w}
                    height={h}
                    cornerRadius={Math.min(w, h) * 0.22}
                    fill="#ffffff"
                    listening={false}
                  />
                );
              })}

            {showFills &&
              [...page.fills, ...(draftFill ? [draftFill] : [])].map((f) => {
                if (f.hidden) return null;
                if (f.box) {
                  return (
                    <Rect
                      key={f.id}
                      x={f.box[0]}
                      y={f.box[1]}
                      width={f.box[2]}
                      height={f.box[3]}
                      fill={f.color || "#ffffff"}
                      listening={false}
                    />
                  );
                }
                if (f.points && f.points.length >= 2) {
                  return (
                    <Line
                      key={f.id}
                      points={f.points}
                      stroke={f.color || "#ffffff"}
                      strokeWidth={f.size || 20}
                      lineCap="round"
                      lineJoin="round"
                      listening={false}
                    />
                  );
                }
                return null;
              })}
          </Layer>
        )}

        {/* Freehand Strokes Layer */}
        {showStrokes && (
          <Layer>
            {[...page.strokes, ...(draftStroke ? [draftStroke] : [])].map((s) => {
              if (s.hidden) return null;
              return (
                <Line
                  key={s.id}
                  points={s.points}
                  stroke={s.tool === "eraser" ? "#ffffff" : s.color}
                  strokeWidth={s.size}
                  opacity={s.tool === "highlighter" ? 0.38 : s.opacity ?? 1}
                  lineCap="round"
                  lineJoin="round"
                  globalCompositeOperation={
                    s.tool === "eraser" ? "destination-out" : "source-over"
                  }
                  listening={false}
                />
              );
            })}
          </Layer>
        )}

        {/* Dialogues, Translated Text & Bounding Boxes Layer */}
        <Layer>
          {page.dialogues.map((d) => {
            const frame = d.text_frame ?? d.box_location;
            const [x, y, w, h] = frame;
            const selected = selectedDialogueId === d.dialogue_id;
            const autoSize = Math.max(11, Math.min(34, Math.floor(h * 0.26)));
            const fontSize = d.overrides?.font_size || autoSize;
            const fontFamily = d.overrides?.font_path || "Plus Jakarta Sans, sans-serif";
            const color = d.overrides?.color || "#141414";
            const fontStyle = d.overrides?.bold ? "bold" : "normal";
            const lineHeight = d.overrides?.line_spacing || 1.12;
            const align = d.overrides?.align || "center";

            return (
              <Group
                key={d.dialogue_id}
                ref={(node) => {
                  if (node) nodeMapRef.current.set(d.dialogue_id, node);
                  else nodeMapRef.current.delete(d.dialogue_id);
                }}
                x={x}
                y={y}
                width={w}
                height={h}
                rotation={d.text_rotation || 0}
                draggable={tool === "select"}
                onClick={() => {
                  if (tool === "select") setSelectedDialogueId(d.dialogue_id);
                }}
                onTap={() => {
                  if (tool === "select") setSelectedDialogueId(d.dialogue_id);
                }}
                onDblClick={() => {
                  setSelectedDialogueId(d.dialogue_id);
                  setInlineEdit({
                    dialogueId: d.dialogue_id,
                    value: d.translated_text ?? "",
                    frame,
                  });
                }}
                onDblTap={() => {
                  setSelectedDialogueId(d.dialogue_id);
                  setInlineEdit({
                    dialogueId: d.dialogue_id,
                    value: d.translated_text ?? "",
                    frame,
                  });
                }}
                onDragEnd={(e) => {
                  const nx = Math.round(e.target.x());
                  const ny = Math.round(e.target.y());
                  onPatchDialogue(d.dialogue_id, {
                    box_location: [nx, ny, w, h],
                    ...(d.text_frame ? { text_frame: [nx, ny, w, h] } : {}),
                  });
                }}
                onTransformEnd={(e) => {
                  const node = e.target;
                  const scaleX = node.scaleX();
                  const scaleY = node.scaleY();
                  node.scaleX(1);
                  node.scaleY(1);
                  const nw = Math.max(24, Math.round(w * scaleX));
                  const nh = Math.max(24, Math.round(h * scaleY));
                  const nx = Math.round(node.x());
                  const ny = Math.round(node.y());
                  const rot = Math.round(node.rotation());
                  onPatchDialogue(d.dialogue_id, {
                    box_location: [nx, ny, nw, nh],
                    text_frame: [nx, ny, nw, nh],
                    text_rotation: rot,
                  });
                }}
              >
                {showTranslated && !d.hide_text && !d.redraw_only && d.translated_text && (
                  <Text
                    x={4}
                    y={4}
                    width={Math.max(12, w - 8)}
                    height={Math.max(12, h - 8)}
                    text={d.translated_text}
                    fontSize={fontSize}
                    fontFamily={fontFamily}
                    fontStyle={fontStyle}
                    fill={color}
                    lineHeight={lineHeight}
                    align={align}
                    verticalAlign="middle"
                    wrap="word"
                  />
                )}

                {showBoxes && (
                  <Rect
                    x={0}
                    y={0}
                    width={w}
                    height={h}
                    cornerRadius={4}
                    stroke={selected ? "#2563eb" : d.edited ? "#d97706" : "#64748b"}
                    strokeWidth={(selected ? 3 : 1.5) / stageScale}
                    fill={selected ? "rgba(37, 99, 235, 0.10)" : "rgba(100, 116, 139, 0.04)"}
                  />
                )}
              </Group>
            );
          })}

          {/* Draft bounding box / region crop */}
          {draftRect && (
            <Rect
              x={draftRect[0]}
              y={draftRect[1]}
              width={draftRect[2]}
              height={draftRect[3]}
              stroke={tool === "translate_region" ? "#059669" : "#2563eb"}
              strokeWidth={2 / stageScale}
              dash={[6 / stageScale, 4 / stageScale]}
              fill={tool === "translate_region" ? "rgba(5, 150, 105, 0.12)" : "rgba(37, 99, 235, 0.12)"}
            />
          )}

          <Transformer
            ref={trRef}
            rotateEnabled={true}
            keepRatio={false}
            boundBoxFunc={(oldBox, newBox) => {
              if (newBox.width < 20 || newBox.height < 20) return oldBox;
              return newBox;
            }}
          />
        </Layer>
      </Stage>

      {/* Inline Double-Click Text Editor Overlay */}
      {inlineEdit && (
        <textarea
          autoFocus
          value={inlineEdit.value ?? ""}
          onChange={(e) =>
            setInlineEdit((prev) => (prev ? { ...prev, value: e.target.value } : null))
          }
          onBlur={() => {
            onPatchDialogue(inlineEdit.dialogueId, { translated_text: inlineEdit.value });
            setInlineEdit(null);
          }}
          onKeyDown={(e) => {
            if (e.key === "Escape") {
              setInlineEdit(null);
            } else if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
              onPatchDialogue(inlineEdit.dialogueId, { translated_text: inlineEdit.value });
              setInlineEdit(null);
            }
          }}
          style={{
            left: `${stagePos.x + inlineEdit.frame[0] * stageScale}px`,
            top: `${stagePos.y + inlineEdit.frame[1] * stageScale}px`,
            width: `${Math.max(120, inlineEdit.frame[2] * stageScale)}px`,
            height: `${Math.max(64, inlineEdit.frame[3] * stageScale)}px`,
          }}
          className="absolute z-20 resize-none rounded-md border-2 border-blue-600 bg-white/95 p-1.5 text-center text-sm font-medium text-zinc-900 shadow-lg outline-none dark:bg-zinc-900/95 dark:text-zinc-100"
        />
      )}

      {/* Bottom-left coordinate & zoom indicator */}
      <div className="pointer-events-none absolute bottom-3 left-3 rounded-lg border border-zinc-200 bg-white/90 px-2.5 py-1 font-mono text-xs text-zinc-700 backdrop-blur-xs dark:border-zinc-800 dark:bg-zinc-900/90 dark:text-zinc-300 tabular-nums">
        {Math.round(stageScale * 100)}% · {imgW}×{imgH}
      </div>
    </div>
  );
}
