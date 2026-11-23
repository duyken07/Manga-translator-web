import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { GoogleGenAI } from "@google/genai";
import cors from "cors";
import express, { type NextFunction, type Request, type Response } from "express";
import { WebSocket, WebSocketServer } from "ws";
import { DEFAULT_GEMINI_MODEL, getServerConfig } from "./config.js";
import { dialoguesRouter } from "./routes/dialogues.js";
import { pagesRouter } from "./routes/pages.js";
import { projectsRouter } from "./routes/projects.js";
import { modelsRouter, settingsRouter } from "./routes/settings.js";
import { jobsRouter, redrawRouter, translationRouter } from "./routes/translation.js";
import {
  checkToken,
  DomainError,
  eventHub,
  getEffectiveApiKey,
  isAuthRequired,
  jobManager,
  projectsMap,
  requireToken,
} from "./store.js";

const cfg = getServerConfig();
const app = express();

// Restrict CORS when origins are explicitly configured; otherwise allow same-origin
app.use(
  cors({
    origin: cfg.corsOrigins.length > 0 ? cfg.corsOrigins : false,
    credentials: false,
    methods: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization", "X-API-Token"],
  })
);

// Public health & runtime inspection endpoints
app.get("/api/health", async (_req, res) => {
  if (cfg.fastapiBackendUrl) {
    try {
      const upstream = await fetch(`${cfg.fastapiBackendUrl}/api/health`);
      const ct = (upstream.headers.get("content-type") || "").toLowerCase();
      const text = await upstream.text();
      if (ct.includes("text/html") || text.trimStart().toLowerCase().startsWith("<!doctype") || text.trimStart().toLowerCase().startsWith("<html")) {
        res.status(502).json({
          status: "degraded",
          proxy: true,
          detail: `FASTAPI_BACKEND_URL returned an HTML page (HTTP ${upstream.status}) instead of FastAPI JSON. Ensure it points directly to a running Public *.hf.space or Cloud Run FastAPI backend.`,
        });
        return;
      }
      const data = JSON.parse(text) as Record<string, unknown>;
      res.status(upstream.status).json({ ...data, proxy: true });
      return;
    } catch (err) {
      res.status(502).json({ status: "degraded", proxy: true, detail: err instanceof Error ? err.message : "Backend unreachable" });
      return;
    }
  }
  res.json({ status: "ok", proxy: false });
});

app.get("/api/runtime", (_req, res) => {
  const currentCfg = getServerConfig();
  const hasRawBackend = Boolean((process.env.FASTAPI_BACKEND_URL || "").trim());
  res.json({
    mode: currentCfg.fastapiBackendUrl ? "fastapi_proxy" : "node_standalone",
    fastapi_configured: Boolean(currentCfg.fastapiBackendUrl),
    fastapi_url_invalid: hasRawBackend && !currentCfg.fastapiBackendUrl && !currentCfg.fastapiWrapperUrlDetected,
    fastapi_hf_wrapper_url: currentCfg.fastapiWrapperUrlDetected,
    auth_required: isAuthRequired(),
    has_gemini_key: Boolean(getEffectiveApiKey()),
  });
});

// ---------------------------------------------------------------------------
// FastAPI Proxy Mode (when FASTAPI_BACKEND_URL is configured for Cloud Run)
// ---------------------------------------------------------------------------
if (cfg.fastapiBackendUrl) {
  app.use("/api", requireToken, async (req, res) => {
    try {
      const targetUrl = `${cfg.fastapiBackendUrl}/api${req.url}`;
      const headers: Record<string, string> = {};
      if (req.headers["content-type"]) headers["content-type"] = String(req.headers["content-type"]);
      if (req.headers["authorization"]) {
        headers["authorization"] = String(req.headers["authorization"]);
      } else if (cfg.apiToken) {
        headers["authorization"] = `Bearer ${cfg.apiToken}`;
      }

      const hasBody = !["GET", "HEAD"].includes(req.method.toUpperCase());
      const chunks: Buffer[] = [];
      if (hasBody) {
        for await (const chunk of req) {
          chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
        }
      }

      const upstream = await fetch(targetUrl, {
        method: req.method,
        headers,
        body: hasBody && chunks.length > 0 ? Buffer.concat(chunks) : undefined,
      });

      const buf = Buffer.from(await upstream.arrayBuffer());
      const upstreamType = (upstream.headers.get("content-type") || "").toLowerCase();
      const head = buf.subarray(0, 64).toString("utf8").trimStart().toLowerCase();
      if (upstreamType.includes("text/html") || head.startsWith("<!doctype") || head.startsWith("<html")) {
        res.status(502).json({
          detail:
            `Upstream FASTAPI_BACKEND_URL returned an HTML page (HTTP ${upstream.status}) instead of API JSON. Verify it points to a Running, Public *.hf.space or Cloud Run FastAPI service (not a web page or private Space).`,
        });
        return;
      }

      res.status(upstream.status);
      upstream.headers.forEach((val, key) => {
        if (!["transfer-encoding", "connection", "content-encoding"].includes(key.toLowerCase())) {
          res.setHeader(key, val);
        }
      });
      res.send(buf);
    } catch (err) {
      res.status(502).json({
        detail: `FastAPI backend proxy error: ${err instanceof Error ? err.message : String(err)}`,
      });
    }
  });
} else {
  // Standalone Node.js Backend Mode
  app.use("/api", express.json({ limit: "15mb" }));

  // Character / Translation Assistant Chat
  app.post("/api/projects/:project_id/chat", requireToken, async (req, res) => {
    const project = projectsMap.get(String(req.params.project_id));
    if (!project) {
      res.status(404).json({ detail: "Project not found" });
      return;
    }
    const message = String(req.body?.message ?? "").trim();
    const character = String(req.body?.character ?? "assistant");
    if (!message) {
      res.status(422).json({ detail: "message is required" });
      return;
    }
    const apiKey = getEffectiveApiKey();
    if (!apiKey) {
      res.status(400).json({ detail: "GEMINI_API_KEY is not configured in server secrets" });
      return;
    }
    const persona =
      character === "sengoku_yuno"
        ? "You are Sengoku Yuno, a sharp, helpful manga translation assistant who explains Japanese nuances, slang, honorifics, and SFX naturally."
        : "You are a precise manga translation and localization assistant helping the translator with terminology, grammar, and phrasing.";
    const glossary = Object.entries(project.glossary)
      .slice(0, 30)
      .map(([k, v]) => `${k} = ${v}`)
      .join(", ");

    try {
      const ai = new GoogleGenAI({
        apiKey,
        httpOptions: { headers: { "User-Agent": "aistudio-build" } },
      });
      const response = await ai.models.generateContent({
        model: DEFAULT_GEMINI_MODEL,
        contents: message,
        config: {
          systemInstruction: `${persona}\nProject: ${project.name} (${project.source_lang} -> ${project.target_lang}).${glossary ? `\nGlossary: ${glossary}` : ""}`,
        },
      });
      res.json({ reply: response.text ?? "" });
    } catch (err) {
      res.status(502).json({ detail: `Chat failed: ${err instanceof Error ? err.message : String(err)}` });
    }
  });

  app.use("/api/projects", requireToken, projectsRouter);
  app.use("/api", requireToken, pagesRouter);
  app.use("/api/pages/:page_id/dialogues", requireToken, dialoguesRouter);
  app.use("/api", requireToken, translationRouter);
  app.use("/api", requireToken, jobsRouter);
  app.use("/api/pages", requireToken, redrawRouter);
  app.use("/api/models", requireToken, modelsRouter);
  app.use("/api/settings", requireToken, settingsRouter);
}

app.use((err: unknown, _req: Request, res: Response, _next: NextFunction) => {
  if (err instanceof DomainError) {
    res.status(err.status).json({ detail: err.message });
    return;
  }
  const status = typeof (err as { status?: number })?.status === "number" ? (err as { status: number }).status : 500;
  const message = err instanceof Error ? err.message : "Internal server error";
  res.status(status).json({ detail: message });
});

async function start() {
  const server = http.createServer(app);
  const wss = new WebSocketServer({ noServer: true });

  // Attach Vite middleware in dev, or serve dist/client in production
  const isProd = process.env.NODE_ENV === "production";
  if (!isProd) {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true, hmr: false },
      appType: "spa",
    });
    app.use(vite.middlewares);

    server.on("upgrade", (req, socket, head) => {
      const hostHeader = req.headers.host || "localhost";
      const url = new URL(req.url || "/", `http://${hostHeader}`);
      if (url.pathname.startsWith("/ws/")) {
        handleWsUpgrade(req, socket, head, url, wss);
      } else {
        socket.destroy();
      }
    });
  } else {
    const clientDir = path.resolve(import.meta.dirname, "../client");
    if (fs.existsSync(clientDir)) {
      app.use(express.static(clientDir));
      app.get("*splat", (_req, res) => {
        res.sendFile(path.join(clientDir, "index.html"));
      });
    }
    server.on("upgrade", (req, socket, head) => {
      const hostHeader = req.headers.host || "localhost";
      const url = new URL(req.url || "/", `http://${hostHeader}`);
      if (url.pathname.startsWith("/ws/")) {
        handleWsUpgrade(req, socket, head, url, wss);
      } else {
        socket.destroy();
      }
    });
  }

  server.listen(cfg.port, cfg.host, () => {
    console.log(
      `Manga Translator Web listening on http://${cfg.host}:${cfg.port} (${
        cfg.fastapiBackendUrl ? `proxy -> ${cfg.fastapiBackendUrl}` : "standalone Node + Gemini"
      })`
    );
  });
}

function handleWsUpgrade(
  req: http.IncomingMessage,
  socket: import("node:stream").Duplex,
  head: Buffer,
  url: URL,
  wss: WebSocketServer
) {
  const match = url.pathname.match(/^\/ws\/projects\/([^/]+)\/progress$/);
  if (!match) {
    socket.destroy();
    return;
  }
  const projectId = decodeURIComponent(match[1]);
  const token = url.searchParams.get("token") ?? undefined;
  const since = Number(url.searchParams.get("since") || "0") || 0;

  wss.handleUpgrade(req, socket, head, (clientWs) => {
    if (isAuthRequired() && !checkToken(token)) {
      clientWs.close(4401);
      return;
    }

    if (cfg.fastapiBackendUrl) {
      const upstreamBase = cfg.fastapiBackendUrl.replace(/^http/, "ws");
      const upstreamToken = token || cfg.apiToken || "";
      const upstreamWs = new WebSocket(
        `${upstreamBase}/ws/projects/${encodeURIComponent(projectId)}/progress?token=${encodeURIComponent(upstreamToken)}&since=${since}`
      );
      upstreamWs.on("message", (data, isBinary) => {
        if (clientWs.readyState === clientWs.OPEN) clientWs.send(data, { binary: isBinary });
      });
      upstreamWs.on("close", (code) => clientWs.close(code));
      upstreamWs.on("error", () => clientWs.close(1011));
      clientWs.on("close", () => upstreamWs.close());
      return;
    }

    if (!projectsMap.has(projectId)) {
      clientWs.close(4404);
      return;
    }

    eventHub.subscribe(projectId, clientWs, since);
    clientWs.send(JSON.stringify({ type: "hello", jobs: jobManager.list(projectId, 10) }));

    const heartbeat = setInterval(() => {
      if (clientWs.readyState === clientWs.OPEN) {
        clientWs.send(JSON.stringify({ type: "ping" }));
      }
    }, 25_000);

    clientWs.on("close", () => {
      clearInterval(heartbeat);
      eventHub.unsubscribe(projectId, clientWs);
    });
  });
}

void start();
