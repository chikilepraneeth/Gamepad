// server.js
import express from "express";
import http from "http";
import { Server as IOServer } from "socket.io";
import cors from "cors";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { createObjectCsvWriter as createCsvWriter } from "csv-writer";
import { parse as csvParse } from "csv-parse/sync";
import archiver from "archiver";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = process.env.PORT || 3000;
const HOST = process.env.HOST || "0.0.0.0";
const app = express();

// CORS: allow your GitHub Pages origin and others
app.use(cors({ origin: "*", credentials: true }));

// Static: optionally serve a local /public for testing pad.html from backend
const publicDir = path.join(__dirname, "public");
if (fs.existsSync(publicDir)) app.use(express.static(publicDir));

// Socket.IO
const server = http.createServer(app);
const io = new IOServer(server, {
  cors: { origin: "*", credentials: true },
  path: "/socket.io",
});

// Data dirs & CSV
const dataDir = path.join(__dirname, "data");
fs.mkdirSync(dataDir, { recursive: true });
const csvPath = path.join(dataDir, "dataset.csv");

const csvWriter = createCsvWriter({
  path: csvPath,
  header: [
    { id: "room",        title: "room" },
    { id: "userId",      title: "userId" },
    { id: "index",       title: "index" },
    { id: "label",       title: "label" },
    { id: "filename",    title: "filename" },   // relative path to PNG
    { id: "timestamp",   title: "timestamp" },
    { id: "width",       title: "width" },
    { id: "height",      title: "height" },
    { id: "matrixSize",  title: "matrixSize" },
    { id: "matrix",      title: "matrix" },     // JSON string (grayscale matrix)
  ],
  append: fs.existsSync(csvPath),
});

// Expose PNGs from /data
app.use("/data", express.static(dataDir));

// Health
app.get("/", (_, res) => res.send("Dataset collector server running"));

// Dataset JSON
app.get("/dataset.json", (req, res) => {
  try {
    if (!fs.existsSync(csvPath)) return res.json([]);
    const raw = fs.readFileSync(csvPath, "utf8");
    if (!raw.trim()) return res.json([]);
    const rows = csvParse(raw, { columns: true, skip_empty_lines: true });
    res.json(rows);
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

// Dataset HTML table (thumbnail + link)
app.get("/dataset", (req, res) => {
  try {
    const rows = fs.existsSync(csvPath)
      ? csvParse(fs.readFileSync(csvPath, "utf8"), { columns: true, skip_empty_lines: true })
      : [];

    const rowsHtml = rows.map(r => {
      const imgHref = `/data/${r.filename}`;
      const thumb = `<a href="${imgHref}" target="_blank"><img src="${imgHref}" style="height:64px;border:1px solid #ccc;border-radius:6px"></a>`;
      const matShort = (r.matrix && r.matrix.length > 120) ? (r.matrix.slice(0,120) + "…") : (r.matrix||"");
      return `
        <tr>
          <td>${r.room}</td>
          <td>${r.userId}</td>
          <td>${r.index}</td>
          <td>${r.label}</td>
          <td>${thumb}<br><small>${r.filename}</small></td>
          <td>${new Date(Number(r.timestamp)||0).toLocaleString()}</td>
          <td>${r.width}×${r.height}</td>
          <td>${r.matrixSize}</td>
          <td style="max-width:420px; word-break:break-word;"><code>${matShort}</code></td>
        </tr>`;
    }).join("");

    res.setHeader("Content-Type", "text/html; charset=utf-8");
    res.end(`<!doctype html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset</title>
<style>
  body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;background:#0f1115;color:#e8eefc}
  h1{font-weight:600}
  a{color:#4cc3ff}
  table{width:100%;border-collapse:collapse;background:#111723}
  th,td{border:1px solid #2b3344;padding:8px 10px;font-size:13px;vertical-align:top}
  th{position:sticky;top:0;background:#1b2332}
  .bar{display:flex;gap:12px;align-items:center;margin:12px 0}
  .btn{background:#1b2332;border:1px solid #2b3344;border-radius:8px;padding:8px 12px;color:#e8eefc;text-decoration:none}
</style>
</head>
<body>
  <div class="bar">
    <h1 style="margin:0;">Dataset</h1>
    <a class="btn" href="/dataset.json" target="_blank">View JSON</a>
    <a class="btn" href="/download" target="_blank">Download ZIP</a>
  </div>
  <table>
    <thead><tr>
      <th>room</th><th>userId</th><th>index</th><th>label</th>
      <th>image</th><th>timestamp</th><th>size</th><th>matrixSize</th><th>matrix</th>
    </tr></thead>
    <tbody>${rowsHtml || "<tr><td colspan='9'>No rows yet.</td></tr>"}</tbody>
  </table>
</body></html>`);
  } catch (e) {
    res.status(500).send(String(e));
  }
});

// ZIP all data + CSV
app.get("/download", (req,res) => {
  try{
    res.setHeader("Content-Type","application/zip");
    res.setHeader("Content-Disposition","attachment; filename=dataset.zip");
    const archive = archiver("zip", { zlib: { level: 9 }});
    archive.on("error", err => res.status(500).end(String(err)));
    archive.pipe(res);
    if (fs.existsSync(csvPath)) archive.file(csvPath, { name: "dataset.csv" });
    if (fs.existsSync(dataDir)) archive.directory(dataDir, "data");
    archive.finalize();
  }catch(e){
    res.status(500).send(String(e));
  }
});

// Room management (optional)
const rooms = new Map(); // room -> Set(userIds)
function ensureRoom(room){ if(!rooms.has(room)) rooms.set(room,new Set()); return rooms.get(room); }

// Helpers
function parseDataURL(dataURL){
  const m = dataURL.match(/^data:(.+);base64,(.*)$/);
  if (!m) throw new Error("Bad data URL");
  return Buffer.from(m[2], "base64");
}

// Socket events
io.on("connection", (socket) => {
  let joinedRoom = null;
  let joinedUser = null;

  socket.on("join", ({ room, userId }, ack) => {
    try{
      if (!room || !userId) return ack?.({ ok:false, error:"Missing room/userId" });
      const set = ensureRoom(room);
      if (set.size >= 10) return ack?.({ ok:false, error:"Room full (max 10)" });
      set.add(userId);
      joinedRoom = room; joinedUser = userId;
      socket.join(room);
      io.to(room).emit("room_update", { room, count: set.size });
      ack?.({ ok:true, count:set.size });
    }catch(e){ ack?.({ ok:false, error:String(e) }); }
  });

  // Optional pad streaming (ignored server-side, but kept for completeness)
  socket.on("pad", (_payload) => { /* no-op */ });

  // Sample payload (PNG + matrix)
  socket.on("sample", async (payload, ack) => {
    try{
      if (!joinedRoom) return ack?.({ ok:false, error:"Not joined" });
      const { room, userId, index, label, ts, size, image, matrix, matrixSize } = payload || {};
      if (!room || !userId || !index || !label || !image || !Array.isArray(matrix)) {
        return ack?.({ ok:false, error:"Missing fields" });
      }

      // Save PNG
      const folder = path.join(dataDir, room, userId);
      fs.mkdirSync(folder, { recursive:true });
      const filenameAbs = path.join(folder, `${index}.png`);
      fs.writeFileSync(filenameAbs, parseDataURL(image));

      // Append CSV row (matrix JSON string)
      await csvWriter.writeRecords([{
        room,
        userId,
        index,
        label,
        filename: path.relative(dataDir, filenameAbs).replace(/\\/g,"/"),
        timestamp: ts || Date.now(),
        width: size?.w || 0,
        height: size?.h || 0,
        matrixSize: matrixSize || 0,
        matrix: JSON.stringify(matrix),
      }]);

      ack?.({ ok:true });
    }catch(e){
      console.error("sample error:", e);
      ack?.({ ok:false, error:String(e) });
    }
  });

  socket.on("disconnect", () => {
    if (joinedRoom && joinedUser) {
      const set = rooms.get(joinedRoom);
      if (set) {
        set.delete(joinedUser);
        io.to(joinedRoom).emit("room_update", { room: joinedRoom, count: set.size });
      }
    }
  });
});

server.listen(PORT, HOST, () => {
  console.log(`✅ Server running on http://${HOST}:${PORT}`);
  console.log(`📁 Data folder: ${dataDir}`);
  console.log(`🧾 CSV: ${csvPath}`);
  console.log(`👀 View rows at:   http://${HOST}:${PORT}/dataset`);
  console.log(`🧩 JSON endpoint:  http://${HOST}:${PORT}/dataset.json`);
});
