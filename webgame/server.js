// server.js
import express from "express";
import http from "http";
import { Server as IOServer } from "socket.io";
import cors from "cors";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { createObjectCsvWriter as createCsvWriter } from "csv-writer";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = process.env.PORT || 3000;
const HOST = "0.0.0.0";

const app = express();

// CORS (safe to leave; we still serve same-origin)
app.use(cors({
  origin: [
    "https://chikilepraneeth.github.io",
    "*"
  ],
  credentials: true
}));

// Serve static frontend (public/pad.html -> /pad.html)
const publicDir = path.join(__dirname, "public");
app.use(express.static(publicDir));
app.get("/", (_req, res) => res.redirect("/pad.html"));

const server = http.createServer(app);
const io = new IOServer(server, {
  cors: {
    origin: [
      "https://chikilepraneeth.github.io",
      "*"
    ],
    methods: ["GET", "POST"],
    credentials: true
  }
});

// ---- Data dirs & CSV ----
const dataDir = path.join(__dirname, "data");
fs.mkdirSync(dataDir, { recursive: true });

const csvPath = path.join(dataDir, "dataset.csv");
const csvWriter = createCsvWriter({
  path: csvPath,
  header: [
    { id: "room",         title: "room" },
    { id: "userId",       title: "userId" },
    { id: "index",        title: "index" },
    { id: "label",        title: "label" },
    { id: "filename",     title: "filename" },      // original PNG
    { id: "grayFilename", title: "grayFilename" },  // grayscale PNG
    { id: "matrixFile",   title: "matrixFile" },    // JSON 64x64
    { id: "matrixSize",   title: "matrixSize" },    // e.g., 64
    { id: "timestamp",    title: "timestamp" },
    { id: "width",        title: "width" },
    { id: "height",       title: "height" }
  ],
  append: fs.existsSync(csvPath),
});

const recentRows = [];
const MAX_RECENT = 500;

function parseDataURL(dataURL) {
  // data:image/png;base64,AAAA...
  const m = dataURL.match(/^data:(.+);base64,(.*)$/);
  if (!m) throw new Error("Bad data URL");
  return Buffer.from(m[2], "base64");
}

io.on("connection", (socket) => {
  let joinedRoom = null;

  socket.on("join", ({ room, userId }, ack) => {
    if (!room || !userId) return ack?.({ ok:false, error:"Missing room/userId" });
    joinedRoom = room;
    socket.join(room);
    ack?.({ ok:true });
  });

  // payload now includes: imageOriginal, imageGray, matrix, matrixSize
  socket.on("sample", async (payload, ack) => {
    try {
      if (!joinedRoom) return ack?.({ ok:false, error:"Not joined" });

      const {
        room, userId, index, label, ts,
        size, imageOriginal, imageGray,
        matrix, matrixSize
      } = payload || {};

      if (!room || !userId || !index || !label || !imageOriginal || !imageGray || !Array.isArray(matrix)) {
        return ack?.({ ok:false, error:"Missing fields (need original, gray, matrix)" });
      }

      // Folder: data/<room>/<userId>/
      const folder = path.join(dataDir, room, userId);
      fs.mkdirSync(folder, { recursive: true });

      // Save original PNG
      const fileOrigAbs = path.join(folder, `${index}.png`);
      fs.writeFileSync(fileOrigAbs, parseDataURL(imageOriginal));

      // Save grayscale PNG
      const fileGrayAbs = path.join(folder, `${index}_gray.png`);
      fs.writeFileSync(fileGrayAbs, parseDataURL(imageGray));

      // Save matrix JSON
      const fileMatAbs = path.join(folder, `${index}_matrix.json`);
      fs.writeFileSync(fileMatAbs, JSON.stringify({
        size: matrixSize || (matrix?.length ?? 0),
        matrix
      }));

      // Row
      const row = {
        room, userId, index, label,
        filename: path.relative(dataDir, fileOrigAbs).replace(/\\/g,"/"),
        grayFilename: path.relative(dataDir, fileGrayAbs).replace(/\\/g,"/"),
        matrixFile: path.relative(dataDir, fileMatAbs).replace(/\\/g,"/"),
        matrixSize: matrixSize || (matrix?.length ?? 0),
        timestamp: ts || Date.now(),
        width: size?.w || 0,
        height: size?.h || 0
      };

      await csvWriter.writeRecords([row]);
      recentRows.push(row);
      if (recentRows.length > MAX_RECENT) recentRows.shift();

      io.to(room).emit("sample_row", row);
      console.log(`[SAVED] ${row.room} | ${row.userId} #${row.index} "${row.label}" -> ${row.filename}, ${row.grayFilename}, ${row.matrixFile}`);
      ack?.({ ok:true });
    } catch (e) {
      console.error("sample error:", e);
      ack?.({ ok:false, error:String(e) });
    }
  });
});

// Simple viewers
app.get("/dataset.json", (_req, res) => res.json(recentRows));

app.get("/dataset", (_req, res) => {
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const trs = recentRows.map(r => `
    <tr>
      <td>${esc(r.room)}</td>
      <td>${esc(r.userId)}</td>
      <td>${esc(r.index)}</td>
      <td>${esc(r.label)}</td>
      <td><a href="/data/${esc(r.filename)}" target="_blank">${esc(r.filename)}</a></td>
      <td><a href="/data/${esc(r.grayFilename)}" target="_blank">${esc(r.grayFilename)}</a></td>
      <td><a href="/data/${esc(r.matrixFile)}" target="_blank">${esc(r.matrixFile)}</a></td>
      <td>${esc(r.matrixSize)}</td>
      <td>${new Date(r.timestamp).toLocaleString()}</td>
      <td>${esc(r.width)}</td>
      <td>${esc(r.height)}</td>
    </tr>
  `).join("");

  res.send(`<!doctype html>
  <meta charset="utf-8">
  <title>Dataset (${recentRows.length})</title>
  <style>
    body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;padding:16px}
    table{border-collapse:collapse;width:100%}
    td,th{border:1px solid #ccc;padding:6px 8px;font-size:14px}
    th{background:#f6f6f6;text-align:left}
  </style>
  <h2>Dataset (latest ${recentRows.length})</h2>
  <p>JSON: <a href="/dataset.json" target="_blank">/dataset.json</a></p>
  <table>
    <thead><tr>
      <th>room</th><th>userId</th><th>index</th><th>label</th>
      <th>orig PNG</th><th>gray PNG</th><th>matrix JSON</th><th>size</th>
      <th>timestamp</th><th>width</th><th>height</th>
    </tr></thead>
    <tbody>${trs || '<tr><td colspan="11" style="text-align:center;color:#888">No rows yet</td></tr>'}</tbody>
  </table>`);
});

// Serve files
app.use("/data", express.static(dataDir));

server.listen(PORT, HOST, () => {
  console.log(`✅ Server running on http://${HOST}:${PORT}`);
  console.log(`📁 Data folder: ${dataDir}`);
  console.log(`🧾 CSV: ${csvPath}`);
  console.log(`👀 View rows at:   http://${HOST}:${PORT}/dataset`);
  console.log(`🧩 JSON endpoint: http://${HOST}:${PORT}/dataset.json`);
  console.log(`🖼️ Open pad:      http://${HOST}:${PORT}/pad.html?room=test&label=triangle`);
});
