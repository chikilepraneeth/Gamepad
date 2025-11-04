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
const HOST = process.env.HOST || "0.0.0.0";

const app = express();
app.use(cors());
app.get("/", (_, res) => res.send("Dataset collector server running"));

const server = http.createServer(app);
const io = new IOServer(server, { cors: { origin: "*", credentials: true } });

// ---- Data dirs & CSV ----
const dataDir = path.join(__dirname, "data");
fs.mkdirSync(dataDir, { recursive: true });
const csvPath = path.join(dataDir, "dataset.csv");
const csvWriter = createCsvWriter({
  path: csvPath,
  header: [
    { id: "room", title: "room" },
    { id: "userId", title: "userId" },
    { id: "index", title: "index" },
    { id: "label", title: "label" },
    { id: "filename", title: "filename" },
    { id: "timestamp", title: "timestamp" },
    { id: "width", title: "width" },
    { id: "height", title: "height" },
  ],
  append: fs.existsSync(csvPath),
});

// ---- Optional in-memory view of the latest rows ----
const recentRows = [];            // newest last
const MAX_RECENT = 500;

// ---- Rooms: max 10 users ----
const rooms = new Map(); // room -> Set(userIds)

function ensureRoom(room) {
  if (!rooms.has(room)) rooms.set(room, new Set());
  return rooms.get(room);
}
function parseDataURL(dataURL) {
  // data:image/png;base64,AAAA...
  const m = dataURL.match(/^data:(.+);base64,(.*)$/);
  if (!m) throw new Error("Bad data URL");
  return Buffer.from(m[2], "base64");
}

io.on("connection", (socket) => {
  let joinedRoom = null;
  let joinedUser = null;

  // Client -> join room
  socket.on("join", ({ room, userId }, ack) => {
    try {
      if (!room || !userId) return ack?.({ ok:false, error:"Missing room/userId" });
      const set = ensureRoom(room);
      if (set.size >= 10) return ack?.({ ok:false, error:"Room full (max 10)" });

      set.add(userId);
      joinedRoom = room; joinedUser = userId;
      socket.join(room);
      io.to(room).emit("room_update", { room, count: set.size });
      ack?.({ ok:true, count:set.size });
    } catch (e) {
      ack?.({ ok:false, error:String(e) });
    }
  });

  // Client -> sample (PNG)
  socket.on("sample", async (payload, ack) => {
    try {
      if (!joinedRoom) return ack?.({ ok:false, error:"Not joined" });
      const { room, userId, index, label, ts, size, image } = payload || {};
      if (!room || !userId || !index || !label || !image) {
        return ack?.({ ok:false, error:"Missing fields" });
      }

      // Save PNG: data/<room>/<userId>/<index>.png
      const folder = path.join(dataDir, room, userId);
      fs.mkdirSync(folder, { recursive:true });
      const filenameAbs = path.join(folder, `${index}.png`);
      const buf = parseDataURL(image);
      fs.writeFileSync(filenameAbs, buf);

      // Build the CSV row
      const row = {
        room,
        userId,
        index,
        label,
        filename: path.relative(dataDir, filenameAbs).replace(/\\/g,"/"),
        timestamp: ts || Date.now(),
        width: size?.w || 0,
        height: size?.h || 0,
      };

      // Append CSV row
      await csvWriter.writeRecords([row]);

      // Keep in-memory latest rows (and log nicely)
      recentRows.push(row);
      if (recentRows.length > MAX_RECENT) recentRows.shift();
      console.table([row]);               // <— shows the row in your terminal

      // Notify clients in the same room (and globally if you like)
      io.to(room).emit("sample_row", row);
      // io.emit("sample_row_all", row);  // uncomment to broadcast to all rooms

      ack?.({ ok:true });
    } catch (e) {
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

// ---- Simple viewers for your data ----

// JSON API with the last N rows
app.get("/dataset.json", (req, res) => {
  res.json(recentRows);
});

// Tiny HTML table with the last N rows
app.get("/dataset", (req, res) => {
  const rows = recentRows.slice(-MAX_RECENT);
  const escape = (v) => String(v ?? "").replace(/[&<>"']/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[s]));
  const tr = rows.map(r => `
    <tr>
      <td>${escape(r.room)}</td>
      <td>${escape(r.userId)}</td>
      <td>${escape(r.index)}</td>
      <td>${escape(r.label)}</td>
      <td><a href="/data/${escape(r.filename)}" target="_blank">${escape(r.filename)}</a></td>
      <td>${new Date(r.timestamp).toLocaleString()}</td>
      <td>${escape(r.width)}</td>
      <td>${escape(r.height)}</td>
    </tr>`).join("");

  res.send(`<!doctype html>
  <meta charset="utf-8">
  <title>Dataset (latest ${rows.length})</title>
  <style>
    body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;padding:16px}
    table{border-collapse:collapse;width:100%}
    td,th{border:1px solid #ccc;padding:6px 8px;font-size:14px}
    th{background:#f6f6f6;text-align:left}
    code{background:#f0f0f0;padding:2px 4px;border-radius:4px}
  </style>
  <h2>Dataset (latest ${rows.length})</h2>
  <p>JSON: <code><a href="/dataset.json" target="_blank">/dataset.json</a></code></p>
  <table>
    <thead>
      <tr>
        <th>room</th><th>userId</th><th>index</th><th>label</th>
        <th>filename</th><th>timestamp</th><th>width</th><th>height</th>
      </tr>
    </thead>
    <tbody>${tr || '<tr><td colspan="8" style="text-align:center;color:#888">No rows yet</td></tr>'}</tbody>
  </table>`);
});

// Serve saved images under /data/*
app.use("/data", express.static(dataDir));

server.listen(PORT, HOST, () => {
  console.log(`✅ Server running on http://${HOST}:${PORT}`);
  console.log(`📁 Data folder: ${dataDir}`);
  console.log(`🧾 CSV: ${csvPath}`);
  console.log(`👀 View rows at:   http://${HOST}:${PORT}/dataset`);
  console.log(`🧩 JSON endpoint: http://${HOST}:${PORT}/dataset.json`);
});
