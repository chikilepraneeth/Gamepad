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
app.use(cors());

// ⬇️ Serve /public (so /pad.html works)
const publicDir = path.join(__dirname, "public");
app.use(express.static(publicDir));
app.get("/", (_req, res) => res.redirect("/pad.html"));

const server = http.createServer(app);
const io = new IOServer(server);

// ==== data setup (same as before) ====
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
const recentRows = [];
const MAX_RECENT = 500;

function parseDataURL(dataURL){
  const m = dataURL.match(/^data:(.+);base64,(.*)$/);
  if (!m) throw new Error("Bad data URL");
  return Buffer.from(m[2], "base64");
}

io.on("connection", (socket) => {
  let joinedRoom = null, joinedUser = null;

  socket.on("join", ({ room, userId }, ack) => {
    if (!room || !userId) return ack?.({ ok:false, error:"Missing room/userId" });
    joinedRoom = room; joinedUser = userId;
    socket.join(room);
    ack?.({ ok:true });
  });

  socket.on("sample", async (payload, ack) => {
    try{
      if (!joinedRoom) return ack?.({ ok:false, error:"Not joined" });
      const { room, userId, index, label, ts, size, image } = payload || {};
      if (!room || !userId || !index || !label || !image)
        return ack?.({ ok:false, error:"Missing fields" });

      const folder = path.join(dataDir, room, userId);
      fs.mkdirSync(folder, { recursive:true });
      const filenameAbs = path.join(folder, `${index}.png`);
      fs.writeFileSync(filenameAbs, parseDataURL(image));

      const row = {
        room, userId, index, label,
        filename: path.relative(dataDir, filenameAbs).replace(/\\/g,"/"),
        timestamp: ts || Date.now(),
        width: size?.w || 0, height: size?.h || 0,
      };
      await csvWriter.writeRecords([row]);
      recentRows.push(row); if (recentRows.length > MAX_RECENT) recentRows.shift();
      io.to(room).emit("sample_row", row);
      ack?.({ ok:true });
    }catch(e){
      console.error("sample error:", e);
      ack?.({ ok:false, error:String(e) });
    }
  });
});

// viewers & static data
app.get("/dataset.json", (_req,res)=>res.json(recentRows));
app.get("/dataset", (_req,res)=>{
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const trs = recentRows.map(r=>`<tr>
    <td>${esc(r.room)}</td><td>${esc(r.userId)}</td><td>${esc(r.index)}</td><td>${esc(r.label)}</td>
    <td><a href="/data/${esc(r.filename)}" target="_blank">${esc(r.filename)}</a></td>
    <td>${new Date(r.timestamp).toLocaleString()}</td><td>${esc(r.width)}</td><td>${esc(r.height)}</td>
  </tr>`).join("");
  res.send(`<!doctype html><meta charset="utf-8"><title>Dataset</title>
  <style>body{font-family:system-ui;padding:16px}table{border-collapse:collapse;width:100%}
  td,th{border:1px solid #ccc;padding:6px 8px}th{background:#f6f6f6;text-align:left}</style>
  <h2>Dataset (${recentRows.length})</h2><p><a href="/dataset.json" target="_blank">/dataset.json</a></p>
  <table><thead><tr><th>room</th><th>userId</th><th>index</th><th>label</th>
  <th>filename</th><th>timestamp</th><th>width</th><th>height</th></tr></thead>
  <tbody>${trs || '<tr><td colspan="8" style="text-align:center;color:#888">No rows yet</td></tr>'}</tbody></table>`);
});
app.use("/data", express.static(dataDir));

server.listen(PORT, HOST, () => {
  console.log(`✅ Server running on http://${HOST}:${PORT}`);
});
