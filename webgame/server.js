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
const ROOM_CAP = Number(process.env.ROOM_CAP || 100); // <-- cap now 100 by default

const app = express();

// CORS
app.use(cors({ origin: "*", credentials: true }));

// Optional static for local testing
const publicDir = path.join(__dirname, "public");
if (fs.existsSync(publicDir)) app.use(express.static(publicDir));

const server = http.createServer(app);
const io = new IOServer(server, {
  cors: { origin: "*", credentials: true },
  path: "/socket.io",
});

// Data dirs & CSV
const dataDir = path.join(__dirname, "data");
fs.mkdirSync(dataDir, { recursive: true });
const csvPath = path.join(dataDir, "dataset.csv");

const csvHeader = [
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
];

const csvWriter = createCsvWriter({
  path: csvPath,
  header: csvHeader,
  append: fs.existsSync(csvPath),
});

// Serve PNGs
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

// 🔥 Delete one row (by room + userId + index)
app.delete("/dataset", (req, res) => {
  try {
    const { room, userId, index } = req.query || {};
    if (!room || !userId || !index) {
      return res.status(400).json({ ok:false, error:"Missing room/userId/index" });
    }
    if (!fs.existsSync(csvPath)) return res.status(404).json({ ok:false, error:"No dataset" });

    const raw = fs.readFileSync(csvPath, "utf8");
    const rows = raw.trim()
      ? csvParse(raw, { columns: true, skip_empty_lines: true })
      : [];

    const i = rows.findIndex(r => r.room === room && r.userId === userId && String(r.index) === String(index));
    if (i === -1) return res.status(404).json({ ok:false, error:"Row not found" });

    // Delete PNG if present
    const rel = rows[i].filename;
    if (rel) {
      const abs = path.join(dataDir, rel);
      if (fs.existsSync(abs)) {
        try { fs.unlinkSync(abs); } catch {}
      }
    }

    // Rewrite CSV without that row
    rows.splice(i, 1);
    const writer = createCsvWriter({ path: csvPath, header: csvHeader, append: false });
    writer.writeRecords(rows).then(() => {
      return res.json({ ok:true });
    }).catch(e => res.status(500).json({ ok:false, error:String(e) }));
  } catch (e) {
    res.status(500).json({ ok:false, error:String(e) });
  }
});

// Dataset HTML
app.get("/dataset", (req, res) => {
  try {
    const rows = fs.existsSync(csvPath)
      ? csvParse(fs.readFileSync(csvPath, "utf8"), { columns: true, skip_empty_lines: true })
      : [];

    const rowsHtml = rows.map(r => {
      const imgHref = `/data/${r.filename}`;
      const thumb = r.filename
        ? `<a href="${imgHref}" target="_blank"><img src="${imgHref}" style="height:64px;border:1px solid #ccc;border-radius:6px"></a>`
        : `<span style="opacity:.7">n/a</span>`;
      const matShort = (r.matrix && r.matrix.length > 120) ? (r.matrix.slice(0,120) + "…") : (r.matrix||"");
      const delAttrs = `data-room="${r.room}" data-user="${r.userId}" data-index="${r.index}"`;
      return `
        <tr id="row-${r.room}-${r.userId}-${r.index}">
          <td>${r.room}</td>
          <td>${r.userId}</td>
          <td>${r.index}</td>
          <td>${r.label}</td>
          <td>${thumb}<br><small>${r.filename||""}</small></td>
          <td>${new Date(Number(r.timestamp)||0).toLocaleString()}</td>
          <td>${r.width}×${r.height}</td>
          <td>${r.matrixSize}</td>
          <td style="max-width:420px; word-break:break-word;"><code>${matShort}</code></td>
          <td><button class="btn btn-del" ${delAttrs}>Delete</button></td>
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
  .btn{background:#1b2332;border:1px solid #2b3344;border-radius:8px;padding:8px 12px;color:#e8eefc;text-decoration:none;cursor:pointer}
  .btn:disabled{opacity:.5;cursor:not-allowed}
  .btn-del{background:#2a1f24;border-color:#4b2a35}
  #toast{position:fixed;right:16px;bottom:16px;background:#1b2332;border:1px solid #2b3344;border-radius:10px;padding:10px 12px}
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
      <th>image</th><th>timestamp</th><th>size</th><th>matrixSize</th><th>matrix</th><th>actions</th>
    </tr></thead>
    <tbody id="tbody">${rowsHtml || "<tr><td colspan='10'>No rows yet.</td></tr>"}</tbody>
  </table>
  <div id="toast" style="display:none"></div>
<script>
  async function delRow(btn){
    const room = btn.dataset.room;
    const user = btn.dataset.user;
    const index = btn.dataset.index;
    if (!confirm('Delete this row?')) return;
    btn.disabled = true;
    try{
      const res = await fetch(\`/dataset?room=\${encodeURIComponent(room)}&userId=\${encodeURIComponent(user)}&index=\${encodeURIComponent(index)}\`, {
        method: 'DELETE'
      });
      const j = await res.json().catch(()=> ({}));
      if (res.ok && j.ok){
        const tr = document.getElementById('row-'+room+'-'+user+'-'+index);
        if (tr) tr.remove();
        toast('Deleted ✓');
      }else{
        toast(j.error || 'Delete failed', true);
        btn.disabled = false;
      }
    }catch(e){
      toast('Delete failed', true);
      btn.disabled = false;
    }
  }
  function toast(msg,bad){
    const t=document.getElementById('toast');
    t.textContent = msg;
    t.style.borderColor = bad ? '#ff6b6b' : '#2b3344';
    t.style.display = 'block';
    setTimeout(()=>{ t.style.display='none'; }, 1200);
  }
  document.addEventListener('click', (e)=>{
    if (e.target && e.target.classList.contains('btn-del')){
      delRow(e.target);
    }
  });
</script>
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

// Rooms
const rooms = new Map();
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
      if (!room || !userId) return ack?.({ ok:false, error:"Missing room/userId", cap: ROOM_CAP });
      const set = ensureRoom(room);
      if (set.size >= ROOM_CAP) return ack?.({ ok:false, error:`Room full (max ${ROOM_CAP})`, cap: ROOM_CAP });
      set.add(userId);
      joinedRoom = room; joinedUser = userId;
      socket.join(room);
      io.to(room).emit("room_update", { room, count: set.size, cap: ROOM_CAP });
      ack?.({ ok:true, count:set.size, cap: ROOM_CAP });
    }catch(e){ ack?.({ ok:false, error:String(e), cap: ROOM_CAP }); }
  });

  // optional pad stream (no-op)
  socket.on("pad", (_payload) => {});

  // Sample
  socket.on("sample", async (payload, ack) => {
    try{
      if (!joinedRoom) return ack?.({ ok:false, error:"Not joined" });
      const { room, userId, index, label, ts, size, image, matrix, matrixSize } = payload || {};
      if (!room || !userId || !index || !label || !image || !Array.isArray(matrix)) {
        return ack?.({ ok:false, error:"Missing fields" });
      }

      const folder = path.join(dataDir, room, userId);
      fs.mkdirSync(folder, { recursive:true });
      const filenameAbs = path.join(folder, `${index}.png`);
      fs.writeFileSync(filenameAbs, parseDataURL(image));

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
        io.to(joinedRoom).emit("room_update", { room: joinedRoom, count: set.size, cap: ROOM_CAP });
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
  console.log(`👥 Room cap: ${ROOM_CAP}`);
});
