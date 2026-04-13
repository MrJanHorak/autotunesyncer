import { spawn } from 'child_process';
import { writeFileSync, existsSync, mkdirSync } from 'fs';
import { join, resolve } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { v4 as uuidv4 } from 'uuid';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const UPLOADS_DIR = resolve(__dirname, '../uploads');
const PYTHON_DIR = resolve(__dirname, '../python');

// In-process queue for pre-cache jobs.
// Concurrency is capped at 1 so only one Python process writes to
// cache_index.json at a time, preventing cross-process corruption.
const precacheQueue = [];
let isProcessing = false;

const processQueue = () => {
  if (isProcessing || precacheQueue.length === 0) return;
  isProcessing = true;

  const { filePath, midiNotes } = precacheQueue.shift();

  const payload = JSON.stringify({ video_path: filePath, midi_notes: midiNotes });
  const scriptPath = join(PYTHON_DIR, 'precache_cli.py');

  const child = spawn('python', [scriptPath], { stdio: ['pipe', 'pipe', 'pipe'] });

  child.stdin.write(payload);
  child.stdin.end();

  child.stdout.on('data', (d) => console.log(`[precache] ${d.toString().trim()}`));
  child.stderr.on('data', (d) => console.error(`[precache] ${d.toString().trim()}`));

  child.on('close', (code) => {
    if (code !== 0) {
      console.error(`[precache] Python exited with code ${code} for ${filePath}`);
    }
    isProcessing = false;
    processQueue(); // process next job if any
  });
};

/**
 * POST /api/autotune/precache
 * Accepts a multipart video upload + JSON body field `midiNotes`.
 * Saves the video to the uploads directory and enqueues a background
 * pre-cache job. Returns 202 immediately.
 */
export const handlePrecache = (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No video file provided' });
    }

    const midiNotesRaw = req.body?.midiNotes;
    if (!midiNotesRaw) {
      return res.status(400).json({ error: 'midiNotes field is required' });
    }

    let midiNotes;
    try {
      midiNotes = JSON.parse(midiNotesRaw);
      if (!Array.isArray(midiNotes) || midiNotes.length === 0) throw new Error();
    } catch {
      return res.status(400).json({ error: 'midiNotes must be a non-empty JSON array' });
    }

    // Save the uploaded blob to a deterministic temp path in uploads dir.
    if (!existsSync(UPLOADS_DIR)) mkdirSync(UPLOADS_DIR, { recursive: true });
    const filePath = join(UPLOADS_DIR, `precache_${uuidv4()}.mp4`);
    writeFileSync(filePath, req.file.buffer);

    // Validate the saved path stays within uploads dir (security)
    const resolved = resolve(filePath);
    if (!resolved.startsWith(UPLOADS_DIR)) {
      return res.status(400).json({ error: 'Invalid file path' });
    }

    precacheQueue.push({ filePath: resolved, midiNotes });
    console.log(`[precache] Queued: ${filePath} (${midiNotes.length} notes, queue depth: ${precacheQueue.length})`);
    processQueue();

    return res.status(202).json({ message: 'Pre-cache job queued', notes: midiNotes.length });
  } catch (err) {
    console.error('[precache] Error:', err);
    return res.status(500).json({ error: err.message });
  }
};
