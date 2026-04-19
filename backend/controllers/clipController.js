import { createReadStream, existsSync, mkdirSync, writeFileSync, unlinkSync } from 'fs';
import { join, resolve } from 'path';
import { v4 as uuidv4 } from 'uuid';
import db from '../db/database.js';
import { BASE_UPLOADS_DIR } from '../middleware/projectOwnership.js';

const SAFE_KEY_RE = /^[a-z0-9_-]{1,80}$/i;

function verifyOwnership(projectId, userId) {
  return db.prepare('SELECT id FROM projects WHERE id = ? AND user_id = ?').get(projectId, userId);
}

export const saveClip = (req, res) => {
  const { id: projectId } = req.params;
  const { instrumentKey } = req.body;

  if (!instrumentKey?.trim() || !SAFE_KEY_RE.test(instrumentKey.trim())) {
    return res.status(400).json({ error: 'Invalid instrumentKey' });
  }
  if (!req.file) {
    return res.status(400).json({ error: 'video file is required' });
  }
  if (!verifyOwnership(projectId, req.user.id)) {
    return res.status(404).json({ error: 'Project not found' });
  }

  const key = instrumentKey.trim();

  // Delete old clip file if it exists
  const existing = db
    .prepare('SELECT file_path FROM project_clips WHERE project_id = ? AND instrument_key = ?')
    .get(projectId, key);
  if (existing?.file_path && existsSync(existing.file_path)) {
    try { unlinkSync(existing.file_path); } catch { /* ignore */ }
  }

  // Save new clip file inside project-scoped uploads dir
  const uploadsDir = join(BASE_UPLOADS_DIR, req.user.id, projectId);
  mkdirSync(uploadsDir, { recursive: true });
  const filename = `clip_${key}_${uuidv4()}.mp4`;
  const filePath = join(uploadsDir, filename);

  if (!resolve(filePath).startsWith(resolve(BASE_UPLOADS_DIR))) {
    return res.status(400).json({ error: 'Invalid file path' });
  }

  writeFileSync(filePath, req.file.buffer);

  db.prepare(`
    INSERT INTO project_clips (project_id, instrument_key, file_path)
    VALUES (?, ?, ?)
    ON CONFLICT(project_id, instrument_key) DO UPDATE
      SET file_path = excluded.file_path, created_at = datetime('now')
  `).run(projectId, key, filePath);

  res.json({ ok: true, instrumentKey: key });
};

export const listClips = (req, res) => {
  const { id: projectId } = req.params;
  if (!verifyOwnership(projectId, req.user.id)) {
    return res.status(404).json({ error: 'Project not found' });
  }
  const clips = db
    .prepare('SELECT instrument_key FROM project_clips WHERE project_id = ? ORDER BY created_at')
    .all(projectId);
  res.json({ clips });
};

export const getClipFile = (req, res) => {
  const { id: projectId, instrumentKey } = req.params;
  if (!verifyOwnership(projectId, req.user.id)) {
    return res.status(404).json({ error: 'Project not found' });
  }

  const clip = db
    .prepare('SELECT file_path FROM project_clips WHERE project_id = ? AND instrument_key = ?')
    .get(projectId, instrumentKey);
  if (!clip) return res.status(404).json({ error: 'Clip not found' });
  if (!existsSync(clip.file_path)) return res.status(404).json({ error: 'Clip file missing' });

  if (!resolve(clip.file_path).startsWith(resolve(BASE_UPLOADS_DIR))) {
    return res.status(403).json({ error: 'Forbidden' });
  }

  res.setHeader('Content-Type', 'video/mp4');
  res.setHeader('Cache-Control', 'no-store');
  createReadStream(clip.file_path).pipe(res);
};

export const deleteClip = (req, res) => {
  const { id: projectId, instrumentKey } = req.params;
  if (!verifyOwnership(projectId, req.user.id)) {
    return res.status(404).json({ error: 'Project not found' });
  }

  const clip = db
    .prepare('SELECT file_path FROM project_clips WHERE project_id = ? AND instrument_key = ?')
    .get(projectId, instrumentKey);
  if (clip?.file_path && existsSync(clip.file_path)) {
    try { unlinkSync(clip.file_path); } catch { /* ignore */ }
  }
  db.prepare('DELETE FROM project_clips WHERE project_id = ? AND instrument_key = ?')
    .run(projectId, instrumentKey);

  res.json({ ok: true });
};
