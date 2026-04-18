import { v4 as uuidv4 } from 'uuid';
import { rmSync, existsSync } from 'fs';
import { join, resolve, sep } from 'path';
import { dirname } from 'path';
import { fileURLToPath } from 'url';
import db from '../db/database.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const BASE_UPLOADS_DIR = resolve(join(__dirname, '../uploads'));

export const listProjects = (req, res) => {
  const projects = db
    .prepare(
      'SELECT id, name, description, created_at, updated_at FROM projects WHERE user_id = ? ORDER BY updated_at DESC'
    )
    .all(req.user.id);
  res.json({ projects });
};

export const createProject = (req, res) => {
  const { name, description = '' } = req.body;
  if (!name?.trim()) {
    return res.status(400).json({ error: 'Project name is required' });
  }

  const id = uuidv4();
  db.prepare('INSERT INTO projects (id, user_id, name, description) VALUES (?, ?, ?, ?)')
    .run(id, req.user.id, name.trim(), description.trim());

  const project = db
    .prepare('SELECT id, name, description, created_at, updated_at FROM projects WHERE id = ?')
    .get(id);
  res.status(201).json({ project });
};

export const getProject = (req, res) => {
  const project = db
    .prepare(
      'SELECT id, name, description, created_at, updated_at FROM projects WHERE id = ? AND user_id = ?'
    )
    .get(req.params.id, req.user.id);
  if (!project) return res.status(404).json({ error: 'Project not found' });
  res.json({ project });
};

export const updateProject = (req, res) => {
  const { name, description } = req.body;
  const project = db
    .prepare('SELECT id FROM projects WHERE id = ? AND user_id = ?')
    .get(req.params.id, req.user.id);
  if (!project) return res.status(404).json({ error: 'Project not found' });

  if (name !== undefined && name.trim()) {
    db.prepare("UPDATE projects SET name = ?, updated_at = datetime('now') WHERE id = ?")
      .run(name.trim(), req.params.id);
  }
  if (description !== undefined) {
    db.prepare("UPDATE projects SET description = ?, updated_at = datetime('now') WHERE id = ?")
      .run(description.trim(), req.params.id);
  }

  const updated = db
    .prepare('SELECT id, name, description, created_at, updated_at FROM projects WHERE id = ?')
    .get(req.params.id);
  res.json({ project: updated });
};

export const deleteProject = (req, res) => {
  const project = db
    .prepare('SELECT id FROM projects WHERE id = ? AND user_id = ?')
    .get(req.params.id, req.user.id);
  if (!project) return res.status(404).json({ error: 'Project not found' });

  // Delete project-scoped uploads folder safely
  const uploadsDir = resolve(join(BASE_UPLOADS_DIR, req.user.id, req.params.id));
  if (uploadsDir.startsWith(BASE_UPLOADS_DIR + sep) && existsSync(uploadsDir)) {
    try {
      rmSync(uploadsDir, { recursive: true, force: true });
    } catch (err) {
      console.warn('Could not delete project uploads:', err.message);
    }
  }

  db.prepare('DELETE FROM projects WHERE id = ?').run(req.params.id);
  res.json({ message: 'Project deleted' });
};

export const saveProjectState = (req, res) => {
  const project = db
    .prepare('SELECT id FROM projects WHERE id = ? AND user_id = ?')
    .get(req.params.id, req.user.id);
  if (!project) return res.status(404).json({ error: 'Project not found' });

  const state = JSON.stringify({
    ...req.body,
    schemaVersion: 1,
    savedAt: new Date().toISOString(),
  });
  db.prepare("UPDATE projects SET state = ?, updated_at = datetime('now') WHERE id = ?")
    .run(state, req.params.id);
  res.json({ message: 'State saved' });
};

export const loadProjectState = (req, res) => {
  const project = db
    .prepare('SELECT state FROM projects WHERE id = ? AND user_id = ?')
    .get(req.params.id, req.user.id);
  if (!project) return res.status(404).json({ error: 'Project not found' });

  if (!project.state) return res.json({ state: null });

  try {
    res.json({ state: JSON.parse(project.state) });
  } catch {
    res.json({ state: null });
  }
};
