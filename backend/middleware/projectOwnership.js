import { join, resolve, sep } from 'path';
import { dirname } from 'path';
import { fileURLToPath } from 'url';
import { mkdirSync } from 'fs';
import db from '../db/database.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
export const BASE_UPLOADS_DIR = resolve(join(__dirname, '../uploads'));

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * Middleware: verify the requesting user owns the project indicated by
 * req.params.projectId, req.query.projectId, or req.body.projectId.
 * Attaches req.project (DB row + .uploadsDir).
 * Must run AFTER authenticateToken.
 */
export const requireProjectOwnership = (req, res, next) => {
  const projectId =
    req.params.projectId ?? req.query.projectId ?? req.body?.projectId;

  if (!projectId) {
    return res.status(400).json({ error: 'projectId is required' });
  }

  if (!UUID_RE.test(projectId)) {
    return res.status(400).json({ error: 'Invalid projectId format' });
  }

  const project = db
    .prepare('SELECT * FROM projects WHERE id = ? AND user_id = ?')
    .get(projectId, req.user.id);

  if (!project) {
    return res.status(404).json({ error: 'Project not found' });
  }

  // Compute and validate project-scoped uploads dir
  const uploadsDir = resolve(join(BASE_UPLOADS_DIR, req.user.id, projectId));
  if (!uploadsDir.startsWith(BASE_UPLOADS_DIR + sep) &&
      uploadsDir !== BASE_UPLOADS_DIR) {
    return res.status(400).json({ error: 'Invalid project path' });
  }

  mkdirSync(uploadsDir, { recursive: true });
  req.project = { ...project, uploadsDir };
  next();
};
