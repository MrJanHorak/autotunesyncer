import {
  createReadStream,
  existsSync,
  mkdirSync,
  writeFileSync,
  unlinkSync,
} from 'fs';
import { extname, join, resolve } from 'path';
import { v4 as uuidv4 } from 'uuid';
import db from '../db/database.js';
import { BASE_UPLOADS_DIR } from '../middleware/projectOwnership.js';

const DEFAULT_EXTENSION_BY_KIND = {
  image: '.png',
  video: '.mp4',
};

function verifyOwnership(projectId, userId) {
  return db
    .prepare('SELECT id FROM projects WHERE id = ? AND user_id = ?')
    .get(projectId, userId);
}

function getExistingBackground(projectId) {
  return db
    .prepare(
      'SELECT file_path, mime_type, media_kind, original_name FROM project_backgrounds WHERE project_id = ?',
    )
    .get(projectId);
}

function deleteExistingBackgroundFile(projectId) {
  const existing = getExistingBackground(projectId);
  if (existing?.file_path && existsSync(existing.file_path)) {
    try {
      unlinkSync(existing.file_path);
    } catch {
      /* ignore */
    }
  }
}

function getMediaKind(mimeType = '') {
  if (mimeType.startsWith('image/')) return 'image';
  if (mimeType.startsWith('video/')) return 'video';
  return null;
}

export const saveBackground = (req, res) => {
  const { id: projectId } = req.params;

  if (!req.file) {
    return res.status(400).json({ error: 'background file is required' });
  }
  if (!verifyOwnership(projectId, req.user.id)) {
    return res.status(404).json({ error: 'Project not found' });
  }

  const mediaKind = getMediaKind(req.file.mimetype || '');
  if (!mediaKind) {
    return res
      .status(400)
      .json({ error: 'Background must be an image or video file' });
  }

  deleteExistingBackgroundFile(projectId);

  const uploadsDir = join(BASE_UPLOADS_DIR, req.user.id, projectId);
  mkdirSync(uploadsDir, { recursive: true });

  const fileExtension =
    extname(req.file.originalname || '') ||
    DEFAULT_EXTENSION_BY_KIND[mediaKind];
  const filePath = join(
    uploadsDir,
    `background_${uuidv4()}${fileExtension.toLowerCase()}`,
  );

  if (!resolve(filePath).startsWith(resolve(BASE_UPLOADS_DIR))) {
    return res.status(400).json({ error: 'Invalid file path' });
  }

  writeFileSync(filePath, req.file.buffer);

  db.prepare(
    `
      INSERT INTO project_backgrounds (project_id, file_path, mime_type, media_kind, original_name)
      VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(project_id) DO UPDATE
        SET file_path = excluded.file_path,
            mime_type = excluded.mime_type,
            media_kind = excluded.media_kind,
            original_name = excluded.original_name,
            created_at = datetime('now')
    `,
  ).run(
    projectId,
    filePath,
    req.file.mimetype,
    mediaKind,
    req.file.originalname || '',
  );

  res.json({
    ok: true,
    background: {
      kind: mediaKind,
      mimeType: req.file.mimetype,
      originalName: req.file.originalname || '',
    },
  });
};

export const getBackgroundFile = (req, res) => {
  const { id: projectId } = req.params;
  if (!verifyOwnership(projectId, req.user.id)) {
    return res.status(404).json({ error: 'Project not found' });
  }

  const background = getExistingBackground(projectId);
  if (!background) {
    return res.status(404).json({ error: 'Background not found' });
  }
  if (!existsSync(background.file_path)) {
    return res.status(404).json({ error: 'Background file missing' });
  }
  if (!resolve(background.file_path).startsWith(resolve(BASE_UPLOADS_DIR))) {
    return res.status(403).json({ error: 'Forbidden' });
  }

  res.setHeader(
    'Content-Type',
    background.mime_type || 'application/octet-stream',
  );
  res.setHeader('Cache-Control', 'no-store');
  createReadStream(background.file_path).pipe(res);
};

export const deleteBackground = (req, res) => {
  const { id: projectId } = req.params;
  if (!verifyOwnership(projectId, req.user.id)) {
    return res.status(404).json({ error: 'Project not found' });
  }

  deleteExistingBackgroundFile(projectId);
  db.prepare('DELETE FROM project_backgrounds WHERE project_id = ?').run(
    projectId,
  );

  res.json({ ok: true });
};
