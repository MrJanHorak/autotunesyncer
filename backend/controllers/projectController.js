import { v4 as uuidv4 } from 'uuid';
import {
  rmSync,
  existsSync,
  mkdirSync,
  writeFileSync,
  createReadStream,
  readdirSync,
} from 'fs';
import { join, resolve, sep, basename, extname } from 'path';
import { dirname } from 'path';
import { fileURLToPath } from 'url';
import { tmpdir } from 'os';
import archiver from 'archiver';
import AdmZip from 'adm-zip';
import db from '../db/database.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const BASE_UPLOADS_DIR = resolve(join(__dirname, '../uploads'));

export const listProjects = (req, res) => {
  const projects = db
    .prepare(
      'SELECT id, name, description, created_at, updated_at FROM projects WHERE user_id = ? ORDER BY updated_at DESC',
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
  db.prepare(
    'INSERT INTO projects (id, user_id, name, description) VALUES (?, ?, ?, ?)',
  ).run(id, req.user.id, name.trim(), description.trim());

  const project = db
    .prepare(
      'SELECT id, name, description, created_at, updated_at FROM projects WHERE id = ?',
    )
    .get(id);
  res.status(201).json({ project });
};

export const getProject = (req, res) => {
  const project = db
    .prepare(
      'SELECT id, name, description, created_at, updated_at FROM projects WHERE id = ? AND user_id = ?',
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
    db.prepare(
      "UPDATE projects SET name = ?, updated_at = datetime('now') WHERE id = ?",
    ).run(name.trim(), req.params.id);
  }
  if (description !== undefined) {
    db.prepare(
      "UPDATE projects SET description = ?, updated_at = datetime('now') WHERE id = ?",
    ).run(description.trim(), req.params.id);
  }

  const updated = db
    .prepare(
      'SELECT id, name, description, created_at, updated_at FROM projects WHERE id = ?',
    )
    .get(req.params.id);
  res.json({ project: updated });
};

export const deleteProject = (req, res) => {
  const project = db
    .prepare('SELECT id FROM projects WHERE id = ? AND user_id = ?')
    .get(req.params.id, req.user.id);
  if (!project) return res.status(404).json({ error: 'Project not found' });

  // Delete project-scoped uploads folder safely
  const uploadsDir = resolve(
    join(BASE_UPLOADS_DIR, req.user.id, req.params.id),
  );
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
  db.prepare(
    "UPDATE projects SET state = ?, updated_at = datetime('now') WHERE id = ?",
  ).run(state, req.params.id);
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

const SAFE_KEY_RE = /^[a-z0-9_()\-]{1,80}$/i;
const MAX_ZIP_SIZE = 500 * 1024 * 1024; // 500 MB
const MAX_ZIP_ENTRIES = 500;

const inferMimeTypeFromExt = (extension = '') => {
  const ext = String(extension || '').toLowerCase();
  const byExt = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.webp': 'image/webp',
    '.gif': 'image/gif',
    '.bmp': 'image/bmp',
    '.mp4': 'video/mp4',
    '.mov': 'video/quicktime',
    '.webm': 'video/webm',
    '.mkv': 'video/x-matroska',
  };
  return byExt[ext] || 'application/octet-stream';
};

export const exportProject = (req, res) => {
  const project = db
    .prepare(
      'SELECT id, name, state FROM projects WHERE id = ? AND user_id = ?',
    )
    .get(req.params.id, req.user.id);
  if (!project) return res.status(404).json({ error: 'Project not found' });

  const clips = db
    .prepare(
      'SELECT instrument_key, file_path FROM project_clips WHERE project_id = ?',
    )
    .all(req.params.id);
  const background = db
    .prepare(
      'SELECT file_path, mime_type, media_kind, original_name FROM project_backgrounds WHERE project_id = ?',
    )
    .get(req.params.id);

  const safeName = (project.name || 'project')
    .replace(/[^a-z0-9_-]/gi, '_')
    .slice(0, 60);
  res.setHeader('Content-Type', 'application/zip');
  res.setHeader(
    'Content-Disposition',
    `attachment; filename="${safeName}.zip"`,
  );

  const zip = archiver('zip', { zlib: { level: 6 } });
  zip.on('error', (err) => {
    console.error('[export] archiver error:', err);
    if (!res.headersSent)
      res.status(500).json({ error: 'ZIP creation failed' });
  });
  zip.pipe(res);

  zip.append(
    JSON.stringify(
      { name: project.name, id: project.id, schemaVersion: 1 },
      null,
      2,
    ),
    { name: 'manifest.json' },
  );
  zip.append(project.state || '{}', { name: 'state.json' });

  for (const { instrument_key: key, file_path: fp } of clips) {
    if (!SAFE_KEY_RE.test(key)) continue;
    const safePath = resolve(fp);
    if (!safePath.startsWith(BASE_UPLOADS_DIR + sep)) continue;
    if (!existsSync(safePath)) continue;
    zip.file(safePath, { name: `clips/${key}.mp4` });
  }

  if (background?.file_path) {
    const safeBackgroundPath = resolve(background.file_path);
    if (
      safeBackgroundPath.startsWith(BASE_UPLOADS_DIR + sep) &&
      existsSync(safeBackgroundPath)
    ) {
      const backgroundExt =
        extname(background.original_name || safeBackgroundPath) || '.bin';
      zip.append(
        JSON.stringify(
          {
            mimeType: background.mime_type,
            mediaKind: background.media_kind,
            originalName: background.original_name,
          },
          null,
          2,
        ),
        { name: 'background/meta.json' },
      );
      zip.file(safeBackgroundPath, {
        name: `background/asset${backgroundExt.toLowerCase()}`,
      });
    }
  }

  zip.finalize();
};

export const importProject = (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'ZIP file required' });
  if (req.file.size > MAX_ZIP_SIZE)
    return res.status(413).json({ error: 'ZIP too large (max 500 MB)' });

  const tmpZipPath = join(tmpdir(), `ats_import_${uuidv4()}.zip`);
  try {
    writeFileSync(tmpZipPath, req.file.buffer);
    const zip = new AdmZip(tmpZipPath);
    const entries = zip.getEntries();

    if (entries.length > MAX_ZIP_ENTRIES) {
      return res.status(400).json({ error: 'ZIP contains too many entries' });
    }

    // Parse manifest + state
    const manifestEntry = entries.find((e) => e.entryName === 'manifest.json');
    const stateEntry = entries.find((e) => e.entryName === 'state.json');
    if (!manifestEntry || !stateEntry) {
      return res
        .status(400)
        .json({
          error: 'Invalid project ZIP (missing manifest.json or state.json)',
        });
    }

    let manifest;
    try {
      manifest = JSON.parse(manifestEntry.getData().toString('utf8'));
    } catch {
      return res.status(400).json({ error: 'Corrupt manifest.json' });
    }

    const stateJson = stateEntry.getData().toString('utf8');

    // Collect clip entries and validate keys
    const clipEntries = entries.filter(
      (e) =>
        e.entryName.startsWith('clips/') &&
        e.entryName.endsWith('.mp4') &&
        !e.isDirectory,
    );
    const backgroundMetaEntry = entries.find(
      (e) => e.entryName === 'background/meta.json',
    );
    const backgroundAssetEntry = entries.find(
      (e) => e.entryName.startsWith('background/asset') && !e.isDirectory,
    );
    let backgroundMeta = null;
    if (backgroundMetaEntry) {
      try {
        backgroundMeta = JSON.parse(
          backgroundMetaEntry.getData().toString('utf8'),
        );
      } catch {
        return res.status(400).json({ error: 'Corrupt background/meta.json' });
      }
    }
    for (const e of clipEntries) {
      const key = basename(e.entryName, '.mp4');
      if (!SAFE_KEY_RE.test(key)) {
        return res
          .status(400)
          .json({ error: `Invalid instrument key in ZIP: ${key}` });
      }
    }

    // Create project + clips inside a transaction
    const newId = uuidv4();
    const newName = (manifest.name || 'Imported Project').slice(0, 120);
    const uploadsDir = resolve(join(BASE_UPLOADS_DIR, req.user.id, newId));

    const importTx = db.transaction(() => {
      db.prepare(
        'INSERT INTO projects (id, user_id, name, description, state) VALUES (?, ?, ?, ?, ?)',
      ).run(newId, req.user.id, newName, '', stateJson);

      mkdirSync(uploadsDir, { recursive: true });

      for (const e of clipEntries) {
        const key = basename(e.entryName, '.mp4');
        const filePath = join(uploadsDir, `clip_${key}_${uuidv4()}.mp4`);
        writeFileSync(filePath, e.getData());
        db.prepare(
          `
          INSERT INTO project_clips (project_id, instrument_key, file_path)
          VALUES (?, ?, ?)
          ON CONFLICT(project_id, instrument_key) DO UPDATE
            SET file_path = excluded.file_path, created_at = datetime('now')
        `,
        ).run(newId, key, filePath);
      }

      if (backgroundAssetEntry) {
        const backgroundExt =
          extname(
            backgroundMeta?.originalName || backgroundAssetEntry.entryName,
          ) ||
          extname(backgroundAssetEntry.entryName) ||
          '.bin';
        const backgroundPath = join(
          uploadsDir,
          `background_${uuidv4()}${backgroundExt.toLowerCase()}`,
        );
        writeFileSync(backgroundPath, backgroundAssetEntry.getData());
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
          newId,
          backgroundPath,
          backgroundMeta?.mimeType || inferMimeTypeFromExt(backgroundExt),
          backgroundMeta?.mediaKind ||
            (inferMimeTypeFromExt(backgroundExt).startsWith('video/')
              ? 'video'
              : 'image'),
          backgroundMeta?.originalName ||
            basename(backgroundAssetEntry.entryName),
        );
      }
    });

    try {
      importTx();
    } catch (txErr) {
      // Rollback: remove any files written before the transaction threw
      try {
        rmSync(uploadsDir, { recursive: true, force: true });
      } catch {
        /* ignore */
      }
      throw txErr;
    }

    const created = db
      .prepare(
        'SELECT id, name, description, created_at, updated_at FROM projects WHERE id = ?',
      )
      .get(newId);
    res.status(201).json({ project: created });
  } catch (err) {
    console.error('[import] error:', err);
    res.status(500).json({ error: 'Import failed' });
  } finally {
    try {
      rmSync(tmpZipPath);
    } catch {
      /* ignore */
    }
  }
};
