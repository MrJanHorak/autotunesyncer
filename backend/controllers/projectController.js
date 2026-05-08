import { v4 as uuidv4 } from 'uuid';
import {
  rmSync,
  existsSync,
  mkdirSync,
  writeFileSync,
  createReadStream,
  readdirSync,
  statSync,
} from 'fs';
import { join, resolve, sep, basename, extname } from 'path';
import { dirname } from 'path';
import { fileURLToPath } from 'url';
import { tmpdir } from 'os';
import archiver from 'archiver';
import AdmZip from 'adm-zip';
import db from '../db/database.js';
import {
  DEFAULT_RENDER_PRESET,
  normalizeRenderPreset,
} from '../../shared/renderPresets.js';
import { normalizeGridArrangement } from '../../shared/gridLayout.js';
import { getProjectRender } from '../services/projectRenderService.js';
import {
  canManageProject,
  canWriteProject,
  getProjectAccess,
} from '../services/projectAccessService.js';
import { emitProjectStateSaved } from '../services/realtimeService.js';
import { getBillingAccess } from '../services/billingAccessService.js';

const PROJECT_STATE_SCHEMA_VERSION = 2;
const PROJECT_CARD_PREVIEW_ITEM_LIMIT = 12;
const EXPORT_REQUIRED_PLAN = 'creator';

const __dirname = dirname(fileURLToPath(import.meta.url));
const BASE_UPLOADS_DIR = resolve(join(__dirname, '../uploads'));

const buildExportBillingState = (billingAccess) => ({
  requiredPlan: EXPORT_REQUIRED_PLAN,
  planKey: billingAccess.planKey,
  isActive: billingAccess.isActive,
  canExportProject: billingAccess.canExportProject,
  canManageCollaboration: billingAccess.canManageCollaboration,
});

function requireCreatorPlan(ownerId, res, message) {
  const billingAccess = getBillingAccess(ownerId);
  if (billingAccess.canExportProject) {
    return true;
  }

  res.status(403).json({
    error: message,
    billing: buildExportBillingState(billingAccess),
  });
  return false;
}

const PROJECT_SELECT = `
  WITH accessible_projects AS (
    SELECT id AS project_id, 'owner' AS access_role
    FROM projects
    WHERE user_id = ?

    UNION

    SELECT project_id, role AS access_role
    FROM project_collaborators
    WHERE user_id = ?
  ),
  clip_counts AS (
    SELECT project_id, COUNT(*) AS clip_count
    FROM project_clips
    GROUP BY project_id
  ),
  collaborator_counts AS (
    SELECT project_id, COUNT(*) AS collaborator_count
    FROM project_collaborators
    GROUP BY project_id
  )
  SELECT
    p.id,
    p.user_id AS owner_id,
    owner.username AS owner_username,
    ap.access_role,
    p.name,
    p.description,
    p.state,
    p.state_version,
    p.created_at,
    p.updated_at,
    COALESCE(cc.clip_count, 0) AS clip_count,
    COALESCE(colc.collaborator_count, 0) AS collaborator_count,
    pb.media_kind AS background_kind,
    pr.status AS render_status,
    pr.progress AS render_progress,
    pr.output_path AS render_output_path,
    pr.error AS render_error,
    pr.completed_at AS render_completed_at
  FROM accessible_projects ap
  JOIN projects p ON p.id = ap.project_id
  JOIN users owner ON owner.id = p.user_id
  LEFT JOIN clip_counts cc ON cc.project_id = p.id
  LEFT JOIN collaborator_counts colc ON colc.project_id = p.id
  LEFT JOIN project_backgrounds pb ON pb.project_id = p.id
  LEFT JOIN project_renders pr ON pr.project_id = p.id
`;

const parseProjectState = (stateText) => {
  if (!stateText) return null;
  try {
    return JSON.parse(stateText);
  } catch {
    return null;
  }
};

const getLayoutPreview = (arrangement) => {
  const normalized = normalizeGridArrangement(arrangement);
  const entries = Object.entries(normalized.items);

  if (entries.length === 0) {
    return {
      itemCount: 0,
      preview: null,
    };
  }

  return {
    itemCount: entries.length,
    preview: {
      columns: normalized.columns,
      rows: normalized.rows,
      items: entries
        .slice(0, PROJECT_CARD_PREVIEW_ITEM_LIMIT)
        .map(([id, item]) => ({
          id,
          x: item.x,
          y: item.y,
          w: item.w,
          h: item.h,
          type: item.type || 'track',
        })),
    },
  };
};

const getWorkflowState = ({
  clipCount,
  hasMidi,
  layoutItemCount,
  hasBackground,
  renderStatus,
  hasRenderOutput,
}) => {
  if (renderStatus === 'processing' || renderStatus === 'queued') {
    return { stage: 'rendering', progress: 86 };
  }

  if (renderStatus === 'failed') {
    return { stage: 'render_failed', progress: 72 };
  }

  if (hasRenderOutput || renderStatus === 'done') {
    return { stage: 'rendered', progress: 94 };
  }

  if (clipCount > 0 && hasMidi && layoutItemCount > 0) {
    return { stage: 'ready', progress: 72 };
  }

  if (clipCount > 0 || hasMidi || layoutItemCount > 0 || hasBackground) {
    return { stage: 'building', progress: 42 };
  }

  return { stage: 'draft', progress: 12 };
};

const buildProjectSummary = (row) => {
  const state = parseProjectState(row.state);
  const renderPreset = normalizeRenderPreset(
    state?.renderPreset,
    DEFAULT_RENDER_PRESET,
  );
  const { itemCount, preview } = getLayoutPreview(state?.gridArrangement);
  const clipCount = Number(row.clip_count) || 0;
  const renderStatus = row.render_status || 'idle';
  const renderProgress =
    renderStatus === 'done'
      ? 100
      : Math.max(0, Math.min(100, Number(row.render_progress) || 0));
  const hasRenderOutput = Boolean(
    row.render_output_path && existsSync(row.render_output_path),
  );
  const hasBackground = Boolean(row.background_kind);
  const hasMidi = Boolean(state?.midiFileBase64);
  const workflow = getWorkflowState({
    clipCount,
    hasMidi,
    layoutItemCount: itemCount,
    hasBackground,
    renderStatus,
    hasRenderOutput,
  });

  return {
    id: row.id,
    ownerId: row.owner_id,
    ownerUsername: row.owner_username,
    accessRole: row.access_role,
    stateVersion: Number(row.state_version) || 1,
    name: row.name,
    description: row.description,
    created_at: row.created_at,
    updated_at: row.updated_at,
    layoutPreview: preview,
    summary: {
      clipCount,
      collaboratorCount: Number(row.collaborator_count) || 0,
      hasBackground,
      backgroundKind: row.background_kind || null,
      hasMidi,
      layoutItemCount: itemCount,
      renderPreset,
      renderStatus,
      renderProgress,
      hasRenderOutput,
      renderError: row.render_error || null,
      renderCompletedAt: row.render_completed_at || null,
      workflowStage: workflow.stage,
      workflowProgress: workflow.progress,
    },
  };
};

export const listProjects = (req, res) => {
  const projects = db
    .prepare(
      `${PROJECT_SELECT}
       ORDER BY p.updated_at DESC`,
    )
    .all(req.user.id, req.user.id)
    .map(buildProjectSummary);
  res.json({ projects });
};

export const createProject = (req, res) => {
  const { name, description = '', renderPreset } = req.body;
  if (!name?.trim()) {
    return res.status(400).json({ error: 'Project name is required' });
  }

  const id = uuidv4();
  const normalizedRenderPreset = normalizeRenderPreset(
    renderPreset,
    DEFAULT_RENDER_PRESET,
  );
  const initialState = JSON.stringify({
    renderPreset: normalizedRenderPreset,
    schemaVersion: PROJECT_STATE_SCHEMA_VERSION,
    savedAt: new Date().toISOString(),
  });
  db.prepare(
    'INSERT INTO projects (id, user_id, name, description, state) VALUES (?, ?, ?, ?, ?)',
  ).run(id, req.user.id, name.trim(), description.trim(), initialState);

  const project = db
    .prepare(
      `${PROJECT_SELECT}
       WHERE p.id = ?`,
    )
    .get(req.user.id, req.user.id, id);
  res.status(201).json({
    project: buildProjectSummary(project),
  });
};

export const getProject = (req, res) => {
  const project = db
    .prepare(
      `${PROJECT_SELECT}
       WHERE p.id = ?`,
    )
    .get(req.user.id, req.user.id, req.params.id);
  if (!project) return res.status(404).json({ error: 'Project not found' });
  res.json({ project: buildProjectSummary(project) });
};

export const updateProject = (req, res) => {
  const { name, description } = req.body;
  const project = getProjectAccess(req.params.id, req.user.id);
  if (!project) return res.status(404).json({ error: 'Project not found' });
  if (!canManageProject(project)) {
    return res
      .status(403)
      .json({ error: 'Only the project owner can edit project details' });
  }

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
      `${PROJECT_SELECT}
       WHERE p.id = ?`,
    )
    .get(req.user.id, req.user.id, req.params.id);
  res.json({ project: buildProjectSummary(updated) });
};

export const deleteProject = (req, res) => {
  const project = getProjectAccess(req.params.id, req.user.id);
  if (!project) return res.status(404).json({ error: 'Project not found' });
  if (!canManageProject(project)) {
    return res
      .status(403)
      .json({ error: 'Only the project owner can delete a project' });
  }

  // Delete project-scoped uploads folder safely
  const uploadsDir = resolve(
    join(BASE_UPLOADS_DIR, project.ownerId, req.params.id),
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
  const project = getProjectAccess(req.params.id, req.user.id);
  if (!project) return res.status(404).json({ error: 'Project not found' });
  if (!canWriteProject(project)) {
    return res.status(403).json({ error: 'Project write access required' });
  }

  const baseStateVersion = Number(req.body?.baseStateVersion);
  const realtimeClientId = req.body?.realtimeClientId || null;
  const currentStateVersion = Number(project.state_version) || 1;

  if (!Number.isFinite(baseStateVersion)) {
    return res.status(400).json({
      error: 'baseStateVersion is required',
      currentStateVersion,
    });
  }

  if (baseStateVersion !== currentStateVersion) {
    let currentState = null;
    try {
      currentState = project.state ? JSON.parse(project.state) : null;
    } catch {
      currentState = null;
    }

    return res.status(409).json({
      error: 'Project state has changed since you loaded it',
      currentStateVersion,
      currentState,
    });
  }

  const nextStateVersion = currentStateVersion + 1;

  const state = JSON.stringify({
    ...req.body,
    schemaVersion: PROJECT_STATE_SCHEMA_VERSION,
    baseStateVersion: undefined,
    realtimeClientId: undefined,
    stateVersion: nextStateVersion,
    savedAt: new Date().toISOString(),
  });
  const result = db
    .prepare(
      "UPDATE projects SET state = ?, state_version = ?, updated_at = datetime('now') WHERE id = ? AND state_version = ?",
    )
    .run(state, nextStateVersion, req.params.id, currentStateVersion);

  if (result.changes === 0) {
    const latestProject = getProjectAccess(req.params.id, req.user.id);
    let currentState = null;
    try {
      currentState = latestProject?.state
        ? JSON.parse(latestProject.state)
        : null;
    } catch {
      currentState = null;
    }

    return res.status(409).json({
      error: 'Project state has changed since you loaded it',
      currentStateVersion:
        Number(latestProject?.state_version) || nextStateVersion,
      currentState,
    });
  }

  const updatedProject = getProjectAccess(req.params.id, req.user.id);
  const updatedAt = updatedProject?.updated_at || new Date().toISOString();

  const hasCollaborators = Boolean(
    project.accessRole !== 'owner' ||
    db
      .prepare(
        'SELECT 1 FROM project_collaborators WHERE project_id = ? LIMIT 1',
      )
      .get(req.params.id),
  );

  if (hasCollaborators) {
    emitProjectStateSaved(req.params.id, {
      actor: {
        id: req.user.id,
        username: req.user.username,
      },
      clientId: realtimeClientId,
      stateVersion: Number(updatedProject?.state_version) || nextStateVersion,
      updatedAt,
    });
  }

  res.json({
    message: 'State saved',
    stateVersion: Number(updatedProject?.state_version) || nextStateVersion,
    updatedAt,
  });
};

export const loadProjectState = (req, res) => {
  const project = getProjectAccess(req.params.id, req.user.id);
  if (!project) return res.status(404).json({ error: 'Project not found' });

  if (!project.state) {
    return res.json({
      state: null,
      stateVersion: Number(project.state_version) || 1,
      updatedAt: project.updated_at || null,
    });
  }

  try {
    res.json({
      state: JSON.parse(project.state),
      stateVersion: Number(project.state_version) || 1,
      updatedAt: project.updated_at || null,
    });
  } catch {
    res.json({
      state: null,
      stateVersion: Number(project.state_version) || 1,
      updatedAt: project.updated_at || null,
    });
  }
};

export const getProjectRenderStatus = (req, res) => {
  const project = getProjectAccess(req.params.id, req.user.id);
  if (!project) return res.status(404).json({ error: 'Project not found' });
  const render = getProjectRender(req.params.id, req.user.id);
  if (!render) return res.status(404).json({ error: 'Project not found' });

  res.json({
    render,
    billing: buildExportBillingState(getBillingAccess(project.ownerId)),
  });
};

export const getProjectRenderFile = (req, res) => {
  const project = getProjectAccess(req.params.id, req.user.id);
  if (!project) return res.status(404).json({ error: 'Project not found' });
  if (
    !requireCreatorPlan(
      project.ownerId,
      res,
      'Downloading rendered videos requires the Creator plan or higher.',
    )
  ) {
    return;
  }

  const render = getProjectRender(req.params.id, req.user.id);
  if (!render) return res.status(404).json({ error: 'Project not found' });
  if (!render.outputPath || !existsSync(render.outputPath)) {
    return res.status(404).json({ error: 'Rendered composition not found' });
  }

  const stats = statSync(render.outputPath);
  res.setHeader('Content-Type', 'video/mp4');
  res.setHeader('Content-Length', stats.size);
  res.setHeader(
    'Content-Disposition',
    'inline; filename="project-composition.mp4"',
  );

  const stream = createReadStream(render.outputPath);
  stream.on('error', (err) => {
    console.error('[project render] stream error:', err.message);
    if (!res.headersSent) {
      res.status(500).json({ error: 'Failed to stream rendered composition' });
    }
  });
  stream.pipe(res);
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
  const project = getProjectAccess(req.params.id, req.user.id);
  if (!project) return res.status(404).json({ error: 'Project not found' });
  if (
    !requireCreatorPlan(
      project.ownerId,
      res,
      'Project export requires the Creator plan or higher.',
    )
  ) {
    return;
  }

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
      {
        name: project.name,
        id: project.id,
        schemaVersion: PROJECT_STATE_SCHEMA_VERSION,
      },
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
      return res.status(400).json({
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
