import fs from 'fs';
import { v4 as uuidv4 } from 'uuid';
import db from '../db/database.js';
import { getProjectAccess } from './projectAccessService.js';

const UPSERT_PROJECT_RENDER_SQL = `
  INSERT INTO project_renders (
    project_id,
    job_id,
    status,
    progress,
    mode,
    output_path,
    error,
    started_at,
    completed_at,
    updated_at
  )
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
  ON CONFLICT(project_id) DO UPDATE SET
    job_id = excluded.job_id,
    status = excluded.status,
    progress = excluded.progress,
    mode = excluded.mode,
    output_path = excluded.output_path,
    error = excluded.error,
    started_at = COALESCE(excluded.started_at, project_renders.started_at),
    completed_at = COALESCE(excluded.completed_at, project_renders.completed_at),
    updated_at = datetime('now')
`;

function toIsoString(value) {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date.toISOString();
}

function hasOutputFile(outputPath) {
  return Boolean(outputPath && fs.existsSync(outputPath));
}

export function getStoredProjectRender(projectId) {
  return (
    db
      .prepare(
        `
          SELECT project_id, job_id, status, progress, mode, output_path, error,
                 started_at, completed_at, updated_at
          FROM project_renders
          WHERE project_id = ?
        `,
      )
      .get(projectId) || null
  );
}

export function getProjectRender(projectId, userId) {
  const access = getProjectAccess(projectId, userId);
  if (!access) return null;
  const row = db
    .prepare(
      `
        SELECT
          p.id AS project_id,
          p.name AS project_name,
          pr.job_id,
          pr.status,
          pr.progress,
          pr.mode,
          pr.output_path,
          pr.error,
          pr.started_at,
          pr.completed_at,
          pr.updated_at
        FROM projects p
        LEFT JOIN project_renders pr ON pr.project_id = p.id
        WHERE p.id = ?
      `,
    )
    .get(projectId);

  const outputAvailable = hasOutputFile(row.output_path);
  return {
    projectId: row.project_id,
    projectName: row.project_name,
    jobId: row.job_id || null,
    status: row.status || 'idle',
    progress: Number.isFinite(Number(row.progress)) ? Number(row.progress) : 0,
    mode: row.mode || 'full',
    error: row.error || null,
    outputPath: outputAvailable ? row.output_path : null,
    hasOutput: outputAvailable,
    startedAt: row.started_at || null,
    completedAt: row.completed_at || null,
    updatedAt: row.updated_at || null,
  };
}

export function syncProjectRender(job) {
  if (!job?.persistRender || !job.projectId) return;

  db.prepare(UPSERT_PROJECT_RENDER_SQL).run(
    job.projectId,
    job.jobId || null,
    job.status || 'idle',
    Number.isFinite(Number(job.progress))
      ? Math.round(Number(job.progress))
      : 0,
    job.mode || 'full',
    job.outputPath || null,
    job.error || null,
    toIsoString(job.startedAt),
    toIsoString(job.completedAt),
  );
}

export function createProjectRenderNotification(userId, projectId, type) {
  if (!userId || !projectId || !type) return;

  db.prepare(
    `
      INSERT INTO notifications (id, user_id, actor_id, type, project_id)
      VALUES (?, ?, ?, ?, ?)
    `,
  ).run(uuidv4(), userId, userId, type, projectId);
}
