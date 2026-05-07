import { mkdirSync } from 'fs';
import { join, resolve, sep } from 'path';
import { dirname } from 'path';
import { fileURLToPath } from 'url';
import db from '../db/database.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
export const BASE_UPLOADS_DIR = resolve(join(__dirname, '../uploads'));

const READ_ROLES = new Set(['viewer', 'editor', 'owner']);
const WRITE_ROLES = new Set(['editor', 'owner']);
const MANAGE_ROLES = new Set(['owner']);

export function canReadProject(access) {
  return Boolean(access && READ_ROLES.has(access.accessRole));
}

export function canWriteProject(access) {
  return Boolean(access && WRITE_ROLES.has(access.accessRole));
}

export function canManageProject(access) {
  return Boolean(access && MANAGE_ROLES.has(access.accessRole));
}

export function getProjectAccess(projectId, userId) {
  const project = db
    .prepare(
      `
        SELECT
          p.*,
          owner.username AS owner_username,
          CASE
            WHEN p.user_id = @userId THEN 'owner'
            ELSE COALESCE(pc.role, NULL)
          END AS access_role
        FROM projects p
        JOIN users owner ON owner.id = p.user_id
        LEFT JOIN project_collaborators pc
          ON pc.project_id = p.id AND pc.user_id = @userId
        WHERE p.id = @projectId
          AND (p.user_id = @userId OR pc.user_id = @userId)
      `,
    )
    .get({ projectId, userId });

  if (!project) return null;

  const uploadsDir = resolve(
    join(BASE_UPLOADS_DIR, project.user_id, projectId),
  );
  if (
    !uploadsDir.startsWith(BASE_UPLOADS_DIR + sep) &&
    uploadsDir !== BASE_UPLOADS_DIR
  ) {
    return null;
  }

  mkdirSync(uploadsDir, { recursive: true });

  return {
    ...project,
    ownerId: project.user_id,
    ownerUsername: project.owner_username,
    accessRole: project.access_role,
    uploadsDir,
  };
}
