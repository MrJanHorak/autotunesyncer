import {
  canWriteProject,
  getProjectAccess,
} from '../services/projectAccessService.js';

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

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

  const project = getProjectAccess(projectId, req.user.id);

  if (!project) {
    return res.status(404).json({ error: 'Project not found' });
  }

  if (!canWriteProject(project)) {
    return res.status(403).json({ error: 'Project write access required' });
  }

  req.project = project;
  next();
};
