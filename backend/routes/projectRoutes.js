import express from 'express';
import multer from 'multer';
import { authenticateToken } from '../middleware/auth.js';
import {
  listProjects,
  createProject,
  getProject,
  updateProject,
  deleteProject,
  saveProjectState,
  loadProjectState,
  getProjectRenderStatus,
  getProjectRenderFile,
  exportProject,
  importProject,
} from '../controllers/projectController.js';
import {
  acceptProjectInvite,
  declineProjectInvite,
  inviteProjectCollaborator,
  listPendingProjectInvites,
  listProjectCollaborators,
} from '../controllers/projectCollaborationController.js';
import {
  saveClip,
  listClips,
  getClipFile,
  deleteClip,
} from '../controllers/clipController.js';
import {
  saveBackground,
  getBackgroundFile,
  deleteBackground,
} from '../controllers/backgroundController.js';

const router = express.Router();

const clipUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 1000 * 1024 * 1024 },
}).single('video');

const backgroundUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 1000 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    if (
      file.mimetype?.startsWith('image/') ||
      file.mimetype?.startsWith('video/')
    ) {
      cb(null, true);
      return;
    }
    cb(new Error('Background must be an image or video file'));
  },
}).single('background');

const zipUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 500 * 1024 * 1024 },
}).single('archive');

router.use(authenticateToken);

router.get('/', listProjects);
router.post('/', createProject);
router.get('/invites', listPendingProjectInvites);
router.post('/invites/:inviteId/accept', acceptProjectInvite);
router.post('/invites/:inviteId/decline', declineProjectInvite);
router.post(
  '/import',
  (req, res, next) =>
    zipUpload(req, res, (err) => {
      if (err) return res.status(400).json({ error: err.message });
      next();
    }),
  importProject,
);
router.get('/:id/collaborators', listProjectCollaborators);
router.post('/:id/invites', inviteProjectCollaborator);
router.get('/:id', getProject);
router.put('/:id', updateProject);
router.delete('/:id', deleteProject);
router.post('/:id/state', saveProjectState);
router.get('/:id/state', loadProjectState);
router.get('/:id/render', getProjectRenderStatus);
router.get('/:id/render/file', getProjectRenderFile);
router.get('/:id/export', exportProject);

// Clip persistence routes
router.get('/:id/clips', listClips);
router.post(
  '/:id/clips',
  (req, res, next) =>
    clipUpload(req, res, (err) => {
      if (err) return res.status(400).json({ error: err.message });
      next();
    }),
  saveClip,
);
router.get('/:id/clips/:instrumentKey/file', getClipFile);
router.delete('/:id/clips/:instrumentKey', deleteClip);

// Background persistence routes
router.post(
  '/:id/background',
  (req, res, next) =>
    backgroundUpload(req, res, (err) => {
      if (err) return res.status(400).json({ error: err.message });
      next();
    }),
  saveBackground,
);
router.get('/:id/background/file', getBackgroundFile);
router.delete('/:id/background', deleteBackground);

export default router;
