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
} from '../controllers/projectController.js';
import { saveClip, listClips, getClipFile, deleteClip } from '../controllers/clipController.js';

const router = express.Router();

const clipUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 1000 * 1024 * 1024 },
}).single('video');

router.use(authenticateToken);

router.get('/', listProjects);
router.post('/', createProject);
router.get('/:id', getProject);
router.put('/:id', updateProject);
router.delete('/:id', deleteProject);
router.post('/:id/state', saveProjectState);
router.get('/:id/state', loadProjectState);

// Clip persistence routes
router.get('/:id/clips', listClips);
router.post('/:id/clips', (req, res, next) => clipUpload(req, res, (err) => {
  if (err) return res.status(400).json({ error: err.message });
  next();
}), saveClip);
router.get('/:id/clips/:instrumentKey/file', getClipFile);
router.delete('/:id/clips/:instrumentKey', deleteClip);

export default router;
