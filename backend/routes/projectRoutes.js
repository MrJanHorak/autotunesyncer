import express from 'express';
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

const router = express.Router();

router.use(authenticateToken);

router.get('/', listProjects);
router.post('/', createProject);
router.get('/:id', getProject);
router.put('/:id', updateProject);
router.delete('/:id', deleteProject);
router.post('/:id/state', saveProjectState);
router.get('/:id/state', loadProjectState);

export default router;
