import { Router } from 'express';
import { composeVideo } from '../controllers/compositionController.js';
import multer from 'multer';

const router = Router();

// Configure multer for file uploads
const storage = multer.memoryStorage(); // Use memory storage to handle file uploads in memory
const upload = multer({ storage });

router.post('/', upload.any(), composeVideo);

export default router;