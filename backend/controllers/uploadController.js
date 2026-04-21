import { writeFileSync, mkdirSync } from 'fs';
import { join, resolve, sep } from 'path';
import { v4 as uuidv4 } from 'uuid';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { preprocessVideo } from '../js/pythonBridge.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const BASE_UPLOADS_DIR = resolve(join(__dirname, '../uploads'));


export const handleUpload = async (req, res) => {
  try {
    const file = req.file;
    if (!file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    // req.project is set by requireProjectOwnership middleware
    const uploadsDir = req.project?.uploadsDir ?? BASE_UPLOADS_DIR;
    mkdirSync(uploadsDir, { recursive: true });

    // Save original upload into project-scoped directory
    const originalPath = join(uploadsDir, `${uuidv4()}.mp4`);
    writeFileSync(originalPath, file.buffer);

    // Return a server-relative identifier, not the raw filesystem path
    const relPath = originalPath.replace(BASE_UPLOADS_DIR + sep, '').replace(/\\/g, '/');
    res.status(200).json({
      message: 'File uploaded successfully',
      filePath: relPath,
    });
  } catch (error) {
    console.error('Error handling upload:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};

