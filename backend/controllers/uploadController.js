import { writeFileSync } from 'fs';
import { join } from 'path';
import { v4 as uuidv4 } from 'uuid';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { preprocessVideo } from '../js/pythonBridge.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const UPLOADS_DIR = join(__dirname, '../uploads');
const PROCESSED_DIR = join(__dirname, '../processed');


// export const handleUpload = (req, res) => {
//   console.log('Uploading file:', req);
//   try {
//     const file = req.file;
//     if (!file) {
//       return res.status(400).json({ error: 'No file uploaded' });
//     }

//     const filePath = join(UPLOADS_DIR, `${uuidv4()}.mp4`);
//     console.log('Uploading file:', filePath);
//     writeFileSync(filePath, file.buffer);

//     res.status(200).json({ message: 'File uploaded successfully', filePath });
//   } catch (error) {
//     console.error('Error handling upload:', error);
//     res.status(500).json({ error: 'Internal server error' });
//   }
// };

export const handleUpload = async (req, res) => {
  try {
    const file = req.file;
    if (!file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    // Save original upload
    const originalPath = join(UPLOADS_DIR, `${uuidv4()}.mp4`);
    writeFileSync(originalPath, file.buffer);

    // Process the video
    const processedPath = join(PROCESSED_DIR, `processed_${uuidv4()}.mp4`);
    try {
      await preprocessVideo(originalPath, processedPath);
      res.status(200).json({ 
        message: 'File uploaded and processed successfully', 
        originalPath,
        processedPath 
      });
    } catch (preprocessError) {
      console.error('Error preprocessing video:', preprocessError);
      res.status(500).json({ 
        error: 'Video preprocessing failed',
        details: preprocessError.message 
      });
    }

  } catch (error) {
    console.error('Error handling upload:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};
