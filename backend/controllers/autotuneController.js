import ffmpeg from 'fluent-ffmpeg';
import { join } from 'path';
import { v4 as uuidv4 } from 'uuid';
import { existsSync, mkdirSync, writeFileSync, readFileSync, rmSync } from 'fs';

import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const TEMP_DIR = join(__dirname, '../temp');
if (!existsSync(TEMP_DIR)) mkdirSync(TEMP_DIR);

export const autotuneVideo = async (req, res) => {
  console.log('Autotuning video...');
  console.log('Request body:', req.body);
  console.log('Request files:', req.files);
  const sessionId = uuidv4();
  const sessionDir = join(TEMP_DIR, sessionId);
  mkdirSync(sessionDir);

  const videoFile = req.files.video;
  const inputPath = join(sessionDir, videoFile.name);
  const outputPath = join(sessionDir, 'autotuned-video.mp4');

  writeFileSync(inputPath, videoFile.data);

  ffmpeg(inputPath)
    .audioFilters('asetrate=44100*2/3,atempo=3/2')
    .videoFilters('setpts=PTS-STARTPTS')
    .output(outputPath)
    .on('end', () => {
      const autotunedVideo = readFileSync(outputPath);
      res.send(autotunedVideo);
      // Cleanup
      rmSync(sessionDir, { recursive: true, force: true });
    })
    .on('error', (err) => {
      console.error('Error autotuning video:', err);
      res.status(500).send('Error autotuning video');
      // Cleanup
      rmSync(sessionDir, { recursive: true, force: true });
    })
    .run();
};