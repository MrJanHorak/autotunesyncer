import express from 'express';
import cors from 'cors';
import { existsSync, readdirSync, statSync, unlinkSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import midiRoutes from './routes/midiRoutes.js';
import videoRoutes from './routes/videoRoutes.js';
import compositionRoutes from './routes/composition.js';
import autotuneRoutes from './routes/autotuneRoutes.js';
import uploadRoutes from './routes/uploadRoutes.js';
import processVideos from './routes/processVideos.js';
import precacheRoutes from './routes/precache.js';
import authRoutes from './routes/authRoutes.js';
import projectRoutes from './routes/projectRoutes.js';
import socialRoutes from './routes/socialRoutes.js';

const app = express();

// Increase payload size limits significantly
app.use(express.json({ limit: '1000mb' }));
app.use(express.urlencoded({ limit: '1000mb', extended: true }));

// Configure timeout
app.use((req, res, next) => {
  res.setTimeout(900000, () => {
    console.log('Request has timed out.');
    console.log('Request has timed out.');
    res.status(408).json({ error: 'Request timeout' });
  });
  next();
});

// Enable CORS with specific options
const allowedOrigins = [
  'http://localhost:5173',
  'http://localhost:8080',
  'http://localhost:4173',
];
app.use(
  cors({
    origin: (origin, callback) => {
      // Allow requests with no origin (e.g., curl, Postman) or from allowed origins
      if (!origin || allowedOrigins.includes(origin)) {
        callback(null, true);
      } else {
        callback(new Error(`CORS: origin ${origin} not allowed`));
      }
    },
    methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Content-Disposition', 'Authorization'],
    maxAge: 600,
    exposedHeaders: ['Content-Length', 'Content-Type', 'Content-Disposition'],
  })
);

// Serve published compositions as static files (public, intentionally shareable)
const __dirnameServer = dirname(fileURLToPath(import.meta.url));
const publishedDir = join(__dirnameServer, 'published');
mkdirSync(publishedDir, { recursive: true });
app.use('/published', express.static(publishedDir));

// Use routes
app.use('/api/auth', authRoutes);
app.use('/api/projects', projectRoutes);
app.use('/api/social', socialRoutes);
app.use('/api/midi', midiRoutes);
app.use('/api/video', videoRoutes);
app.use('/api/compose', compositionRoutes);
app.use('/api/autotune', autotuneRoutes);
app.use('/api/upload', uploadRoutes);
app.use('/api/process-videos', processVideos);
app.use('/api/autotune/precache', precacheRoutes);

// Add error handling for large payloads
app.use((err, req, res, next) => {
  console.error('Express error:', err);
  if (err.type === 'entity.too.large') {
    return res.status(413).json({ error: 'Request entity too large' });
  }
  next(err);
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(err.status || 500).json({
    error: err.message || 'Internal Server Error',
  });
});

app.listen(3000, () => {
  console.log('Server running on port 3000');
  console.log('Server running on port 3000');
});

// Recursively clean up processed_* files in uploads/ (including user/project subdirs) older than 7 days
const __dirname = dirname(fileURLToPath(import.meta.url));
const uploadsDir = join(__dirname, 'uploads');
const TTL_MS = 7 * 24 * 60 * 60 * 1000;

function cleanUploadsDir(dir) {
  if (!existsSync(dir)) return 0;
  const now = Date.now();
  let removed = 0;
  for (const name of readdirSync(dir)) {
    const fullPath = join(dir, name);
    try {
      const stat = statSync(fullPath);
      if (stat.isDirectory()) {
        removed += cleanUploadsDir(fullPath); // recurse into user/project dirs
      } else if (name.startsWith('processed_')) {
        const age = now - stat.mtimeMs;
        if (age > TTL_MS) {
          unlinkSync(fullPath);
          removed++;
        }
      }
    } catch {
      // file already gone or stat failed — skip
    }
  }
  return removed;
}

function cleanUploads() {
  const removed = cleanUploadsDir(uploadsDir);
  if (removed > 0) console.log(`[TTL cleanup] Removed ${removed} stale processed_ file(s) from uploads/`);
}

cleanUploads();
setInterval(cleanUploads, 24 * 60 * 60 * 1000);
