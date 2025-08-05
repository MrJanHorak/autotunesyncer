import express from 'express';
import cors from 'cors';
import midiRoutes from './routes/midiRoutes.js';
import videoRoutes from './routes/videoRoutes.js';
import compositionRoutes from './routes/composition.js';
import autotuneRoutes from './routes/autotuneRoutes.js';
import uploadRoutes from './routes/uploadRoutes.js';
import processVideos from './routes/processVideos.js';

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
app.use(
  cors({
    origin: 'http://localhost:5173', // Your frontend URL
    methods: ['GET', 'POST'],
    allowedHeaders: ['Content-Type', 'Content-Disposition'],
    maxAge: 600,
    exposedHeaders: ['Content-Length', 'Content-Type', 'Content-Disposition'],
  })
);

// Use routes
app.use('/api/midi', midiRoutes);
app.use('/api/video', videoRoutes);
app.use('/api/compose', compositionRoutes);
app.use('/api/autotune', autotuneRoutes);
app.use('/api/upload', uploadRoutes);
app.use('/api/process-videos', processVideos);

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
