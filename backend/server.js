import express from 'express';
import cors from 'cors';
import midiRoutes from './routes/midiRoutes.js';
import videoRoutes from './routes/videoRoutes.js';
import compositionRoutes from './routes/composition.js';
import autotuneRoutes from './routes/autotuneRoutes.js';
import uploadRoutes from './routes/uploadRoutes.js'; // Add this line

const app = express();

// Enable CORS for all routes
app.use(cors());
app.use(express.json());

// Use routes
app.use('/api/midi', midiRoutes);
app.use('/api/video', videoRoutes);
app.use('/api/compose', compositionRoutes);
app.use('/api/autotune', autotuneRoutes);
app.use('/api/upload', uploadRoutes); // Add this line

app.listen(3000, () => {
  console.log('Server running on port 3000');
});