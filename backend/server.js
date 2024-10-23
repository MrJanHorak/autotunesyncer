import express from 'express';
import midiRoutes from './routes/midiRoutes.js';
import videoRoutes from './routes/videoRoutes.js';

const app = express();

app.use('/api/midi', midiRoutes);
app.use('/api/video', videoRoutes);

app.listen(3000, () => {
  console.log('Server running on port 3000');
});