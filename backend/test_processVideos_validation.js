import assert from 'assert';
import express from 'express';
import processVideosRouter from './routes/processVideos.js';

const makeMidiPayload = (overrides = {}) => ({
  tracks: [
    {
      channel: 0,
      instrument: { name: 'Piano', family: 'piano' },
      notes: [{ midi: 60, time: 0, duration: 0.5, velocity: 100 }],
    },
  ],
  gridArrangement: {
    track_0_piano: { row: 0, column: 0 },
  },
  ...overrides,
});

const withServer = async (fn) => {
  const app = express();
  app.use('/api/process-videos', processVideosRouter);

  const server = await new Promise((resolve) => {
    const s = app.listen(0, () => resolve(s));
  });

  const { port } = server.address();
  const baseUrl = `http://127.0.0.1:${port}`;

  try {
    await fn(baseUrl);
  } finally {
    await new Promise((resolve, reject) => {
      server.close((err) => (err ? reject(err) : resolve()));
    });
  }
};

const postForm = async (baseUrl, form) => {
  const response = await fetch(`${baseUrl}/api/process-videos`, {
    method: 'POST',
    body: form,
  });

  let data = null;
  try {
    data = await response.json();
  } catch {
    data = { error: 'non-json-response' };
  }

  return { response, data };
};

const testInvalidMidiJson = async (baseUrl) => {
  const form = new FormData();
  form.append(
    'midiData',
    new Blob(['{not-json'], { type: 'application/json' }),
    'midi.json',
  );

  const { response, data } = await postForm(baseUrl, form);
  assert.strictEqual(
    response.status,
    400,
    'Invalid midiData JSON should return 400',
  );
  assert.strictEqual(data.error, 'Invalid composition input');
  assert.ok(
    String(data.details || '').includes('Invalid midiData JSON payload'),
  );
};

const testMissingGridArrangement = async (baseUrl) => {
  const form = new FormData();
  const midiPayload = makeMidiPayload({ gridArrangement: {} });
  form.append(
    'midiData',
    new Blob([JSON.stringify(midiPayload)], { type: 'application/json' }),
    'midi.json',
  );
  form.append(
    'videos',
    new Blob(['fake-video'], { type: 'video/mp4' }),
    'piano.mp4',
  );

  const { response, data } = await postForm(baseUrl, form);
  assert.strictEqual(
    response.status,
    400,
    'Missing grid arrangement should return 400',
  );
  assert.strictEqual(data.error, 'Invalid composition input');
  assert.ok(String(data.details || '').includes('Grid arrangement is empty'));
};

const testMissingVideos = async (baseUrl) => {
  const form = new FormData();
  form.append(
    'midiData',
    new Blob([JSON.stringify(makeMidiPayload())], { type: 'application/json' }),
    'midi.json',
  );

  const { response, data } = await postForm(baseUrl, form);
  assert.strictEqual(response.status, 400, 'Missing videos should return 400');
  assert.strictEqual(data.error, 'Invalid composition input');
  assert.ok(
    String(data.details || '').includes('At least one video file is required'),
  );
};

const testInvalidGridPosition = async (baseUrl) => {
  const form = new FormData();
  const midiPayload = makeMidiPayload({
    gridArrangement: {
      track_0_piano: { row: -1, column: 0 },
    },
  });

  form.append(
    'midiData',
    new Blob([JSON.stringify(midiPayload)], { type: 'application/json' }),
    'midi.json',
  );
  form.append(
    'videos',
    new Blob(['fake-video'], { type: 'video/mp4' }),
    'piano.mp4',
  );

  const { response, data } = await postForm(baseUrl, form);
  assert.strictEqual(
    response.status,
    400,
    'Invalid grid position should return 400',
  );
  assert.strictEqual(data.error, 'Invalid composition input');
  assert.ok(
    String(data.details || '').includes('invalid row/column positions'),
  );
};

const run = async () => {
  await withServer(async (baseUrl) => {
    await testInvalidMidiJson(baseUrl);
    await testMissingGridArrangement(baseUrl);
    await testMissingVideos(baseUrl);
    await testInvalidGridPosition(baseUrl);
  });

  console.log('PASS test_processVideos_validation');
};

run().catch((err) => {
  console.error('FAIL test_processVideos_validation');
  console.error(err);
  process.exit(1);
});
