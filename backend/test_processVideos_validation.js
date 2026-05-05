import assert from 'assert';
import {
  getResolvedVideoLayout,
  validateComposeInputs,
} from './routes/processVideos.js';

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

const makeVideoFiles = () => [
  { fieldname: 'videos', originalname: 'piano.mp4' },
];

const testInvalidMidiPayload = () => {
  assert.strictEqual(
    validateComposeInputs(null, makeVideoFiles()),
    'Invalid MIDI payload',
  );
};

const testMissingGridArrangement = () => {
  const midiPayload = makeMidiPayload({ gridArrangement: {} });
  assert.strictEqual(
    validateComposeInputs(midiPayload, makeVideoFiles()),
    'Grid arrangement is empty',
  );
};

const testMissingVideos = () => {
  assert.strictEqual(
    validateComposeInputs(makeMidiPayload(), []),
    'At least one video file is required',
  );
};

const testV2GridArrangementAccepted = () => {
  const midiPayload = makeMidiPayload({
    gridArrangement: {
      version: 2,
      columns: 1,
      rows: 1,
      items: {
        track_0_piano: { x: 0, y: 0, w: 1, h: 1 },
      },
    },
  });
  assert.strictEqual(
    validateComposeInputs(midiPayload, makeVideoFiles()),
    null,
  );
};

const testInvalidGridPosition = () => {
  const midiPayload = makeMidiPayload({
    gridArrangement: {
      track_0_piano: { row: -1, column: 0 },
    },
  });
  assert.strictEqual(
    validateComposeInputs(midiPayload, makeVideoFiles()),
    'Grid arrangement contains invalid row/column positions',
  );
};

const testV2GridArrangementOutOfBounds = () => {
  const midiPayload = makeMidiPayload({
    gridArrangement: {
      version: 2,
      columns: 12,
      rows: 12,
      items: {
        track_0_piano: { x: 11, y: 11, w: 2, h: 2 },
      },
    },
  });
  assert.strictEqual(
    validateComposeInputs(midiPayload, makeVideoFiles()),
    'Grid arrangement contains tiles outside the available 12x12 layout space',
  );
};

const testInstrumentVideoLayoutResolvesNumericTrackKey = () => {
  const layout = getResolvedVideoLayout(
    'piano',
    {
      0: { row: 1, column: 2, w: 3, h: 2 },
    },
    {
      tracks: [
        {
          instrument: { name: 'Piano', family: 'piano' },
        },
      ],
      totalWidth: 1920,
      totalHeight: 1080,
      gridColumns: 12,
      gridRows: 12,
    },
  );

  assert.deepStrictEqual(
    {
      matchKey: layout.matchKey,
      row: layout.row,
      column: layout.column,
      spanW: layout.spanW,
      spanH: layout.spanH,
      width: layout.width,
      height: layout.height,
    },
    {
      matchKey: '0',
      row: 1,
      column: 2,
      spanW: 3,
      spanH: 2,
      width: 480,
      height: 180,
    },
  );
};

const run = async () => {
  testInvalidMidiPayload();
  testMissingGridArrangement();
  testMissingVideos();
  testV2GridArrangementAccepted();
  testInvalidGridPosition();
  testV2GridArrangementOutOfBounds();
  testInstrumentVideoLayoutResolvesNumericTrackKey();

  console.log('PASS test_processVideos_validation');
};

run().catch((err) => {
  console.error('FAIL test_processVideos_validation');
  console.error(err);
  process.exit(1);
});
