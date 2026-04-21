import { mkdir, copyFile, stat } from 'fs/promises';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function exists(path) {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

async function main() {
  const srcDist = join(
    __dirname,
    '..',
    'node_modules',
    'js-synthesizer',
    'dist',
  );
  const srcExternals = join(
    __dirname,
    '..',
    'node_modules',
    'js-synthesizer',
    'externals',
  );
  const dstDir = join(__dirname, '..', 'public', 'sf2');

  await mkdir(dstDir, { recursive: true });

  // Copy externals (libfluidsynth-*.js)
  const externals = [
    'libfluidsynth-2.4.6.js',
    'libfluidsynth-2.4.6-with-libsndfile.js',
    'libfluidsynth-2.3.0.js',
    'libfluidsynth-2.3.0-with-libsndfile.js',
  ];
  for (const f of externals) {
    const src = join(srcExternals, f);
    const dst = join(dstDir, f);
    const srcExists = await exists(src);
    if (!srcExists) continue;
    await copyFile(src, dst);
    console.log(`[prepare:sf2] Copied ${f} -> ${dst}`);
  }
  // Copy dist main in case needed by loader
  for (const f of ['js-synthesizer.js']) {
    const src = join(srcDist, f);
    const dst = join(dstDir, f);
    const srcExists = await exists(src);
    if (!srcExists) continue;
    await copyFile(src, dst);
    console.log(`[prepare:sf2] Copied ${f} -> ${dst}`);
  }

  console.log('[prepare:sf2] Done.');
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
