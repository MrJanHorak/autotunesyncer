/* eslint-disable react/prop-types */
import { useState, useEffect, useRef, useMemo } from 'react';
import { composeVideos } from '../../../services/videoServices.js';

const VideoComposer = ({
  videoFiles,
  midiData,
  instrumentTrackMap,
  gridArrangement,
  trackVolumes,
  muteStates = {},
  soloTrack = null,
}) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingMode, setProcessingMode] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [composedVideoUrl, setComposedVideoUrl] = useState(null);
  const [error, setError] = useState(null);
  const timerRef = useRef(null);

  const validationErrors = useMemo(() => {
    const errors = [];

    if (
      !midiData ||
      !Array.isArray(midiData.tracks) ||
      midiData.tracks.length === 0
    ) {
      errors.push('MIDI data is missing tracks.');
    } else {
      const hasInvalidNoteTiming = midiData.tracks.some((track) =>
        (track?.notes || []).some(
          (note) =>
            !Number.isFinite(Number(note?.time)) ||
            !Number.isFinite(Number(note?.duration)) ||
            Number(note.duration) <= 0,
        ),
      );

      if (hasInvalidNoteTiming) {
        errors.push('MIDI notes contain invalid time or duration values.');
      }
    }

    const mapSize =
      instrumentTrackMap instanceof Map
        ? instrumentTrackMap.size
        : Object.keys(instrumentTrackMap || {}).length;
    if (mapSize === 0) {
      errors.push(
        'Instrument-to-track mapping is missing. Re-parse the MIDI file.',
      );
    }

    if (!gridArrangement || Object.keys(gridArrangement).length === 0) {
      errors.push('Grid arrangement is missing.');
    }

    const entries = Object.entries(videoFiles || {});
    if (entries.length === 0) {
      errors.push('At least one video recording is required.');
    } else {
      const invalidVideos = entries.filter(([, value]) => {
        if (!value) return true;
        if (value instanceof Blob) return value.size === 0;
        return typeof value !== 'string' || value.length === 0;
      });
      if (invalidVideos.length > 0) {
        errors.push(
          `Invalid video input for: ${invalidVideos.map(([name]) => name).join(', ')}`,
        );
      }
    }

    return errors;
  }, [midiData, instrumentTrackMap, gridArrangement, videoFiles]);

  const canCompose = validationErrors.length === 0;

  useEffect(() => {
    console.log('VideoFiles received:', videoFiles);
  }, [videoFiles]);

  // Compute effective volumes applying mute/solo logic
  const effectiveVolumes = useMemo(() => {
    const result = { ...trackVolumes };
    const hasSolo = soloTrack !== null;
    for (const key of Object.keys(result)) {
      const isMuted = muteStates[key];
      const isSolo = key === soloTrack;
      if (isMuted || (hasSolo && !isSolo)) {
        result[key] = -Infinity; // silenced
      }
    }
    return result;
  }, [trackVolumes, muteStates, soloTrack]);

  const startComposition = async (isPreview = false) => {
    if (!canCompose) {
      setError(validationErrors.join(' '));
      return;
    }

    console.log('Grid arrangement:', gridArrangement);
    setIsProcessing(true);
    setProcessingMode(isPreview ? 'preview' : 'full');
    setUploadProgress(0);
    setElapsedSeconds(0);
    setError(null);
    if (composedVideoUrl) {
      URL.revokeObjectURL(composedVideoUrl);
      setComposedVideoUrl(null);
    }

    // Start elapsed-time counter
    timerRef.current = setInterval(() => {
      setElapsedSeconds((s) => s + 1);
    }, 1000);

    try {
      const formData = new FormData();

      // Add MIDI data — substitute effective volumes so mute/solo is baked in
      const midiPayload = {
        ...midiData,
        gridArrangement,
        trackVolumes: effectiveVolumes,
      };
      console.log('Midi data being sent:', midiPayload);
      const midiBlob = new Blob([JSON.stringify(midiPayload)], {
        type: 'application/json',
      });
      formData.append('midiData', midiBlob);

      if (isPreview) {
        formData.append('preview', 'true');
      }

      // Process and append videos
      for (const [instrumentName, videoData] of Object.entries(videoFiles)) {
        if (!videoData) {
          console.error(`No video data for ${instrumentName}`);
          continue;
        }

        let videoBlob = videoData;
        if (!(videoData instanceof Blob)) {
          try {
            const fetchRes = await fetch(videoData);
            if (!fetchRes.ok)
              throw new Error(`Failed to fetch video for ${instrumentName}`);
            videoBlob = await fetchRes.blob();
          } catch (fetchErr) {
            console.error(
              `Error processing video for ${instrumentName}:`,
              fetchErr,
            );
            continue;
          }
        }

        formData.append('videos', videoBlob, `${instrumentName}.mp4`);
        console.log(
          `Added video for ${instrumentName}, size: ${videoBlob.size}`,
        );
      }

      const response = await composeVideos(formData, {
        onUploadProgress: (pct) => setUploadProgress(pct),
      });

      if (response.data instanceof Blob) {
        if (response.data.type.includes('application/json')) {
          const text = await response.data.text();
          const errData = JSON.parse(text);
          throw new Error(
            errData.error || errData.details || 'Failed to process video',
          );
        }

        const url = URL.createObjectURL(response.data);
        setComposedVideoUrl(url);
      } else {
        throw new Error('Invalid response format from server');
      }
    } catch (err) {
      console.error('Composition failed:', err);
      setError(err.message || 'Failed to compose video');
    } finally {
      clearInterval(timerRef.current);
      timerRef.current = null;
      setIsProcessing(false);
      setProcessingMode(null);
    }
  };

  // Cleanup timer and URL on unmount
  useEffect(() => {
    return () => {
      clearInterval(timerRef.current);
      if (composedVideoUrl) URL.revokeObjectURL(composedVideoUrl);
    };
  }, [composedVideoUrl]);

  return (
    <div className='video-composer'>
      <div className='flex gap-4 mb-4'>
        <button
          onClick={() => startComposition(true)}
          disabled={isProcessing || !canCompose}
          className='px-4 py-2 bg-yellow-500 text-white rounded hover:bg-yellow-600 disabled:opacity-50 font-medium'
        >
          {isProcessing && processingMode === 'preview'
            ? 'Generating Preview...'
            : 'Generate Preview (Fast)'}
        </button>
        <button
          onClick={() => startComposition(false)}
          disabled={isProcessing || !canCompose}
          className='px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50 font-medium'
        >
          {isProcessing && processingMode === 'full'
            ? 'Processing Full Video...'
            : 'Start Full Composition'}
        </button>
      </div>

      {isProcessing && (
        <div className='mt-4'>
          {/* Upload phase: real progress bar */}
          {uploadProgress < 100 ? (
            <>
              <div className='w-full h-2 bg-gray-200 rounded overflow-hidden'>
                <div
                  className={`h-full rounded transition-all duration-300 ${
                    processingMode === 'preview'
                      ? 'bg-yellow-500'
                      : 'bg-blue-500'
                  }`}
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <p className='text-sm text-gray-600 mt-1'>
                Uploading videos… {uploadProgress}%
              </p>
            </>
          ) : (
            /* Processing phase: indeterminate animated bar */
            <>
              <div className='w-full h-2 bg-gray-200 rounded overflow-hidden'>
                <div
                  className={`h-full rounded ${
                    processingMode === 'preview'
                      ? 'bg-yellow-400'
                      : 'bg-blue-500'
                  }`}
                  style={{
                    width: '40%',
                    animation:
                      'indeterminate-progress 1.4s infinite ease-in-out',
                  }}
                />
              </div>
              <p className='text-sm text-gray-600 mt-1'>
                {processingMode === 'preview' ? '⚡ Preview' : '🎬 Full'}{' '}
                rendering… {elapsedSeconds}s elapsed
              </p>
            </>
          )}
        </div>
      )}

      {error && (
        <div className='mt-4 p-4 bg-red-100 text-red-700 rounded'>{error}</div>
      )}

      {!canCompose && !error && (
        <div className='mt-4 p-4 bg-amber-100 text-amber-800 rounded'>
          {validationErrors.join(' ')}
        </div>
      )}

      {composedVideoUrl && (
        <div className='mt-4'>
          <video src={composedVideoUrl} controls className='w-full max-w-4xl' />
          <a
            href={composedVideoUrl}
            download='composition.mp4'
            className='inline-block mt-2 px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 font-medium'
          >
            ⬇ Download Composition
          </a>
        </div>
      )}
    </div>
  );
};

export default VideoComposer;
