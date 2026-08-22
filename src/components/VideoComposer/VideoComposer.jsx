/* eslint-disable react/prop-types */
import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
  startCompositionJob,
  trackCompositionJob,
} from '../../../services/videoServices.js';
import {
  fetchProjectRenderFile,
  fetchProjectRenderStatus,
} from '../../services/apiService.js';
import {
  hasGridArrangement,
  toLegacyGridArrangement,
} from '../../../shared/gridLayout.js';
import { DEFAULT_RENDER_PRESET } from '../../../shared/renderPresets.js';
import './VideoComposer.css';

const getLivePreviewStageDimensions = () => {
  if (typeof document === 'undefined') {
    return null;
  }

  const stageElement = document.querySelector('.grid-preview-stage');
  if (!stageElement) {
    return null;
  }

  const rect = stageElement.getBoundingClientRect();
  const width = Math.round(rect.width || 0);
  const height = Math.round(rect.height || 0);

  if (width <= 0 || height <= 0) {
    return null;
  }

  return { width, height };
};

const DEFAULT_EXPORT_BILLING = {
  checked: false,
  requiredPlan: 'creator',
  canExportProject: true,
};

const normalizeExportBilling = (billing) => ({
  checked: true,
  requiredPlan: billing?.requiredPlan || 'creator',
  canExportProject: billing?.canExportProject !== false,
});

const VideoComposer = ({
  videoFiles,
  midiData,
  instrumentTrackMap,
  gridArrangement,
  trackVolumes,
  muteStates = {},
  soloTrack = null,
  compositionStyle = null,
  clipStyles = null,
  renderPreset = DEFAULT_RENDER_PRESET,
  projectId = null,
  onProgress = null,
  onError = null,
  onStart = null,
  onComplete = null,
  onResetLayout = null,
  onOpenBillingSettings = null,
}) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingMode, setProcessingMode] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [renderProgress, setRenderProgress] = useState(0);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [composedVideoUrl, setComposedVideoUrl] = useState(null);
  const [error, setError] = useState(null);
  const [exportBilling, setExportBilling] = useState(DEFAULT_EXPORT_BILLING);
  const timerRef = useRef(null);
  const abortRef = useRef(null);
  const onProgressRef = useRef(onProgress);
  const onErrorRef = useRef(onError);
  const onStartRef = useRef(onStart);
  const onCompleteRef = useRef(onComplete);
  // Track the current blob URL in a ref so cleanup is always unmount-only (not
  // triggered by every state change) and the old URL is revoked before replacement.
  const composedVideoUrlRef = useRef(null);
  // Preserved across finally so Retry knows which mode was last used.
  const lastModeRef = useRef(false);
  const MIN_NOTE_DURATION_SECONDS = 1 / 120;
  const normalizedGridArrangement = useMemo(
    () => toLegacyGridArrangement(gridArrangement),
    [gridArrangement],
  );

  const clearComposedVideo = useCallback(() => {
    if (composedVideoUrlRef.current) {
      URL.revokeObjectURL(composedVideoUrlRef.current);
      composedVideoUrlRef.current = null;
    }
    setComposedVideoUrl(null);
  }, []);

  const applyComposedBlob = useCallback((blob) => {
    if (!blob) return;
    if (composedVideoUrlRef.current) {
      URL.revokeObjectURL(composedVideoUrlRef.current);
    }
    const url = URL.createObjectURL(blob);
    composedVideoUrlRef.current = url;
    setComposedVideoUrl(url);
  }, []);

  const startElapsedTimer = useCallback(() => {
    clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      setElapsedSeconds((seconds) => seconds + 1);
    }, 1000);
  }, []);

  const resetTrackedRenderState = useCallback(() => {
    clearInterval(timerRef.current);
    timerRef.current = null;
    abortRef.current?.abort();
    abortRef.current = null;
    setIsProcessing(false);
    setProcessingMode(null);
    setUploadProgress(0);
    setRenderProgress(0);
    setElapsedSeconds(0);
    setError(null);
  }, []);

  const getNormalizedNoteTime = useCallback((note) => {
    const candidates = [note?.time, note?.start, note?.startTime];
    for (const candidate of candidates) {
      const value = Number(candidate);
      if (Number.isFinite(value)) return value;
    }
    return NaN;
  }, []);

  const getNormalizedNoteDuration = useCallback(
    (note) => {
      const direct = Number(note?.duration);
      if (Number.isFinite(direct)) {
        return direct > 0 ? direct : MIN_NOTE_DURATION_SECONDS;
      }

      const start = getNormalizedNoteTime(note);
      const endCandidates = [note?.end, note?.endTime];
      for (const candidate of endCandidates) {
        const end = Number(candidate);
        if (Number.isFinite(start) && Number.isFinite(end)) {
          const computed = end - start;
          return computed > 0 ? computed : MIN_NOTE_DURATION_SECONDS;
        }
      }

      return NaN;
    },
    [MIN_NOTE_DURATION_SECONDS, getNormalizedNoteTime],
  );

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
            !Number.isFinite(getNormalizedNoteTime(note)) ||
            !Number.isFinite(getNormalizedNoteDuration(note)) ||
            getNormalizedNoteDuration(note) <= 0,
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

    if (!hasGridArrangement(normalizedGridArrangement)) {
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
  }, [
    midiData,
    instrumentTrackMap,
    normalizedGridArrangement,
    videoFiles,
    getNormalizedNoteDuration,
    getNormalizedNoteTime,
  ]);

  const canCompose = validationErrors.length === 0;
  const exportLocked =
    exportBilling.checked && exportBilling.canExportProject === false;
  const canStartComposition = canCompose && !exportLocked;

  useEffect(() => {
    onProgressRef.current = onProgress;
    onErrorRef.current = onError;
    onStartRef.current = onStart;
    onCompleteRef.current = onComplete;
  }, [onComplete, onError, onProgress, onStart]);

  useEffect(() => {
    let cancelled = false;
    const reconnectController = new AbortController();
    let reconnectingJob = false;

    resetTrackedRenderState();
    clearComposedVideo();
    setExportBilling(DEFAULT_EXPORT_BILLING);

    if (!projectId) {
      return () => {
        cancelled = true;
        reconnectController.abort();
      };
    }

    const hydrateProjectRender = async () => {
      try {
        const { render, billing } = await fetchProjectRenderStatus(projectId);
        if (cancelled) return;

        setExportBilling(normalizeExportBilling(billing));

        if (!render) return;

        const hasActiveRender =
          ['queued', 'processing'].includes(render.status) && !!render.jobId;

        if (render.hasOutput && !hasActiveRender) {
          try {
            const savedBlob = await fetchProjectRenderFile(projectId);
            if (!cancelled) {
              applyComposedBlob(savedBlob);
            }
          } catch (fileErr) {
            console.warn(
              '[VideoComposer] Failed to load saved render file:',
              fileErr,
            );
          }
        }

        if (!hasActiveRender) {
          if (render.status === 'failed' && render.error && !cancelled) {
            setError(render.error);
          }
          return;
        }

        reconnectingJob = true;
        lastModeRef.current = false;
        onStartRef.current?.();
        setIsProcessing(true);
        setProcessingMode('full');
        setUploadProgress(100);
        setRenderProgress(Number(render.progress) || 0);
        setError(null);

        if (render.startedAt) {
          const startedAtMs = new Date(render.startedAt).getTime();
          if (Number.isFinite(startedAtMs)) {
            setElapsedSeconds(
              Math.max(0, Math.floor((Date.now() - startedAtMs) / 1000)),
            );
          }
        }

        abortRef.current = reconnectController;
        startElapsedTimer();

        const trackedBlob = await trackCompositionJob(
          render.jobId,
          (pct) => {
            if (cancelled) return;
            setUploadProgress(100);
            setRenderProgress(pct);
            onProgressRef.current?.(pct);
          },
          reconnectController.signal,
        );

        if (cancelled) return;
        applyComposedBlob(trackedBlob);
        setUploadProgress(100);
        setRenderProgress(100);
        setError(null);
        onCompleteRef.current?.();
      } catch (err) {
        if (cancelled || err?.name === 'AbortError') return;
        console.warn('[VideoComposer] Failed to hydrate render state:', err);
        setError(err.message || 'Failed to load saved render state');
      } finally {
        if (!cancelled && reconnectingJob) {
          clearInterval(timerRef.current);
          timerRef.current = null;
          if (abortRef.current === reconnectController) {
            abortRef.current = null;
          }
          setIsProcessing(false);
          setProcessingMode(null);
        }
      }
    };

    hydrateProjectRender();

    return () => {
      cancelled = true;
      clearInterval(timerRef.current);
      timerRef.current = null;
      if (abortRef.current === reconnectController) {
        abortRef.current = null;
      }
      reconnectController.abort();
    };
  }, [
    applyComposedBlob,
    clearComposedVideo,
    projectId,
    resetTrackedRenderState,
    startElapsedTimer,
  ]);

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
    if (isProcessing) return;
    if (exportLocked) {
      setError('Video rendering requires the Creator plan or higher.');
      return;
    }
    if (!canCompose) {
      setError(validationErrors.join(' '));
      return;
    }

    // Create AbortController immediately so Cancel works during upload too.
    const abort = new AbortController();
    abortRef.current = abort;
    lastModeRef.current = isPreview;
    clearComposedVideo();

    onStartRef.current?.();
    setIsProcessing(true);
    setProcessingMode(isPreview ? 'preview' : 'full');
    setUploadProgress(0);
    setRenderProgress(0);
    setElapsedSeconds(0);
    setError(null);

    // Start elapsed-time counter
    startElapsedTimer();

    try {
      const formData = new FormData();

      const normalizedTracks = (midiData.tracks || []).map((track) => ({
        ...track,
        notes: (track?.notes || []).map((note) => ({
          ...note,
          time: getNormalizedNoteTime(note),
          duration: getNormalizedNoteDuration(note),
        })),
      }));

      // Add MIDI data — substitute effective volumes so mute/solo is baked in
      const midiPayload = {
        ...midiData,
        tracks: normalizedTracks,
        gridArrangement: gridArrangement || normalizedGridArrangement,
        trackVolumes: effectiveVolumes,
        compositionStyle: compositionStyle || {},
        clipStyles: clipStyles || {},
        renderPreset,
        previewStageDimensions: getLivePreviewStageDimensions(),
      };
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

      }

      // ── Async job: upload → get jobId → poll → download ──────────────────
      const jobId = await startCompositionJob(formData, {
        onUploadProgress: (pct) => setUploadProgress(pct),
        signal: abort.signal,
      });

      const blob = await trackCompositionJob(
        jobId,
        (pct) => {
          setRenderProgress(pct);
          onProgressRef.current?.(pct);
        },
        abort.signal,
      );

      applyComposedBlob(blob);
      onCompleteRef.current?.();
    } catch (err) {
      // AbortError = user hit Cancel; don't surface as an error
      if (err?.name === 'AbortError') return;
      console.error('Composition failed:', err);
      const normalizedErr = err instanceof Error ? err : new Error(String(err));
      if (/Creator plan or higher/i.test(normalizedErr.message)) {
        setExportBilling({
          checked: true,
          requiredPlan: 'creator',
          canExportProject: false,
        });
      }
      setError(normalizedErr.message);
      onErrorRef.current?.(normalizedErr);
    } finally {
      clearInterval(timerRef.current);
      timerRef.current = null;
      setIsProcessing(false);
      setProcessingMode(null);
    }
  };

  // Cleanup timer, SSE stream, and URL on unmount only.
  // Using [] dep array (not [composedVideoUrl]) prevents abortRef from
  // firing mid-composition whenever the URL state changes.
  useEffect(() => {
    return () => {
      clearInterval(timerRef.current);
      abortRef.current?.abort();
      if (composedVideoUrlRef.current)
        URL.revokeObjectURL(composedVideoUrlRef.current);
    };
  }, []);

  return (
    <div className='video-composer'>
      <div className='composition-toolbar'>
        <div className='composition-actions'>
          <button
            onClick={() => startComposition(true)}
            disabled={isProcessing || !canStartComposition}
            className='composition-btn composition-btn--preview'
            title={
              exportLocked
                ? 'Creator plan required to render previews'
                : 'Generate fast preview at lower quality'
            }
          >
            <span className='composition-btn__icon'>⚡</span>
            <span className='composition-btn__text'>
              {isProcessing && processingMode === 'preview'
                ? 'Generating Preview…'
                : 'Generate Preview (Fast)'}
            </span>
          </button>
          <button
            onClick={() => startComposition(false)}
            disabled={isProcessing || !canStartComposition}
            className='composition-btn composition-btn--full'
            title={
              exportLocked
                ? 'Creator plan required to render full compositions'
                : 'Render full high-quality composition'
            }
          >
            <span className='composition-btn__icon'>✓</span>
            <span className='composition-btn__text'>
              {isProcessing && processingMode === 'full'
                ? 'Processing Full Video…'
                : 'Start Full Composition'}
            </span>
          </button>
          {isProcessing && (
            <button
              onClick={() => abortRef.current?.abort()}
              className='composition-btn composition-btn--cancel'
              aria-label='Cancel composition'
              title='Cancel current operation'
            >
              <span className='composition-btn__icon'>✕</span>
              <span className='composition-btn__text'>Cancel</span>
            </button>
          )}
          {onResetLayout && !isProcessing && (
            <button
              onClick={onResetLayout}
              className='composition-btn composition-btn--reset'
              title='Reset all clips to default grid layout'
            >
              <svg
                width='16'
                height='16'
                viewBox='0 0 24 24'
                fill='none'
                stroke='currentColor'
                strokeWidth='2'
                strokeLinecap='round'
                strokeLinejoin='round'
              >
                <path d='M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8' />
                <path d='M21 3v5h-5' />
                <path d='M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16' />
                <path d='M3 21v-5h5' />
              </svg>
              <span className='composition-btn__text'>Reset Layout</span>
            </button>
          )}
        </div>

        {isProcessing && (
          <div
            className='progress-container'
            aria-live='polite'
            aria-label='Composition progress'
          >
            {uploadProgress < 100 ? (
              <>
                <div className='progress-bar-wrapper'>
                  <div
                    className='progress-bar-fill'
                    style={{
                      width: `${uploadProgress}%`,
                      background:
                        processingMode === 'preview'
                          ? 'linear-gradient(90deg, #fbbf24 0%, #f59e0b 100%)'
                          : 'linear-gradient(90deg, #3b82f6 0%, #2563eb 100%)',
                    }}
                  />
                </div>
                <p className='progress-text'>
                  <span style={{ fontWeight: 600, color: '#3b82f6' }}>
                    📤 Uploading
                  </span>
                  {' — '}
                  {uploadProgress}%
                </p>
              </>
            ) : renderProgress > 0 ? (
              <>
                <div className='progress-bar-wrapper'>
                  <div
                    className='progress-bar-fill'
                    style={{
                      width: `${renderProgress}%`,
                      background:
                        processingMode === 'preview'
                          ? 'linear-gradient(90deg, #fbbf24 0%, #f59e0b 100%)'
                          : 'linear-gradient(90deg, #3b82f6 0%, #2563eb 100%)',
                    }}
                  />
                </div>
                <p className='progress-text'>
                  <span className='font-medium'>
                    {renderProgress <= 25
                      ? '🔬 Preprocessing'
                      : renderProgress <= 85
                        ? `${processingMode === 'preview' ? '⚡ Preview' : '🎬 Full'} Rendering`
                        : '✨ Finalizing'}
                  </span>
                  {' — '}
                  {renderProgress}%{' · '}
                  {elapsedSeconds}s elapsed
                  {elapsedSeconds > 2 &&
                    (() => {
                      const etaSec = Math.round(
                        (elapsedSeconds * (100 - renderProgress)) /
                          renderProgress,
                      );
                      return etaSec > 0 ? `, ~${etaSec}s remaining` : null;
                    })()}
                </p>
              </>
            ) : (
              <>
                <div className='progress-bar-wrapper'>
                  <div
                    className='progress-bar-fill progress-bar-indeterminate'
                    style={{
                      background:
                        processingMode === 'preview'
                          ? 'linear-gradient(90deg, #fbbf24 0%, #f59e0b 100%)'
                          : 'linear-gradient(90deg, #3b82f6 0%, #2563eb 100%)',
                    }}
                  />
                </div>
                <p className='progress-text'>
                  <span style={{ fontWeight: 600, color: '#6b7280' }}>
                    ⚙️ Queued
                  </span>
                  {' — '}
                  {elapsedSeconds}s elapsed
                </p>
              </>
            )}
          </div>
        )}

        {error && (
          <div className='composition-error' role='alert'>
            <span className='composition-error__text'>{error}</span>
            <button
              onClick={() => {
                setError(null);
                startComposition(lastModeRef.current);
              }}
              className='composition-error__retry'
            >
              ↺ Retry
            </button>
          </div>
        )}

        {exportLocked && !error && (
          <div className='composition-warning composition-warning--billing'>
            <span>
              Video rendering and downloads require the Creator plan or higher.
            </span>
            {onOpenBillingSettings && (
              <button
                type='button'
                onClick={onOpenBillingSettings}
                className='composition-warning__action'
              >
                Open Billing
              </button>
            )}
          </div>
        )}

        {!canCompose && !error && !exportLocked && (
          <div className='composition-warning'>
            {validationErrors.join(' ')}
          </div>
        )}
      </div>

      {composedVideoUrl && (
        <div className='mt-4'>
          <div style={{ maxWidth: '800px', margin: '0 auto' }}>
            <video
              key={composedVideoUrl}
              src={composedVideoUrl}
              controls
              style={{
                width: '100%',
                maxHeight: '65vh',
                objectFit: 'contain',
                background: '#000',
                borderRadius: '8px',
                display: 'block',
              }}
            />
            <div
              style={{
                display: 'flex',
                gap: '0.75rem',
                marginTop: '0.75rem',
                flexWrap: 'wrap',
              }}
            >
              <a
                href={composedVideoUrl}
                download='composition.mp4'
                style={{
                  padding: '0.5rem 1.1rem',
                  background: '#16a34a',
                  color: '#fff',
                  borderRadius: '8px',
                  fontWeight: 600,
                  fontSize: '0.875rem',
                  textDecoration: 'none',
                }}
              >
                ⬇ Download
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoComposer;
