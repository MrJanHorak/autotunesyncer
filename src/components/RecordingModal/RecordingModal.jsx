import { useState, useEffect, useLayoutEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import PropTypes from 'prop-types';
import VideoRecorder from '../VideoRecorder/VideoRecorder';
import './RecordingModal.css';

const toClipKey = (instrument) => {
  if (instrument.isDrum) {
    return `drum_${(instrument.group || '').toLowerCase().replace(/\s+/g, '_')}`;
  }
  return (instrument.name || '').toLowerCase().replace(/\s+/g, '_');
};

const displayName = (instrument) => {
  if (instrument.isDrum) {
    const g = instrument.group || '';
    return 'Drum – ' + g.charAt(0).toUpperCase() + g.slice(1);
  }
  return `${instrument.family || ''} – ${instrument.name || ''}`;
};

function RecordingModalContent({
  instrument,
  instrumentVideos,
  longestNotes,
  midiData,
  onRecordingComplete,
  onVideoReady,
  onClose,
}) {
  const [rerecording, setRerecording] = useState(false);
  const panelRef = useRef(null);

  const clipKey = toClipKey(instrument);
  const hasVideo = !!instrumentVideos?.[clipKey];
  const currentVideo = instrumentVideos?.[clipKey];
  const showRecorder = !hasVideo || rerecording;

  const instrKey = instrument.isDrum
    ? `drum_${instrument.group}`
    : instrument.name;
  const minSec = longestNotes?.[instrKey] || 0;
  const recSec = Math.ceil(minSec + 1);

  // Block modal close while VideoRecorder is visible (may be mid-recording).
  // User must explicitly hit X to confirm they want to leave.
  const safeClose = () => {
    if (showRecorder) {
      // Only close via the X button (not backdrop/Escape) while recorder is up
      return;
    }
    onClose();
  };

  // ── Focus management ──────────────────────────────────────────────────────
  // Save the element that had focus before the modal opened so we can restore it.
  useLayoutEffect(() => {
    const previousFocus = document.activeElement;

    // Focus the first tabbable element inside the panel
    const panel = panelRef.current;
    if (panel) {
      const tabbable = panel.querySelectorAll(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );
      tabbable[0]?.focus();
    }

    return () => {
      // Restore focus only if the opener element is still in the DOM
      if (previousFocus?.isConnected) previousFocus.focus();
    };
  }, []);

  // Close on Escape; trap Tab within the panel
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'Escape') {
        safeClose();
        return;
      }
      if (e.key === 'Tab') {
        const panel = panelRef.current;
        if (!panel) return;
        const tabbable = Array.from(
          panel.querySelectorAll(
            'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
          ),
        );
        if (!tabbable.length) return;
        const first = tabbable[0];
        const last = tabbable[tabbable.length - 1];
        if (e.shiftKey) {
          if (document.activeElement === first) {
            e.preventDefault();
            last.focus();
          }
        } else {
          if (document.activeElement === last) {
            e.preventDefault();
            first.focus();
          }
        }
      }
    };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showRecorder]);

  const handleRecordingComplete = (blob) => {
    onRecordingComplete(blob, instrument);
    setRerecording(false);
  };

  return (
    <div className='recording-modal__overlay' onClick={safeClose}>
      <div
        className='recording-modal__panel'
        ref={panelRef}
        onClick={(e) => e.stopPropagation()}
        role='dialog'
        aria-modal='true'
        aria-label={`Record: ${displayName(instrument)}`}
      >
        {/* Header */}
        <div className='recording-modal__header'>
          <span className='recording-modal__title'>
            🎙 {displayName(instrument)}
          </span>
          <span className='recording-modal__hint'>
            Min. {recSec}s clip
          </span>
          <button className='recording-modal__close' onClick={onClose} aria-label='Close'>✕</button>
        </div>

        {/* Body */}
        <div className='recording-modal__body'>
          {hasVideo && !rerecording ? (
            /* Existing clip preview */
            <div className='recording-modal__preview'>
              <video
                className='recording-modal__video'
                src={currentVideo}
                controls
                playsInline
              />
              <div className='recording-modal__preview-actions'>
                <button
                  className='recording-modal__action-btn'
                  onClick={() => setRerecording(true)}
                >
                  ⏺ Re-record
                </button>
              </div>
            </div>
          ) : (
            /* VideoRecorder */
            <VideoRecorder
              onRecordingComplete={handleRecordingComplete}
              instrument={instrument}
              onVideoReady={onVideoReady}
              minDuration={recSec}
              currentVideo={currentVideo}
              midiData={midiData}
            />
          )}

          {rerecording && (
            <button
              className='recording-modal__action-btn recording-modal__action-btn--ghost'
              onClick={() => setRerecording(false)}
            >
              ← Keep existing clip
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default function RecordingModal(props) {
  return createPortal(<RecordingModalContent {...props} />, document.body);
}

RecordingModal.propTypes = {
  instrument: PropTypes.object.isRequired,
  instrumentVideos: PropTypes.object,
  longestNotes: PropTypes.object,
  midiData: PropTypes.object,
  onRecordingComplete: PropTypes.func.isRequired,
  onVideoReady: PropTypes.func.isRequired,
  onClose: PropTypes.func.isRequired,
};
