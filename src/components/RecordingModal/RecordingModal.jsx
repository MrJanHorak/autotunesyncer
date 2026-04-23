import { useState, useEffect } from 'react';
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

  // Close on Escape — only when not in recording view
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'Escape') safeClose();
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
