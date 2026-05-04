import { useEffect } from 'react';
import PropTypes from 'prop-types';
import MixerChannel from './MixerChannel';
import './ExpandedMixerModal.css';

const formatClock = (seconds) => {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const mins = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

const ExpandedMixerModal = ({
  instruments,
  volumes,
  onVolumeChange,
  muteStates = {},
  soloTrack = null,
  onMuteChange,
  onSoloChange,
  activeLevels = {},
  onTogglePreview,
  isPreviewPlaying = false,
  previewElapsed = 0,
  previewDuration = 0,
  onClose,
}) => {
  const handleMute = (key) => {
    if (onMuteChange) onMuteChange(key, !muteStates[key]);
  };

  const handleSolo = (key) => {
    if (onSoloChange) onSoloChange(key);
  };

  const handleBackdropClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  // Keyboard shortcut: Esc to close
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  return (
    <div className='soundboard-backdrop' onClick={handleBackdropClick}>
      <div className='soundboard-console'>
        <div className='soundboard-header'>
          <h2 className='soundboard-title'>🎛 Master Soundboard</h2>
          <p className='soundboard-subtitle'>{instruments.length} Channels</p>
          <button
            className={[
              'soundboard-transport',
              isPreviewPlaying ? 'is-playing' : '',
            ]
              .filter(Boolean)
              .join(' ')}
            onClick={onTogglePreview}
            title={
              isPreviewPlaying
                ? 'Stop preview playback'
                : 'Start preview playback'
            }
            aria-label={
              isPreviewPlaying
                ? 'Stop preview playback'
                : 'Start preview playback'
            }
          >
            <span className='soundboard-transport__left'>
              <span
                className={[
                  'soundboard-transport__status',
                  isPreviewPlaying ? 'is-active' : '',
                ]
                  .filter(Boolean)
                  .join(' ')}
                aria-hidden='true'
              />
              <span className='soundboard-transport__label'>
                {isPreviewPlaying ? 'Stop Preview' : 'Play Preview'}
              </span>
            </span>
            <span className='soundboard-transport__time'>
              {formatClock(previewElapsed)} / {formatClock(previewDuration)}
            </span>
          </button>
          <button
            className='soundboard-close'
            onClick={onClose}
            aria-label='Close soundboard'
            title='Close (Esc)'
          >
            ✕
          </button>
        </div>

        <div className='soundboard-channels'>
          {instruments.map((inst, index) => {
            const key = inst.isDrum
              ? `drum_${inst.group.toLowerCase().replace(/\s+/g, '_')}`
              : inst.name.toLowerCase().replace(/\s+/g, '_');

            const volume = volumes[key] || 0;
            const isMuted = muteStates[key];
            const isSolo = soloTrack === key;

            return (
              <MixerChannel
                key={`${key}-${index}`}
                name={inst.isDrum ? inst.group : inst.name}
                volume={volume}
                levelDb={activeLevels[key]}
                isMuted={Boolean(isMuted)}
                isSolo={isSolo}
                onVolumeChange={(nextVolume) => onVolumeChange(key, nextVolume)}
                onMute={() => handleMute(key)}
                onSolo={() => handleSolo(key)}
                variant='expanded'
              />
            );
          })}
        </div>
      </div>
    </div>
  );
};

ExpandedMixerModal.propTypes = {
  instruments: PropTypes.array.isRequired,
  volumes: PropTypes.object.isRequired,
  onVolumeChange: PropTypes.func.isRequired,
  muteStates: PropTypes.object,
  soloTrack: PropTypes.string,
  onMuteChange: PropTypes.func,
  onSoloChange: PropTypes.func,
  activeLevels: PropTypes.object,
  onTogglePreview: PropTypes.func,
  isPreviewPlaying: PropTypes.bool,
  previewElapsed: PropTypes.number,
  previewDuration: PropTypes.number,
  onClose: PropTypes.func.isRequired,
};

export default ExpandedMixerModal;
