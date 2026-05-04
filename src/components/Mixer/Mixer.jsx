import { useState } from 'react';
import PropTypes from 'prop-types';
import ExpandedMixerModal from './ExpandedMixerModal';
import MixerChannel from './MixerChannel';
import './Mixer.css';

const Mixer = ({
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
}) => {
  const [showExpanded, setShowExpanded] = useState(false);

  const handleMute = (key) => {
    if (onMuteChange) onMuteChange(key, !muteStates[key]);
  };

  const handleSolo = (key) => {
    if (onSoloChange) onSoloChange(key);
  };

  return (
    <>
      <div className='mixer-container'>
        <div className='mixer-header'>
          <div className='mixer-header__titleGroup'>
            <h3>Master Mixer</h3>
            <span className='channel-count'>{instruments.length} channels</span>
          </div>
          <div className='mixer-info'>
            {instruments.length > 3 && (
              <button
                className='mixer-expand-btn'
                onClick={() => setShowExpanded(true)}
                title='Expand mixer to full view'
                aria-label='Expand mixer'
              >
                ⛶ Expand
              </button>
            )}
          </div>
        </div>

        <div className='mixer-wrapper'>
          <div className='mixer-channels'>
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
                  onVolumeChange={(nextVolume) =>
                    onVolumeChange(key, nextVolume)
                  }
                  onMute={() => handleMute(key)}
                  onSolo={() => handleSolo(key)}
                  variant='compact'
                />
              );
            })}
          </div>
        </div>
        {instruments.length > 8 && (
          <span className='scroll-hint'>Scroll sideways for more channels</span>
        )}
      </div>

      {/* Expanded mixer modal */}
      {showExpanded && (
        <ExpandedMixerModal
          instruments={instruments}
          volumes={volumes}
          onVolumeChange={onVolumeChange}
          muteStates={muteStates}
          soloTrack={soloTrack}
          onMuteChange={onMuteChange}
          onSoloChange={onSoloChange}
          activeLevels={activeLevels}
          onTogglePreview={onTogglePreview}
          isPreviewPlaying={isPreviewPlaying}
          previewElapsed={previewElapsed}
          previewDuration={previewDuration}
          onClose={() => setShowExpanded(false)}
        />
      )}
    </>
  );
};

Mixer.propTypes = {
  instruments: PropTypes.array.isRequired,
  volumes: PropTypes.object.isRequired,
  onVolumeChange: PropTypes.func.isRequired,
  activeLevels: PropTypes.object,
  onTogglePreview: PropTypes.func,
  isPreviewPlaying: PropTypes.bool,
  previewElapsed: PropTypes.number,
  previewDuration: PropTypes.number,
};

export default Mixer;
