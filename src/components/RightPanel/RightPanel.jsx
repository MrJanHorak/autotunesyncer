import { useState, useMemo, useEffect } from 'react';
import PropTypes from 'prop-types';
import Mixer from '../Mixer/Mixer';
import PreviewPlayer from '../PreviewPlayer/PreviewPlayer';
import CompositionStylePanel from '../CompositionStylePanel/CompositionStylePanel';
import './RightPanel.css';

const TABS = [
  { id: 'style', icon: '🎨', label: 'Style' },
  { id: 'mix', icon: '🎚', label: 'Mix' },
];

export default function RightPanel({
  isOpen,
  onToggle,
  // Style tab
  compositionStyle,
  onStyleChange,
  backgroundAsset,
  onBackgroundUpload,
  onBackgroundRemove,
  // Mix tab
  instruments,
  volumes,
  muteStates,
  soloTrack,
  onVolumeChange,
  onMuteChange,
  onSoloChange,
  activeLevels,
  midiData,
  videoFiles,
  onMeterUpdate,
  onPlayStateChange,
  isPreviewPlaying = false,
}) {
  const [activeTab, setActiveTab] = useState('style');
  const [previewElapsed, setPreviewElapsed] = useState(0);

  const { autoTransitionIntervalSeconds, autoTransitionReason } =
    useMemo(() => {
      const tracks = Array.isArray(midiData?.tracks) ? midiData.tracks : [];
      const totalNotes = tracks.reduce(
        (acc, track) =>
          acc + (Array.isArray(track?.notes) ? track.notes.length : 0),
        0,
      );

      let maxTime = 0;
      tracks.forEach((track) => {
        track?.notes?.forEach((note) => {
          const start = Number(
            note?.time ?? note?.start ?? note?.startTime ?? note?.ticks ?? 0,
          );
          const duration = Number(note?.duration ?? 0);
          const end = Number(note?.end ?? note?.endTime ?? start + duration);
          maxTime = Math.max(maxTime, Number.isFinite(end) ? end : 0);
        });
      });

      const safeDuration = Math.max(1, maxTime);
      const noteDensity = totalNotes / safeDuration;

      if (noteDensity >= 12) {
        return {
          autoTransitionIntervalSeconds: 2.5,
          autoTransitionReason: 'Very dense arrangement detected',
        };
      }
      if (noteDensity >= 8) {
        return {
          autoTransitionIntervalSeconds: 3.5,
          autoTransitionReason: 'Dense arrangement detected',
        };
      }
      if (noteDensity >= 4) {
        return {
          autoTransitionIntervalSeconds: 5,
          autoTransitionReason: 'Balanced arrangement detected',
        };
      }
      if (noteDensity >= 2) {
        return {
          autoTransitionIntervalSeconds: 6.5,
          autoTransitionReason: 'Light arrangement detected',
        };
      }
      return {
        autoTransitionIntervalSeconds: 8,
        autoTransitionReason: 'Sparse arrangement detected',
      };
    }, [midiData]);

  useEffect(() => {
    if (!compositionStyle || typeof onStyleChange !== 'function') {
      return;
    }

    const nextCadence = Number(autoTransitionIntervalSeconds || 8);
    const currentCadence = Number(
      compositionStyle.transitionAutoCadenceSeconds,
    );
    const cadenceChanged =
      !Number.isFinite(currentCadence) ||
      Math.abs(currentCadence - nextCadence) > 0.001;

    const nextReason = autoTransitionReason || '';
    const currentReason = String(compositionStyle.transitionAutoReason || '');
    const reasonChanged = currentReason !== nextReason;

    if (!cadenceChanged && !reasonChanged) {
      return;
    }

    onStyleChange({
      ...compositionStyle,
      transitionAutoCadenceSeconds: nextCadence,
      transitionAutoReason: nextReason,
    });
  }, [
    autoTransitionIntervalSeconds,
    autoTransitionReason,
    compositionStyle,
    onStyleChange,
  ]);

  const handleTabClick = (tabId) => {
    if (!isOpen) {
      // clicking a tab icon while collapsed: expand and switch tab
      onToggle();
    }
    setActiveTab(tabId);
  };

  const handlePreviewToggleFromMixer = () => {
    const button = document.querySelector(
      '.right-panel__preview .preview-player__button',
    );
    if (button && !button.disabled) {
      button.click();
    }
  };

  return (
    <div
      className={`right-panel${isOpen ? '' : ' right-panel--collapsed'} editor-right${isOpen ? '' : ' editor-right--collapsed'}`}
    >
      {/* Tab strip */}
      <div className='right-panel__tabs'>
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`right-panel__tab${activeTab === tab.id && isOpen ? ' right-panel__tab--active' : ''}`}
            onClick={() => handleTabClick(tab.id)}
            title={tab.label}
          >
            <span className='right-panel__tab-icon'>{tab.icon}</span>
            {isOpen && (
              <span className='right-panel__tab-label'>{tab.label}</span>
            )}
          </button>
        ))}
        <div className='right-panel__tabs-spacer' />
        <button
          className='panel-toggle-btn'
          onClick={onToggle}
          title={isOpen ? 'Collapse panel' : 'Expand panel'}
        >
          {isOpen ? '▶' : '◀'}
        </button>
      </div>

      {/* Panel content — only visible when open */}
      {isOpen && (
        <div className='right-panel__body'>
          {activeTab === 'style' && (
            <div className='right-panel__style'>
              <CompositionStylePanel
                style={compositionStyle}
                onChange={onStyleChange}
                backgroundAsset={backgroundAsset}
                onBackgroundUpload={onBackgroundUpload}
                onBackgroundRemove={onBackgroundRemove}
                autoTransitionIntervalSeconds={autoTransitionIntervalSeconds}
                autoTransitionReason={autoTransitionReason}
              />
            </div>
          )}

          {activeTab === 'mix' && (
            <div className='right-panel__mix'>
              {instruments.length > 0 ? (
                <>
                  <Mixer
                    instruments={instruments}
                    volumes={volumes}
                    onVolumeChange={onVolumeChange}
                    muteStates={muteStates}
                    soloTrack={soloTrack}
                    onMuteChange={onMuteChange}
                    onSoloChange={onSoloChange}
                    activeLevels={activeLevels}
                    onTogglePreview={handlePreviewToggleFromMixer}
                    isPreviewPlaying={isPreviewPlaying}
                    previewElapsed={previewElapsed}
                    previewDuration={midiData?.duration || 0}
                  />
                  {midiData && (
                    <div className='right-panel__preview'>
                      <PreviewPlayer
                        midiData={midiData}
                        videoFiles={videoFiles}
                        volumes={volumes}
                        muteStates={muteStates}
                        soloTrack={soloTrack}
                        instruments={instruments}
                        onMeterUpdate={onMeterUpdate}
                        onPlayStateChange={onPlayStateChange}
                        onTransportTimeUpdate={setPreviewElapsed}
                      />
                    </div>
                  )}
                </>
              ) : (
                <p className='right-panel__empty'>
                  Load a MIDI to see mix controls
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

RightPanel.propTypes = {
  isOpen: PropTypes.bool.isRequired,
  onToggle: PropTypes.func.isRequired,
  compositionStyle: PropTypes.object,
  onStyleChange: PropTypes.func.isRequired,
  backgroundAsset: PropTypes.object,
  onBackgroundUpload: PropTypes.func,
  onBackgroundRemove: PropTypes.func,
  instruments: PropTypes.array.isRequired,
  volumes: PropTypes.object,
  muteStates: PropTypes.object,
  soloTrack: PropTypes.string,
  onVolumeChange: PropTypes.func.isRequired,
  onMuteChange: PropTypes.func.isRequired,
  onSoloChange: PropTypes.func.isRequired,
  activeLevels: PropTypes.object,
  midiData: PropTypes.object,
  videoFiles: PropTypes.object,
  onMeterUpdate: PropTypes.func.isRequired,
  onPlayStateChange: PropTypes.func,
  isPreviewPlaying: PropTypes.bool,
};
