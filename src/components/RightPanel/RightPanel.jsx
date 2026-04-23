import { useState } from 'react';
import PropTypes from 'prop-types';
import Mixer from '../Mixer/Mixer';
import PreviewPlayer from '../PreviewPlayer/PreviewPlayer';
import CompositionStylePanel from '../CompositionStylePanel/CompositionStylePanel';
import './RightPanel.css';

const TABS = [
  { id: 'style', icon: '🎨', label: 'Style' },
  { id: 'mix',   icon: '🎚', label: 'Mix' },
];

export default function RightPanel({
  isOpen,
  onToggle,
  // Style tab
  compositionStyle,
  onStyleChange,
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
}) {
  const [activeTab, setActiveTab] = useState('style');

  const handleTabClick = (tabId) => {
    if (!isOpen) {
      // clicking a tab icon while collapsed: expand and switch tab
      onToggle();
    }
    setActiveTab(tabId);
  };

  return (
    <div className={`right-panel${isOpen ? '' : ' right-panel--collapsed'} editor-right${isOpen ? '' : ' editor-right--collapsed'}`}>
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
            {isOpen && <span className='right-panel__tab-label'>{tab.label}</span>}
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
            <CompositionStylePanel
              style={compositionStyle}
              onChange={onStyleChange}
            />
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
                      />
                    </div>
                  )}
                </>
              ) : (
                <p className='right-panel__empty'>Load a MIDI to see mix controls</p>
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
};
