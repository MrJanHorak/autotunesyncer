import { useState } from 'react';
import PropTypes from 'prop-types';
import './MidiInfoDisplay.css';

const MidiInfoDisplay = ({ midiData }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!midiData) return null;

  return (
    <div className='midi-info-container'>
      <div
        className='midi-info-header'
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className='midi-info-summary'>
          <span className='midi-info-filename'>
            {midiData.summary?.name || 'Unknown'}
          </span>
          <span className='midi-info-stats'>
            {midiData.tracks?.length || 0} tracks ·{' '}
            {Math.round(midiData.duration || 0)}s ·{' '}
            {Math.round(midiData.header?.tempo || 120)} BPM
          </span>
        </div>
        <span className={`arrow ${isExpanded ? 'expanded' : ''}`}>▼</span>
      </div>
      <div className={`midi-info-content ${isExpanded ? 'expanded' : ''}`}>
        <div className='midi-info-grid'>
          <div className='midi-info-item'>
            <div className='info-label'>Format</div>
            <div className='info-value'>
              {midiData.header?.format || 'Unknown'}
            </div>
          </div>
          <div className='midi-info-item'>
            <div className='info-label'>Time Signature</div>
            <div className='info-value'>
              {midiData.header?.timeSignature || '4/4'}
            </div>
          </div>
          <div className='midi-info-item'>
            <div className='info-label'>Key</div>
            <div className='info-value'>
              {midiData.header?.key || 'Unknown'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

MidiInfoDisplay.propTypes = {
  midiData: PropTypes.shape({
    tracks: PropTypes.array,
    duration: PropTypes.number,
    header: PropTypes.shape({
      format: PropTypes.number,
      timeSignature: PropTypes.string,
      tempo: PropTypes.number,
      key: PropTypes.string,
    }),
    summary: PropTypes.shape({
      name: PropTypes.string,
      totalTracks: PropTypes.number,
      totalNotes: PropTypes.number,
    }),
  }),
};

export default MidiInfoDisplay;
