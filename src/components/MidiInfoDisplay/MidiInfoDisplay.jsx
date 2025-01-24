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
        <h3>MIDI File Information</h3>
        <span className={`arrow ${isExpanded ? 'expanded' : ''}`}>▼</span>
      </div>
      <div className={`midi-info-content ${isExpanded ? 'expanded' : ''}`}>
        <div className='midi-info'>
          <p>File: {midiData.summary?.name || 'Unknown'}</p>
          <p>Total Tracks: {midiData.tracks?.length || 0}</p>
          <p>Duration: {Math.round(midiData.duration || 0)}s</p>
          <p>Format: {midiData.header?.format || 'Unknown'}</p>
          <p>Time Signature: {midiData.header?.timeSignature || '4/4'}</p>
          <p>Key: {midiData.header?.key || 'Unknown'}</p>
          <p>Tempo: {Math.round(midiData.header?.tempo || 120)} BPM</p>
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
