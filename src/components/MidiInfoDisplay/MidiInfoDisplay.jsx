import PropTypes from 'prop-types';

const MidiInfoDisplay = ({ midiData }) => {
  if (!midiData) return null;
  console.log('MidiData:', midiData);
  return (
    <div className="midi-info-container">
      <h3>MIDI File Information</h3>
      <div className="midi-info">
        <p>Total Tracks: {midiData.tracks?.length || 0}</p>
        <p>Duration: {Math.round(midiData.duration || 0)}s</p>
        <p>Format: {midiData.header?.format || 'Unknown'}</p>
        <p>Time Signature: {midiData.header?.timeSignature || '4/4'}</p>
        <p>Key: {midiData.header?.key || 'Unknown'}</p>
        <p>Tempo: {Math.round(midiData.header?.tempo || 120)} BPM</p>
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
      key: PropTypes.string
    })
  })
};

export default MidiInfoDisplay;