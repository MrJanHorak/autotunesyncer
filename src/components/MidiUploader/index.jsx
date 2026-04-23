import { useState } from 'react';
import PropTypes from 'prop-types';
import Dropzone from 'react-dropzone';

const MidiUploader = ({ onMidiProcessed, compact = false }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleMidiUpload = (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    onMidiProcessed(file);
    setLoading(false);
  };

  if (compact) {
    return (
      <Dropzone
        onDrop={handleMidiUpload}
        accept={{ 'audio/midi': ['.mid', '.midi'] }}
        disabled={loading}
      >
        {({ getRootProps, getInputProps }) => (
          <button
            {...getRootProps()}
            className='editor-topbar__btn'
            type='button'
            title='Load a MIDI file'
          >
            <input {...getInputProps()} />
            {loading ? '⏳' : '🎵'} {loading ? 'Loading…' : 'Load MIDI'}
          </button>
        )}
      </Dropzone>
    );
  }

  return (
    <div className="midi-uploader">
      <Dropzone 
        onDrop={handleMidiUpload} 
        accept={{ 'audio/midi': ['.mid', '.midi'] }}
        disabled={loading}
      >
        {({ getRootProps, getInputProps }) => (
          <div {...getRootProps()} className={`dropzone ${loading ? 'loading' : ''}`}>
            <input {...getInputProps()} />
            {loading ? (
              <p>Processing MIDI file...</p>
            ) : (
              <p>Drag & drop a MIDI file here, or click to select one</p>
            )}
          </div>
        )}
      </Dropzone>
      {error && <div className="error-message">{error}</div>}
    </div>
  );
};

MidiUploader.propTypes = {
  onMidiProcessed: PropTypes.func.isRequired,
  compact: PropTypes.bool,
};

export default MidiUploader;