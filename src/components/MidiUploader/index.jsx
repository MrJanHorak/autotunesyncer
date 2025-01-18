import { useState } from 'react';
import PropTypes from 'prop-types';
import Dropzone from 'react-dropzone';
import { Midi } from '@tonejs/midi';

const MidiUploader = ({ onMidiProcessed }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleMidiUpload = (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    
    const reader = new FileReader();
    
    reader.onload = async () => {
      try {
        const arrayBuffer = reader.result;
        const midiData = new Midi(arrayBuffer);
        if (typeof onMidiProcessed === 'function') {
          onMidiProcessed(midiData);
        } else {
          throw new Error('onMidiProcessed must be a function');
        }
      } catch (error) {
        console.error('Error parsing MIDI file:', error);
        setError(error.message);
      } finally {
        setLoading(false);
      }
    };
    
    reader.readAsArrayBuffer(file);
  };

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
  onMidiProcessed: PropTypes.func.isRequired
};

export default MidiUploader;