/* eslint-disable react/prop-types */
import SampleSoundButton from './SampleSoundButton';
import './styles.css';

const ControlButtons = ({ 
  isRecording, 
  hasRecordedVideo,
  onStartRecording, 
  onStopRecording, 
  onReRecord,
  instrument
}) => (
  <div className="controls-container">
    {isRecording ? (
      <button className="control-button" onClick={onStopRecording}>
        Stop Recording
      </button>
    ) : (
      <>
        <button className="control-button" onClick={onStartRecording}>
          Start Recording
        </button>
        {hasRecordedVideo && (
          <button className="control-button" onClick={onReRecord}>
            Re-record
          </button>
        )}
        <SampleSoundButton 
          instrument={instrument}
          className="control-button"
        />
      </>
    )}
  </div>
);

export default ControlButtons;
