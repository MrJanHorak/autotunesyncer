/* eslint-disable react/prop-types */
import SampleSoundButton from '../SampleSoundButton/SampleSoundButton';
import '../styles.css';

const ControlButtons = ({
  isRecording,
  hasRecordedVideo,
  onStartRecording,
  onStopRecording,
  onReRecord,
  instrument,
}) => (
  <>
    <button
      className='control-button'
      onClick={isRecording ? onStopRecording : onStartRecording}
    >
      {isRecording ? 'Stop Recording' : 'Start Recording'}
    </button>
    {!isRecording && (
      <SampleSoundButton instrument={instrument} className='control-button' />
    )}
    {hasRecordedVideo && !isRecording && (
      <button className='control-button' onClick={onReRecord}>
        Re-record
      </button>
    )}
  </>
);

export default ControlButtons;
