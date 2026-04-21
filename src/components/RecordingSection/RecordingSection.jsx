/* eslint-disable react/prop-types */
import VideoRecorder from '../VideoRecorder/VideoRecorder';
import './RecordingSection.css';

const toClipKey = (instrument) => {
  if (instrument.isDrum) {
    return `drum_${(instrument.group || '').toLowerCase().replace(/\s+/g, '_')}`;
  }
  return (instrument.name || '').toLowerCase().replace(/\s+/g, '_');
};

const RecordingSection = ({
  instruments,
  longestNotes,
  onRecordingComplete,
  onVideoReady,
  instrumentVideos,
  midiData,
}) => {
  return (
    <div className={'recording-section'}>
      {instruments.map((instrument, index) => {
        // Raw key used by longestNotes (matches useMidiProcessing storage format)
        const instrumentName = instrument.isDrum
          ? `drum_${instrument.group}`
          : instrument.name;
        // Normalized key used by instrumentVideos and clip persistence
        const clipKey = toClipKey(instrument);
        const minDuration = longestNotes[instrumentName] || 0;
        const recommendedDuration = Math.ceil(minDuration + 1);

        return (
          <div
            className='recording-container'
            key={index}
            style={{ marginBottom: '20px' }}
          >
            <h3 className='instrument-name'>
              {instrument.isDrum
                ? `Drum - ${
                    instrument.group.charAt(0).toUpperCase() +
                    instrument.group.slice(1)
                  }`
                : `${instrument.family} - ${instrument.name}`}
            </h3>
            <p className='recording-length'>
              Minimum recording duration: {recommendedDuration} seconds
            </p>
            <VideoRecorder
              onRecordingComplete={(blob) =>
                onRecordingComplete(blob, instrument)
              }
              // style={{ width: '300px', height: '200px' }}
              instrument={instrument}
              onVideoReady={onVideoReady}
              minDuration={recommendedDuration}
              currentVideo={instrumentVideos[clipKey]}
              midiData={midiData}
            />
          </div>
        );
      })}
    </div>
  );
};

export default RecordingSection;
