/* eslint-disable react/prop-types */
import VideoRecorder from '../VideoRecorder/VideoRecorder';
import './RecordingSection.css';

const RecordingSection = ({
  instruments,
  longestNotes,
  onRecordingComplete,
  onVideoReady,
  instrumentVideos,
}) => {
  return (
    <div className={'recording-section'}>
      {instruments.map((instrument, index) => {
        const instrumentName = instrument.isDrum
          ? `drum_${instrument.group}`
          : instrument.name;
        const minDuration = longestNotes[instrumentName] || 0;
        const recommendedDuration = Math.ceil(minDuration + 1);

        return (
          <div key={index} style={{ marginBottom: '20px' }}>
            <h3>
              {instrument.isDrum
                ? `Drum - ${
                    instrument.group.charAt(0).toUpperCase() +
                    instrument.group.slice(1)
                  }`
                : `${instrument.family} - ${instrument.name}`}
            </h3>
            <p>Minimum recording duration: {recommendedDuration} seconds</p>
            <VideoRecorder
              onRecordingComplete={(blob) =>
                onRecordingComplete(blob, instrument)
              }
              style={{ width: '300px', height: '200px' }}
              instrument={instrument}
              onVideoReady={onVideoReady}
              minDuration={recommendedDuration}
              currentVideo={instrumentVideos[instrumentName]}
            />
          </div>
        );
      })}
    </div>
  );
};

export default RecordingSection;
