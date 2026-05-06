import { useState } from 'react';
import PropTypes from 'prop-types';
import VideoComposer from '../VideoComposer/VideoComposer';

const CompositionSection = ({
  videoFiles,
  midiData,
  instrumentTrackMap,
  gridArrangement,
  trackVolumes,
  muteStates,
  soloTrack,
  compositionStyle,
  clipStyles,
  renderPreset,
  projectName,
  onResetLayout,
}) => {
  const [composing, setComposing] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);

  const handleCompositionProgress = (currentProgress) => {
    setProgress(currentProgress);
  };

  const handleCompositionError = (err) => {
    setError(err.message);
    setComposing(false);
  };

  return (
    <div className='composition-section'>
      {error && <div className='composition-error'>{error}</div>}

      <VideoComposer
        videoFiles={videoFiles}
        midiData={midiData}
        instrumentTrackMap={instrumentTrackMap}
        gridArrangement={gridArrangement}
        trackVolumes={trackVolumes}
        muteStates={muteStates}
        soloTrack={soloTrack}
        compositionStyle={compositionStyle}
        clipStyles={clipStyles}
        renderPreset={renderPreset}
        projectName={projectName}
        onProgress={handleCompositionProgress}
        onError={handleCompositionError}
        onStart={() => setComposing(true)}
        onComplete={() => setComposing(false)}
        onResetLayout={onResetLayout}
      />
    </div>
  );
};

CompositionSection.propTypes = {
  videoFiles: PropTypes.object.isRequired,
  midiData: PropTypes.object.isRequired,
  instrumentTrackMap: PropTypes.object.isRequired,
  gridArrangement: PropTypes.object.isRequired,
  trackVolumes: PropTypes.object,
  muteStates: PropTypes.object,
  soloTrack: PropTypes.string,
  compositionStyle: PropTypes.object,
  clipStyles: PropTypes.object,
  renderPreset: PropTypes.string,
  projectName: PropTypes.string,
  onResetLayout: PropTypes.func,
};

export default CompositionSection;
