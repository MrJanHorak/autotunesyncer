import { useState } from 'react';
import PropTypes from 'prop-types';
import VideoComposer from '../VideoComposer/VideoComposer';

const CompositionSection = ({ videoFiles, midiData, instrumentTrackMap, gridArrangement }) => {
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
    <div className="mt-4 bg-green-100 p-4 rounded">
      <h2 className="text-xl font-bold mb-2">Video Composition</h2>
      
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {composing && (
        <div className="mb-4">
          <div className="w-full bg-gray-200 rounded">
            <div 
              className="bg-blue-600 text-xs font-medium text-blue-100 text-center p-0.5 leading-none rounded" 
              style={{ width: `${progress}%` }}
            >
              {progress}%
            </div>
          </div>
        </div>
      )}

      <VideoComposer
        videoFiles={videoFiles}
        midiData={midiData}
        instrumentTrackMap={instrumentTrackMap}
        gridArrangement={gridArrangement}
        onProgress={handleCompositionProgress}
        onError={handleCompositionError}
        onStart={() => setComposing(true)}
        onComplete={() => setComposing(false)}
      />
    </div>
  );
};

CompositionSection.propTypes = {
  videoFiles: PropTypes.object.isRequired,
  midiData: PropTypes.object.isRequired,
  instrumentTrackMap: PropTypes.object.isRequired,
  gridArrangement: PropTypes.object.isRequired
};

export default CompositionSection;