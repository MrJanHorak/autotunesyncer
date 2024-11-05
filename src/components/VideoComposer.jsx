/* eslint-disable no-unused-vars */
/* eslint-disable react/prop-types */

import { useState, useEffect } from 'react';
import axios from 'axios';

const VideoComposer = ({ videoFiles, midiData }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [composedVideoUrl, setComposedVideoUrl] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    console.log('VideoFiles received:', videoFiles);
  }, [videoFiles]);

  const startComposition = async () => {
    setIsProcessing(true);
    setProgress(0);
    setError(null);

    try {
      const formData = new FormData();

      // Convert MIDI data to JSON and append
      const midiJsonString = JSON.stringify(midiData.toJSON());
      const midiBlob = new Blob([midiJsonString], { type: 'application/json' });
      formData.append('midiData', midiBlob, 'midi.json');

      // Process and append each video file
      await Promise.all(Object.entries(videoFiles).map(async ([instrumentName, videoUrl]) => {
        try {
          // Fetch the video from the URL
          const response = await fetch(videoUrl);
          if (!response.ok) throw new Error(`Failed to fetch video for ${instrumentName}`);
          
          const videoBlob = await response.blob();
          formData.append(`videos[${instrumentName}]`, videoBlob, `${instrumentName}.webm`);
          console.log(`Added video for ${instrumentName}, size: ${videoBlob.size}`);
        } catch (error) {
          console.error(`Error processing video for ${instrumentName}:`, error);
          throw error;
        }
      }));

      // Log form data contents for debugging
      for (let pair of formData.entries()) {
        console.log('FormData entry:', pair[0], pair[1] instanceof Blob ? `Blob size: ${pair[1].size}` : pair[1]);
      }

      const response = await axios.post('http://localhost:3000/api/compose', formData, {
        responseType: 'blob',
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setProgress(percentCompleted);
          }
        },
      });

      const url = URL.createObjectURL(response.data);
      setComposedVideoUrl(url);
    } catch (error) {
      console.error('Composition failed:', error);
      setError(error.response?.data?.error || 'Composition failed');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="video-composer">
      <button
        onClick={startComposition}
        disabled={isProcessing}
        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
      >
        {isProcessing ? 'Processing...' : 'Start Composition'}
      </button>

      {isProcessing && (
        <div className="mt-4">
          <div className="w-full h-2 bg-gray-200 rounded">
            <div
              className="h-full bg-blue-500 rounded"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-sm text-gray-600 mt-1">{progress}% complete</p>
        </div>
      )}

      {error && (
        <div className="mt-4 p-4 bg-red-100 text-red-700 rounded">
          {error}
        </div>
      )}

      {composedVideoUrl && (
        <div className="mt-4">
          <video
            src={composedVideoUrl}
            controls
            className="w-full max-w-4xl"
          />
        </div>
      )}
    </div>
  );
};

export default VideoComposer;