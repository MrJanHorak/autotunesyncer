/* eslint-disable no-unused-vars */
/* eslint-disable react/prop-types */

import { useState, useEffect } from 'react';
import axios from 'axios';
import { composeVideos } from '../../services/videoServices.js';

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

      // Add MIDI data
      const midiJsonString = JSON.stringify(midiData);
      const midiBlob = new Blob([midiJsonString], { type: 'application/json' });
      formData.append('midiData', midiBlob);

      // Process and append videos
      for (const [instrumentName, videoData] of Object.entries(videoFiles)) {
        console.log(`Processing ${instrumentName} video:`, videoData);
        
        if (!videoData) {
          console.error(`No video data for ${instrumentName}`);
          continue;
        }

        // Convert video URL to Blob if needed
        let videoBlob = videoData;
        if (!(videoData instanceof Blob)) {
          try {
            const response = await fetch(videoData);
            if (!response.ok) throw new Error(`Failed to fetch video for ${instrumentName}`);
            videoBlob = await response.blob();
          } catch (error) {
            console.error(`Error processing video for ${instrumentName}:`, error);
            continue;
          }
        }

        formData.append('videos', videoBlob, `${instrumentName}.mp4`);
        console.log(`Added video for ${instrumentName}, size: ${videoBlob.size}`);
      }

      // Log FormData contents for debugging
      for (const pair of formData.entries()) {
        console.log('FormData entry:', pair[0], pair[1]);
      }

      const response = await composeVideos(formData, {
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProgress(percentCompleted);
        },
        onDownloadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setProgress(percent);
          }
        },
      });

      if (response.data instanceof Blob) {
        // Check if the blob is an error message
        if (response.data.type.includes('application/json')) {
          const text = await response.data.text();
          const error = JSON.parse(text);
          throw new Error(error.error || 'Failed to process video');
        }
        
        const url = URL.createObjectURL(response.data);
        setComposedVideoUrl(url);
      } else {
        throw new Error('Invalid response format');
      }
    } catch (error) {
      console.error('Composition failed:', error);
      setError(error.message || 'Failed to compose video');
      
      if (error.code === 'ECONNABORTED') {
        setError('Video processing took too long. Try with fewer or shorter videos.');
      }
    } finally {
      setIsProcessing(false);
    }
  };

  // Cleanup URLs when component unmounts
  useEffect(() => {
    return () => {
      if (composedVideoUrl) {
        URL.revokeObjectURL(composedVideoUrl);
      }
    };
  }, [composedVideoUrl]);

  return (
    <div className="video-composer">
      <button
        onClick={startComposition}
        disabled={isProcessing || Object.keys(videoFiles).length === 0}
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