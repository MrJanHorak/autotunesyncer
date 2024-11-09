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

      // Add MIDI data
      const midiJsonString = JSON.stringify(midiData);
      const midiBlob = new Blob([midiJsonString], { type: 'application/json' });
      formData.append('midiData', midiBlob, 'midi.json');

      // Process and append videos
      let totalSize = midiBlob.size;
      for (const [instrumentName, videoUrl] of Object.entries(videoFiles)) {
        try {
          console.log(`Processing video for ${instrumentName}`);
          
          let videoBlob;
          if (typeof videoUrl === 'string') {
            const response = await fetch(videoUrl);
            if (!response.ok) throw new Error(`Failed to fetch video for ${instrumentName}`);
            videoBlob = await response.blob();
          } else if (videoUrl instanceof Blob) {
            videoBlob = videoUrl;
          } else {
            console.error(`Invalid video data for ${instrumentName}:`, videoUrl);
            continue;
          }

          totalSize += videoBlob.size;
          if (totalSize > 500 * 1024 * 1024) { // 500MB limit
            throw new Error('Total upload size exceeds limit');
          }

          formData.append(`videos[${instrumentName}]`, videoBlob, `${instrumentName}.mp4`);
          console.log(`Added video for ${instrumentName}, size: ${videoBlob.size}, total: ${totalSize}`);
        } catch (error) {
          console.error(`Error processing ${instrumentName}:`, error);
          throw error;
        }
      }

      const response = await axios.post('http://localhost:3000/api/compose', formData, {
        responseType: 'blob',
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
        timeout: 900000, // 15 minutes
        timeoutErrorMessage: 'Video composition took too long. Please try again.',
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProgress(percentCompleted);
        },
        // Add download progress handling
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