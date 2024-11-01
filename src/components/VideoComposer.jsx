/* eslint-disable no-unused-vars */
/* eslint-disable react/prop-types */
// import { useState } from 'react';
// import axios from 'axios';

// const VideoComposer = ({ videoFiles, midiData }) => {
//   const [isProcessing, setIsProcessing] = useState(false);
//   const [progress, setProgress] = useState(0);
//   const [composedVideoUrl, setComposedVideoUrl] = useState(null);

//   const startComposition = async () => {
//     setIsProcessing(true);
//     setProgress(0);

//     try {
//       const formData = new FormData();
//       formData.append('midiData', new Blob([midiData], { type: 'audio/midi' }));
      
//       // Append video files
//       Object.entries(videoFiles).forEach(([instrument, blob]) => {
//         formData.append(`videos[${instrument}]`, blob);
//       });

//       const response = await axios.post('http://localhost:3000/api/compose', formData, {
//         responseType: 'blob',
//         onUploadProgress: (progressEvent) => {
//           const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
//           setProgress(percentCompleted);
//         }
//       });

//       const url = URL.createObjectURL(response.data);
//       setComposedVideoUrl(url);
//     } catch (error) {
//       console.error('Composition failed:', error);
//     } finally {
//       setIsProcessing(false);
//     }
//   };

//   return (
//     <div className="video-composer">
//       <button
//         onClick={startComposition}
//         disabled={isProcessing}
//         className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
//       >
//         {isProcessing ? 'Processing...' : 'Start Composition'}
//       </button>

//       {isProcessing && (
//         <div className="mt-4">
//           <div className="w-full h-2 bg-gray-200 rounded">
//             <div
//               className="h-full bg-blue-500 rounded"
//               style={{ width: `${progress}%` }}
//             />
//           </div>
//           <p className="text-sm text-gray-600 mt-1">{progress}% complete</p>
//         </div>
//       )}

//       {composedVideoUrl && (
//         <div className="mt-4">
//           <video
//             src={composedVideoUrl}
//             controls
//             className="w-full max-w-4xl"
//           />
//         </div>
//       )}
//     </div>
//   );
// };

// export default VideoComposer;

import { useState } from 'react';
import { videoService } from '../../services/videoServices';

const VideoComposer = ({ videoFiles, midiData }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [composedVideoUrl, setComposedVideoUrl] = useState(null);
  const [error, setError] = useState(null);

  // const startComposition = async () => {
  //   setIsProcessing(true);
  //   setProgress(0);
  //   setError(null);

  //   try {
  //     const formData = new FormData();
      
  //     // Convert MIDI data to proper format
  //     // If midiData is already a Uint8Array or ArrayBuffer, use it directly
  //     // Otherwise, serialize it properly
  //     let midiBlob;
  //     if (midiData instanceof Uint8Array || midiData instanceof ArrayBuffer) {
  //       midiBlob = new Blob([midiData], { type: 'audio/midi' });
  //     } else {
  //       // Assuming midiData is an object with MIDI properties
  //       const midiString = JSON.stringify(midiData);
  //       midiBlob = new Blob([midiString], { type: 'application/json' });
  //     }
  //     formData.append('midiData', midiBlob);
      
  //     // Append video files
  //     Object.entries(videoFiles).forEach(([instrument, blob]) => {
  //       // Ensure the blob is actually a Blob or File object
  //       if (blob instanceof Blob || blob instanceof File) {
  //         formData.append(`videos[${instrument}]`, blob);
  //       } else {
  //         throw new Error(`Invalid video file format for instrument: ${instrument}`);
  //       }
  //     });

  //     const response = await axios.post('http://localhost:3000/api/compose', formData, {
  //       responseType: 'blob',
  //       headers: {
  //         'Content-Type': 'multipart/form-data',
  //       },
  //       onUploadProgress: (progressEvent) => {
  //         const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
  //         setProgress(percentCompleted);
  //       }
  //     });

  //     // Check if the response is an error message
  //     const contentType = response.headers['content-type'];
  //     if (contentType && contentType.includes('application/json')) {
  //       // Parse error message
  //       const reader = new FileReader();
  //       reader.onload = () => {
  //         const errorData = JSON.parse(reader.result);
  //         setError(errorData.error || 'Composition failed');
  //       };
  //       reader.readAsText(response.data);
  //       return;
  //     }

  //     const url = URL.createObjectURL(response.data);
  //     setComposedVideoUrl(url);
  //   } catch (error) {
  //     console.error('Composition failed:', error);
  //     setError(error.message || 'Composition failed');
  //   } finally {
  //     setIsProcessing(false);
  //   }
  // };

  const startComposition = async () => {
    try {
      setIsProcessing(true);
      const composedVideoBlob = await videoService.composeVideos(
        videoFiles,
        midiData
      );
      const url = URL.createObjectURL(composedVideoBlob);
      setComposedVideoUrl(url);
    } catch (error) {
      setError(error.message);
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