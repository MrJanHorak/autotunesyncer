export const uploadVideo = async (videoFile) => {
  const formData = new FormData();
  formData.append('video', videoFile);

  const response = await fetch('http://localhost:3000/api/upload/', {
    method: 'POST',
    body: formData,
  });

  const audioFile = await response.blob();
  // Process and handle audio file (e.g., download or play)
  return audioFile;
};

// eslint-disable-next-line no-unused-vars
const uploadDirectVideo = async (videoFile) => {
  try {
    console.log('Uploading video directly...', videoFile.size);
    const formData = new FormData();
    formData.append('video', videoFile);

    const response = await fetch('http://localhost:3000/api/upload', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
    }

    return videoFile;
  } catch (error) {
    console.error('Error in direct upload:', error);
    throw error;
  }
};

// eslint-disable-next-line no-unused-vars
const autotuneToMiddleC = async (videoFile) => {
  const formData = new FormData();
  formData.append('video', videoFile);

  try {
    console.log('Sending video for autotuning...', videoFile.size);
    const response = await fetch('http://localhost:3000/api/autotune', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
    }

    const contentType = response.headers.get('content-type');
    if (!contentType || !contentType.includes('video/mp4')) {
      console.warn('Unexpected content type:', contentType);
    }

    const autotunedVideoBlob = await response.blob();
    console.log('Received autotuned video:', {
      size: autotunedVideoBlob.size,
      type: autotunedVideoBlob.type
    });
    
    if (autotunedVideoBlob.size === 0) {
      throw new Error('Received empty video file');
    }

    return new File([autotunedVideoBlob], 'autotuned-video.mp4', { 
      type: 'video/mp4'
    });
  } catch (error) {
    console.error('Error in autotuning:', error);
    throw error;
  }
};

// export async function handleRecord(onRecordingComplete, onAutotuneComplete, autotuneEnabled = true) {
//   let mediaRecorder;
//   const chunks = [];

//   return new Promise((resolve, reject) => {
//     // Start recording function
//     const startRecording = (stream) => {
//       mediaRecorder = new MediaRecorder(stream);

//       mediaRecorder.ondataavailable = (e) => {
//         if (e.data.size > 0) {
//           chunks.push(e.data);
//         }
//       };

//       mediaRecorder.onstop = async () => {
//         const blob = new Blob(chunks, { type: 'video/webm' });
//         const recordedURL = URL.createObjectURL(blob);
//         onRecordingComplete(recordedURL);

//         if (autotuneEnabled) {
//           try {
//             // Here you would implement your autotune processing
//             // For now, we'll just pass through the recorded URL
//             onAutotuneComplete(recordedURL);
//           } catch (error) {
//             console.error('Autotune processing failed:', error);
//             onAutotuneComplete(recordedURL); // Fallback to original
//           }
//         } else {
//           onAutotuneComplete(recordedURL);
//         }
//         resolve();
//       };

//       mediaRecorder.start(1000); // Save data every second
//     };

//     // Get the stream from the VideoRecorder component
//     navigator.mediaDevices
//       .getUserMedia({ video: true, audio: true })
//       .then(startRecording)
//       .catch((error) => {
//         console.error('Error accessing media devices:', error);
//         reject(error);
//       });
//   });
// }

export const handleRecord = async (setRecordedVideoURL, setAutotunedVideoURL, isAutotuneEnabled = true, duration = 5000) => {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: true,
    audio: true,
  });

  return new Promise((resolve, reject) => {
    const mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'video/webm;codecs=h264,opus'
    });

    let chunks = [];
    let recordingStopped = false;

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        chunks.push(e.data);
      }
    };

    mediaRecorder.onstop = async () => {
      try {
        const videoBlob = new Blob(chunks, { type: 'video/mp4' });
        const videoURL = URL.createObjectURL(videoBlob);
        setRecordedVideoURL(videoURL);

        const videoFile = new File([videoBlob], 'webcam-video.mp4', { type: 'video/mp4' });

        if (isAutotuneEnabled) {
          console.log('Processing video for autotuning...');
          const autotunedVideoFile = await autotuneToMiddleC(videoFile);
          const autotunedVideoURL = URL.createObjectURL(autotunedVideoFile);
          setAutotunedVideoURL(autotunedVideoURL);
        } else {
          console.log('Recording completed without autotuning.');
          setAutotunedVideoURL(videoURL);
        }
        resolve();
      } catch (error) {
        console.error('Error in recording stop handler:', error);
        reject(error);
      }
    };

    // Start recording with smaller time slices for more frequent data availability
    mediaRecorder.start(100);

    // Set timeout to stop recording after specified duration
    setTimeout(() => {
      if (!recordingStopped) {
        recordingStopped = true;
        mediaRecorder.stop();
        stream.getTracks().forEach(track => track.stop());
      }
    }, duration);
  });
};

export const handleUploadedVideoAutotune = async (videoFile, onAutotuneComplete) => {
  try {
    // Here you would process the uploaded video file similar to how
    // you process the recorded video in handleRecord
    
    // Example implementation:
    // 1. Extract audio from video
    // 2. Process audio with autotune
    // 3. Merge autotuned audio back with video
    // 4. Create URL for final video
    
    // For now, this is a placeholder that just passes through the original video
    const videoUrl = URL.createObjectURL(videoFile);
    
    // Call the callback with the processed video URL
    onAutotuneComplete(videoUrl);
    
    return videoUrl;
  } catch (error) {
    console.error('Error processing uploaded video:', error);
    throw error;
  }
};