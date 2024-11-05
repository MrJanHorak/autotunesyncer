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

export const handleRecord = async (setRecordedVideoURL, setAutotunedVideoURL) => {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: true,
    audio: true,
  });
  const mediaRecorder = new MediaRecorder(stream);

  mediaRecorder.start();
  let chunks = [];
  mediaRecorder.ondataavailable = (e) => {
    chunks.push(e.data);
  };

  mediaRecorder.onstop = async () => {
    try {
      const videoBlob = new Blob(chunks, { type: 'video/mp4' });
      const videoURL = URL.createObjectURL(videoBlob);
      setRecordedVideoURL(videoURL);

      const videoFile = new File([videoBlob], 'webcam-video.mp4', { type: 'video/mp4' });

      console.log('Processing video for autotuning...');
      const autotunedVideoFile = await autotuneToMiddleC(videoFile);
      console.log('Autotuned video received, creating URL...');
      const autotunedVideoURL = URL.createObjectURL(autotunedVideoFile);
      setAutotunedVideoURL(autotunedVideoURL);
    } catch (error) {
      console.error('Error in recording stop handler:', error);
    }
  };

  // Stop recording after 5 seconds
  setTimeout(() => {
    mediaRecorder.stop();
  }, 5000);
};