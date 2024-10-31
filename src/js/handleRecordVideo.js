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

  const response = await fetch('http://localhost:3000/api/autotune', {
    method: 'POST',
    body: formData,
  });

  const autotunedVideoFile = await response.blob();
  return new File([autotunedVideoFile], 'autotuned-video.mp4');
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
    const videoBlob = new Blob(chunks, { type: 'video/mp4' });
    const videoURL = URL.createObjectURL(videoBlob);
    setRecordedVideoURL(videoURL);

    const videoFile = new File([videoBlob], 'webcam-video.mp4');

    // Autotune the video to middle C
    const autotunedVideoFile = await autotuneToMiddleC(videoFile);
    const autotunedVideoURL = URL.createObjectURL(autotunedVideoFile);
    setAutotunedVideoURL(autotunedVideoURL);
  };

  // Stop recording after 5 seconds
  setTimeout(() => {
    mediaRecorder.stop();
  }, 5000);
};