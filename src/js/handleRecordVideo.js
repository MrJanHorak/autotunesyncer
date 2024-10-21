const uploadVideo = async (videoFile) => {
  const formData = new FormData();
  formData.append('video', videoFile);

  const response = await fetch('/api/upload-video', {
    method: 'POST',
    body: formData,
  });

  const audioFile = await response.blob();
  // Process and handle audio file (e.g., download or play)
  return audioFile;
};

export const handleRecord = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  const mediaRecorder = new MediaRecorder(stream);
  
  mediaRecorder.start();
  let chunks = [];
  mediaRecorder.ondataavailable = (e) => {
    chunks.push(e.data);
  };

  mediaRecorder.onstop = () => {
    const videoBlob = new Blob(chunks, { type: 'video/mp4' });
    const videoFile = new File([videoBlob], 'webcam-video.mp4');
    // Upload to backend
    uploadVideo(videoFile);
  };

  // Stop recording after 5 seconds
  setTimeout(() => {
    mediaRecorder.stop();
  }, 5000);
};