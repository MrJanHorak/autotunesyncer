import { useEffect, useRef, useState } from 'react';
import PropTypes from 'prop-types';

const VideoComposer = ({ videoFiles, midiData }) => {
  const canvasRef = useRef(null);
  const audioContextRef = useRef(null);
  const [isRendering, setIsRendering] = useState(false);
  const [progress, setProgress] = useState(0);
  const [composedVideoUrl, setComposedVideoUrl] = useState(null);
  const mediaRecorderRef = useRef(null);
  const videoElementsRef = useRef({});
  const chunksRef = useRef([]);
  const progressRef = useRef(0);

  const getTotalDuration = () => {
    if (!midiData?.tracks) return 0;
    return midiData.tracks.reduce((maxDuration, track) => {
      const trackDuration = track.notes.reduce((max, note) => {
        return Math.max(max, note.time + note.duration);
      }, 0);
      return Math.max(maxDuration, trackDuration);
    }, 0);
  };

  const setupCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const totalVideos = Object.keys(videoFiles).length;
    
    const cols = Math.ceil(Math.sqrt(totalVideos));
    const rows = Math.ceil(totalVideos / cols);
    
    canvas.width = cols * 320;
    canvas.height = rows * 240;
    
    return { ctx, cols, rows };
  };

  const setupAudioProcessing = async () => {
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      await audioContextRef.current.close();
    }
    audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    const audioDestination = audioContextRef.current.createMediaStreamDestination();

    const audioSources = await Promise.all(
      Object.entries(videoElementsRef.current).map(async ([key, video]) => {
        const source = audioContextRef.current.createMediaElementSource(video);
        const gainNode = audioContextRef.current.createGain();
        gainNode.gain.value = 1.0;

        source.connect(gainNode);
        gainNode.connect(audioDestination);

        return { key, source, gainNode };
      })
    );

    return audioDestination.stream;
  };

  const startRendering = async () => {
    setIsRendering(true);
    progressRef.current = 0;
    setProgress(0);

    try {
      const { ctx, cols, rows } = setupCanvas();
      const canvas = canvasRef.current;
      const totalDuration = getTotalDuration();

      await Promise.all(Object.entries(videoFiles).map(async ([key, blob]) => {
        const video = document.createElement('video');
        video.src = URL.createObjectURL(blob);
        video.muted = true;
        video.preload = 'auto';
        video.onloadeddata = () => {
          videoElementsRef.current[key] = video;
        };
      }));

      const audioStream = await setupAudioProcessing();

      const canvasStream = canvas.captureStream(30);
      const combinedTracks = [
        ...canvasStream.getVideoTracks(),
        ...audioStream.getAudioTracks()
      ];

      const combinedStream = new MediaStream(combinedTracks);

      const mediaRecorder = new MediaRecorder(combinedStream, {
        mimeType: 'video/webm; codecs=vp8,opus',
        videoBitsPerSecond: 8000000,
        audioBitsPerSecond: 128000
      });

      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        if (chunksRef.current.length > 0) {
          const blob = new Blob(chunksRef.current, { type: 'video/webm' });
          const url = URL.createObjectURL(blob);
          setComposedVideoUrl(url);
          setIsRendering(false);
          setProgress(100);
        } else {
          console.error('No media data captured');
        }

        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
          audioContextRef.current.close();
        }
      };

      await Promise.all(
        Object.values(videoElementsRef.current).map(video => video.play())
      );

      mediaRecorder.start(1000);

      let startTime = performance.now();
      let lastFrameTime = startTime;
      const targetFrameTime = 1000 / 30;

      const animate = (timestamp) => {
        const elapsed = (timestamp - startTime) / 1000;

        if (elapsed > totalDuration) {
          mediaRecorder.stop();
          return;
        }

        if (timestamp - lastFrameTime >= targetFrameTime) {
          renderFrame(ctx, cols);
          lastFrameTime = timestamp;
        }

        if (elapsed / totalDuration * 100 > progressRef.current + 5) {
          progressRef.current = (elapsed / totalDuration) * 100;
          setProgress(progressRef.current);
        }

        requestAnimationFrame(animate);
      };

      requestAnimationFrame(animate);

    } catch (error) {
      console.error('Error during rendering:', error);
      setIsRendering(false);
    }
  };

  const renderFrame = (ctx, cols) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    Object.entries(videoElementsRef.current).forEach(([key, video], index) => {
      const col = index % cols;
      const row = Math.floor(index / cols);
      const x = col * 320;
      const y = row * 240;

      ctx.drawImage(video, x, y, 320, 240);

      ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
      ctx.fillRect(x, y, 320, 30);
      ctx.fillStyle = 'white';
      ctx.font = '14px Arial';
      const [instrument] = key.split('-');
      ctx.fillText(instrument, x + 10, y + 20);
    });
  };

  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
      if (composedVideoUrl) {
        URL.revokeObjectURL(composedVideoUrl);
      }
      Object.values(videoElementsRef.current).forEach(video => {
        video.pause();
        video.src = '';
      });
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
      chunksRef.current = [];
    };
  }, [composedVideoUrl]);

  return (
    <div className="video-composer">
      <div className="controls">
        {!isRendering && !composedVideoUrl && (
          <button 
            onClick={startRendering}
            disabled={!videoFiles || Object.keys(videoFiles).length === 0}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
          >
            Start Composition
          </button>
        )}
        
        {isRendering && (
          <div className="rendering-progress space-y-2">
            <p>Rendering: {Math.round(progress)}%</p>
            <div className="w-full h-5 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-green-500 transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}
      </div>
      
      <canvas 
        ref={canvasRef}
        style={{ display: 'none' }}
      />
      
      {composedVideoUrl && (
        <div className="final-video mt-4">
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

VideoComposer.propTypes = {
  videoFiles: PropTypes.objectOf(PropTypes.instanceOf(Blob)).isRequired,
  midiData: PropTypes.object.isRequired
};

export default VideoComposer;
