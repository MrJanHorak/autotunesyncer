const VideoRecorder = ({ track, onRecordingComplete }) => {
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const [isRecording, setIsRecording] = useState(false);
  const [countdown, setCountdown] = useState(3);
  const [volume, setVolume] = useState(1);
  const [videoConstraints, setVideoConstraints] = useState({
    width: { ideal: 1280 },
    height: { ideal: 720 },
    frameRate: { ideal: 30 }
  });

  const startCountdown = () => {
    setCountdown(3);
    const interval = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          clearInterval(interval);
          startRecording();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: true
      });
      
      videoRef.current.srcObject = stream;
      
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'video/webm;codecs=vp9,opus'
      });
      
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'video/webm' });
        onRecordingComplete(blob);
      };

      setIsRecording(true);
      mediaRecorder.start();

      // Stop recording after track duration
      setTimeout(() => {
        stopRecording();
      }, track.duration * 1000);
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      setIsRecording(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="relative">
        <video
          ref={videoRef}
          autoPlay
          muted
          className="w-full rounded-lg border"
        />
        {countdown > 0 && (
          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 text-white text-6xl">
            {countdown}
          </div>
        )}
      </div>

      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Volume2 className="h-4 w-4" />
          <Slider
            value={[volume]}
            onValueChange={([value]) => setVolume(value)}
            max={1}
            step={0.1}
            className="w-32"
          />
        </div>

        <div className="flex gap-2">
          {!isRecording ? (
            <Button onClick={startCountdown} className="flex-1">
              Start Recording
            </Button>
          ) : (
            <Button onClick={stopRecording} variant="destructive" className="flex-1">
              Stop Recording
            </Button>
          )}
          <Button
            variant="outline"
            onClick={() => setVideoConstraints(prev => ({
              ...prev,
              width: { ideal: prev.width.ideal === 1280 ? 1920 : 1280 },
              height: { ideal: prev.height.ideal === 720 ? 1080 : 720 }
            }))}
          >
            <RefreshCcw className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
};

export default VideoRecorder;