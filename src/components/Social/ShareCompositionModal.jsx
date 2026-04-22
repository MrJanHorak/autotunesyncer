import { useState, useRef, useEffect } from 'react';
import PropTypes from 'prop-types';
import './Social.css';

const API_BASE = 'http://localhost:3000/api';

function getToken() {
  return localStorage.getItem('auth_token');
}

const ShareCompositionModal = ({ blob, onClose, onShared }) => {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [uploadPct, setUploadPct] = useState(0);

  // Thumbnail state
  const [thumbMode, setThumbMode] = useState('capture'); // 'capture' | 'upload'
  const [thumbnailBlob, setThumbnailBlob] = useState(null);
  const [thumbnailPreviewUrl, setThumbnailPreviewUrl] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const videoObjectUrlRef = useRef(null);

  // Create an object URL from the blob for the preview video
  useEffect(() => {
    if (blob) {
      const url = URL.createObjectURL(blob);
      videoObjectUrlRef.current = url;
      if (videoRef.current) videoRef.current.src = url;
    }
    return () => {
      if (videoObjectUrlRef.current) URL.revokeObjectURL(videoObjectUrlRef.current);
    };
  }, [blob]);

  // Revoke thumbnail preview URL when it changes
  useEffect(() => {
    return () => {
      if (thumbnailPreviewUrl) URL.revokeObjectURL(thumbnailPreviewUrl);
    };
  }, [thumbnailPreviewUrl]);

  const captureFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 360;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((b) => {
      if (!b) return;
      if (thumbnailPreviewUrl) URL.revokeObjectURL(thumbnailPreviewUrl);
      setThumbnailBlob(b);
      setThumbnailPreviewUrl(URL.createObjectURL(b));
    }, 'image/jpeg', 0.85);
  };

  const handleImageUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (thumbnailPreviewUrl) URL.revokeObjectURL(thumbnailPreviewUrl);
    setThumbnailBlob(file);
    setThumbnailPreviewUrl(URL.createObjectURL(file));
    e.target.value = '';
  };

  const clearThumbnail = () => {
    if (thumbnailPreviewUrl) URL.revokeObjectURL(thumbnailPreviewUrl);
    setThumbnailBlob(null);
    setThumbnailPreviewUrl(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!title.trim()) {
      setError('Please enter a title');
      return;
    }
    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('video', blob, 'composition.mp4');
    formData.append('title', title.trim());
    formData.append('description', description.trim());
    if (thumbnailBlob) {
      const ext = thumbnailBlob.type === 'image/png' ? 'png' : 'jpg';
      formData.append('thumbnail', thumbnailBlob, `thumbnail.${ext}`);
    }

    try {
      await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', `${API_BASE}/social/compositions`);
        const token = getToken();
        if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`);

        xhr.upload.onprogress = (ev) => {
          if (ev.lengthComputable) setUploadPct(Math.round((ev.loaded / ev.total) * 100));
        };

        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try { resolve(JSON.parse(xhr.responseText)); }
            catch { resolve({}); }
          } else {
            try { reject(new Error(JSON.parse(xhr.responseText).error || `Upload failed (${xhr.status})`)); }
            catch { reject(new Error(`Upload failed (${xhr.status})`)); }
          }
        };
        xhr.onerror = () => reject(new Error('Network error during upload'));
        xhr.send(formData);
      });

      setSuccess(true);
      if (onShared) onShared();
      setTimeout(() => onClose(), 2000);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className='modal-overlay' onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className='modal'>
        <h2 className='modal__title'>🎬 Share to Feed</h2>

        {success ? (
          <div className='modal__success'>
            ✅ Your composition is now live on the feed! Closing…
          </div>
        ) : (
          <form onSubmit={handleSubmit}>
            <div className='modal__field'>
              <label className='modal__label'>Title *</label>
              <input
                className='modal__input'
                type='text'
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder='Give your composition a name'
                maxLength={120}
                autoFocus
              />
            </div>

            <div className='modal__field'>
              <label className='modal__label'>Description</label>
              <textarea
                className='modal__textarea'
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder='Tell people what this is about (optional)'
                maxLength={500}
              />
            </div>

            {/* Thumbnail picker */}
            <div className='modal__field'>
              <label className='modal__label'>Preview Thumbnail</label>
              <div className='modal__thumb-tabs'>
                <button
                  type='button'
                  className={`modal__thumb-tab${thumbMode === 'capture' ? ' modal__thumb-tab--active' : ''}`}
                  onClick={() => setThumbMode('capture')}
                >
                  🎞 Capture Frame
                </button>
                <button
                  type='button'
                  className={`modal__thumb-tab${thumbMode === 'upload' ? ' modal__thumb-tab--active' : ''}`}
                  onClick={() => setThumbMode('upload')}
                >
                  📁 Upload Image
                </button>
              </div>

              {thumbMode === 'capture' && (
                <div className='modal__video-capture'>
                  <video
                    ref={videoRef}
                    className='modal__video-preview'
                    src={videoObjectUrlRef.current}
                    controls
                    muted
                  />
                  <button type='button' className='modal__capture-btn' onClick={captureFrame}>
                    📸 Capture current frame
                  </button>
                </div>
              )}

              {thumbMode === 'upload' && (
                <>
                  <button
                    type='button'
                    className='modal__capture-btn'
                    onClick={() => fileInputRef.current?.click()}
                  >
                    📁 Choose image…
                  </button>
                  <input
                    ref={fileInputRef}
                    type='file'
                    accept='image/*'
                    style={{ display: 'none' }}
                    onChange={handleImageUpload}
                  />
                </>
              )}

              {thumbnailPreviewUrl && (
                <div className='modal__thumb-preview'>
                  <img src={thumbnailPreviewUrl} alt='Thumbnail preview' className='modal__thumb-img' />
                  <button type='button' className='modal__thumb-clear' onClick={clearThumbnail} title='Remove'>✕</button>
                </div>
              )}
              {!thumbnailPreviewUrl && (
                <p className='modal__thumb-hint'>No thumbnail chosen — one will be auto-generated.</p>
              )}
            </div>

            {/* Hidden canvas for frame capture */}
            <canvas ref={canvasRef} style={{ display: 'none' }} />

            {uploading && (
              <div style={{ margin: '0.75rem 0' }}>
                <div style={{ background: 'rgba(255,255,255,0.1)', borderRadius: 4, height: 6, overflow: 'hidden' }}>
                  <div style={{ width: `${uploadPct}%`, height: '100%', background: 'var(--social-accent)', transition: 'width 0.3s' }} />
                </div>
                <p style={{ fontSize: '0.8rem', color: 'var(--social-text-muted)', marginTop: 4 }}>Uploading… {uploadPct}%</p>
              </div>
            )}
            {error && <p className='modal__error'>{error}</p>}
            <div className='modal__actions'>
              <button type='button' className='modal__btn modal__btn--cancel' onClick={onClose} disabled={uploading}>
                Cancel
              </button>
              <button type='submit' className='modal__btn modal__btn--submit' disabled={uploading}>
                {uploading ? 'Sharing…' : 'Share'}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};

ShareCompositionModal.propTypes = {
  blob: PropTypes.instanceOf(Blob).isRequired,
  onClose: PropTypes.func.isRequired,
  onShared: PropTypes.func,
};

export default ShareCompositionModal;

