import { useState } from 'react';
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
            try {
              const data = JSON.parse(xhr.responseText);
              resolve(data);
            } catch { resolve({}); }
          } else {
            try {
              const errData = JSON.parse(xhr.responseText);
              reject(new Error(errData.error || `Upload failed (${xhr.status})`));
            } catch {
              reject(new Error(`Upload failed (${xhr.status})`));
            }
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
            {uploading && (
              <div style={{ margin: '0.75rem 0' }}>
                <div style={{ background: '#e5e7eb', borderRadius: 4, height: 6, overflow: 'hidden' }}>
                  <div style={{ width: `${uploadPct}%`, height: '100%', background: '#0f3460', transition: 'width 0.3s' }} />
                </div>
                <p style={{ fontSize: '0.8rem', color: '#6b7280', marginTop: 4 }}>Uploading… {uploadPct}%</p>
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
