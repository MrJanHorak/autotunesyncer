import { useState } from 'react';
import PropTypes from 'prop-types';
import { X, User, Lock, Check, AlertCircle } from 'lucide-react';
import { useAuth } from '../../context/AuthContext.jsx';

const API_BASE = 'http://localhost:3000/api';

function getToken() {
  return localStorage.getItem('auth_token');
}

async function apiFetch(path, options = {}) {
  const token = getToken();
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `Error ${res.status}`);
  return data;
}

function FieldMsg({ ok, msg }) {
  if (!msg) return null;
  return (
    <div className={`settings-msg ${ok ? 'settings-msg--ok' : 'settings-msg--err'}`}>
      {ok ? <Check size={13} /> : <AlertCircle size={13} />} {msg}
    </div>
  );
}

FieldMsg.propTypes = { ok: PropTypes.bool, msg: PropTypes.string };

const Settings = ({ onClose }) => {
  const { user, updateUser } = useAuth();
  const [tab, setTab] = useState('profile');

  // Profile tab state
  const [username, setUsername] = useState(user?.username || '');
  const [bio, setBio] = useState(user?.bio || '');
  const [profileSaving, setProfileSaving] = useState(false);
  const [profileMsg, setProfileMsg] = useState({ ok: false, msg: '' });

  // Email tab state
  const [email, setEmail] = useState(user?.email || '');
  const [emailPassword, setEmailPassword] = useState('');
  const [emailSaving, setEmailSaving] = useState(false);
  const [emailMsg, setEmailMsg] = useState({ ok: false, msg: '' });

  // Password tab state
  const [currentPw, setCurrentPw] = useState('');
  const [newPw, setNewPw] = useState('');
  const [confirmPw, setConfirmPw] = useState('');
  const [pwSaving, setPwSaving] = useState(false);
  const [pwMsg, setPwMsg] = useState({ ok: false, msg: '' });

  const handleSaveProfile = async (e) => {
    e.preventDefault();
    setProfileSaving(true);
    setProfileMsg({ ok: false, msg: '' });
    try {
      const data = await apiFetch('/auth/profile', {
        method: 'PATCH',
        body: JSON.stringify({ username: username.trim(), bio: bio.trim() }),
      });
      updateUser(data.user, data.token);
      setProfileMsg({ ok: true, msg: 'Profile updated!' });
    } catch (err) {
      setProfileMsg({ ok: false, msg: err.message });
    } finally {
      setProfileSaving(false);
    }
  };

  const handleSaveEmail = async (e) => {
    e.preventDefault();
    setEmailSaving(true);
    setEmailMsg({ ok: false, msg: '' });
    try {
      const data = await apiFetch('/auth/email', {
        method: 'PATCH',
        body: JSON.stringify({ email: email.trim(), currentPassword: emailPassword }),
      });
      updateUser(data.user, data.token);
      setEmailMsg({ ok: true, msg: 'Email updated!' });
      setEmailPassword('');
    } catch (err) {
      setEmailMsg({ ok: false, msg: err.message });
    } finally {
      setEmailSaving(false);
    }
  };

  const handleSavePassword = async (e) => {
    e.preventDefault();
    if (newPw !== confirmPw) {
      setPwMsg({ ok: false, msg: 'New passwords do not match' });
      return;
    }
    setPwSaving(true);
    setPwMsg({ ok: false, msg: '' });
    try {
      await apiFetch('/auth/password', {
        method: 'PATCH',
        body: JSON.stringify({ currentPassword: currentPw, newPassword: newPw }),
      });
      setPwMsg({ ok: true, msg: 'Password changed!' });
      setCurrentPw(''); setNewPw(''); setConfirmPw('');
    } catch (err) {
      setPwMsg({ ok: false, msg: err.message });
    } finally {
      setPwSaving(false);
    }
  };

  return (
    <div className='settings-overlay' onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className='settings-modal'>
        <div className='settings-modal__header'>
          <span className='settings-modal__title'>Account Settings</span>
          <button className='settings-modal__close' onClick={onClose} aria-label='Close settings'>
            <X size={18} />
          </button>
        </div>

        <div className='settings-tabs'>
          <button className={`settings-tab ${tab === 'profile' ? 'settings-tab--active' : ''}`} onClick={() => setTab('profile')}>
            <User size={14} /> Profile
          </button>
          <button className={`settings-tab ${tab === 'security' ? 'settings-tab--active' : ''}`} onClick={() => setTab('security')}>
            <Lock size={14} /> Security
          </button>
        </div>

        {tab === 'profile' && (
          <form className='settings-form' onSubmit={handleSaveProfile}>
            <label className='settings-label'>
              Username
              <input
                className='settings-input'
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                minLength={2}
                maxLength={30}
              />
            </label>
            <label className='settings-label'>
              Bio
              <textarea
                className='settings-input settings-textarea'
                value={bio}
                onChange={(e) => setBio(e.target.value)}
                rows={3}
                maxLength={200}
                placeholder='Tell the world about yourself…'
              />
              <span className='settings-char-count'>{bio.length}/200</span>
            </label>
            <FieldMsg {...profileMsg} />
            <button className='settings-save-btn' type='submit' disabled={profileSaving}>
              {profileSaving ? 'Saving…' : 'Save Profile'}
            </button>
          </form>
        )}

        {tab === 'security' && (
          <div className='settings-security'>
            {/* Email section */}
            <form className='settings-form' onSubmit={handleSaveEmail}>
              <h4 className='settings-section-title'>Email Address</h4>
              <label className='settings-label'>
                New email
                <input
                  className='settings-input'
                  type='email'
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </label>
              <label className='settings-label'>
                Current password (required to change email)
                <input
                  className='settings-input'
                  type='password'
                  value={emailPassword}
                  onChange={(e) => setEmailPassword(e.target.value)}
                  required
                  placeholder='Your current password'
                />
              </label>
              <FieldMsg {...emailMsg} />
              <button className='settings-save-btn' type='submit' disabled={emailSaving}>
                {emailSaving ? 'Saving…' : 'Update Email'}
              </button>
            </form>

            <hr className='settings-divider' />

            {/* Password section */}
            <form className='settings-form' onSubmit={handleSavePassword}>
              <h4 className='settings-section-title'>Change Password</h4>
              <label className='settings-label'>
                Current password
                <input className='settings-input' type='password' value={currentPw} onChange={(e) => setCurrentPw(e.target.value)} required />
              </label>
              <label className='settings-label'>
                New password
                <input className='settings-input' type='password' value={newPw} onChange={(e) => setNewPw(e.target.value)} required minLength={6} />
              </label>
              <label className='settings-label'>
                Confirm new password
                <input className='settings-input' type='password' value={confirmPw} onChange={(e) => setConfirmPw(e.target.value)} required />
              </label>
              <FieldMsg {...pwMsg} />
              <button className='settings-save-btn' type='submit' disabled={pwSaving}>
                {pwSaving ? 'Saving…' : 'Change Password'}
              </button>
            </form>
          </div>
        )}
      </div>
    </div>
  );
};

Settings.propTypes = {
  onClose: PropTypes.func.isRequired,
};

export default Settings;
