import { useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import LegalModal from '../Legal/LegalModal.jsx';
import { CURRENT_LEGAL_VERSIONS } from '../../../shared/legalDocuments.js';
import './AuthPage.css';

export default function AuthPage() {
  const { login, register } = useAuth();
  const [mode, setMode] = useState('login'); // 'login' | 'register'
  const [form, setForm] = useState({
    username: '',
    email: '',
    password: '',
    acceptLegal: false,
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeLegalDocument, setActiveLegalDocument] = useState(null);

  const handleChange = (e) => {
    const value =
      e.target.type === 'checkbox' ? e.target.checked : e.target.value;
    setForm((prev) => ({ ...prev, [e.target.name]: value }));
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      if (mode === 'login') {
        await login(form.email, form.password);
      } else {
        if (!form.acceptLegal) {
          throw new Error(
            'You must accept the Terms, Privacy Policy, and Copyright Policy to create an account.',
          );
        }
        await register(form.username, form.email, form.password, {
          accepted: true,
          versions: CURRENT_LEGAL_VERSIONS,
        });
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      {activeLegalDocument && (
        <LegalModal
          documentKey={activeLegalDocument}
          onClose={() => setActiveLegalDocument(null)}
        />
      )}
      <div className="auth-card">
        <h1 className="auth-title">🎵 AutoTuneSyncer</h1>
        <p className="auth-subtitle">Sign in to manage your projects and clips</p>

        <div className="auth-tabs">
          <button
            className={`auth-tab ${mode === 'login' ? 'active' : ''}`}
            onClick={() => { setMode('login'); setError(''); }}
          >
            Sign In
          </button>
          <button
            className={`auth-tab ${mode === 'register' ? 'active' : ''}`}
            onClick={() => { setMode('register'); setError(''); }}
          >
            Create Account
          </button>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          {mode === 'register' && (
            <div className="auth-field">
              <label htmlFor="username">Username</label>
              <input
                id="username"
                name="username"
                type="text"
                placeholder="Your username"
                value={form.username}
                onChange={handleChange}
                required
                autoComplete="username"
              />
            </div>
          )}

          <div className="auth-field">
            <label htmlFor="email">Email</label>
            <input
              id="email"
              name="email"
              type="email"
              placeholder="you@example.com"
              value={form.email}
              onChange={handleChange}
              required
              autoComplete="email"
            />
          </div>

          <div className="auth-field">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              name="password"
              type="password"
              placeholder={mode === 'register' ? 'At least 6 characters' : 'Your password'}
              value={form.password}
              onChange={handleChange}
              required
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              minLength={mode === 'register' ? 6 : undefined}
            />
          </div>

          {mode === 'register' && (
            <label className="auth-consent">
              <input
                name="acceptLegal"
                type="checkbox"
                checked={form.acceptLegal}
                onChange={handleChange}
                required
              />
              <span>
                I agree to the{' '}
                <button type="button" onClick={() => setActiveLegalDocument('terms')}>
                  Terms of Use
                </button>
                ,{' '}
                <button type="button" onClick={() => setActiveLegalDocument('privacy')}>
                  Privacy Policy
                </button>
                , and{' '}
                <button type="button" onClick={() => setActiveLegalDocument('copyright')}>
                  Copyright Policy
                </button>
                . I understand I may upload only media I own or am authorized to use.
              </span>
            </label>
          )}

          {error && <p className="auth-error">{error}</p>}

          <button className="auth-submit" type="submit" disabled={loading}>
            {loading ? 'Please wait…' : mode === 'login' ? 'Sign In' : 'Create Account'}
          </button>
        </form>

        <div className="auth-legal-note">
          Upload only content you own or are licensed to use. Valid infringement notices may lead to removal and repeat-infringer enforcement.
        </div>

        <div className="auth-legal-links">
          <button type="button" onClick={() => setActiveLegalDocument('terms')}>
            Terms
          </button>
          <button type="button" onClick={() => setActiveLegalDocument('privacy')}>
            Privacy
          </button>
          <button type="button" onClick={() => setActiveLegalDocument('copyright')}>
            Copyright
          </button>
        </div>
      </div>
    </div>
  );
}
