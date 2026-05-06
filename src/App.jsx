/* eslint-disable no-unused-vars */
import { useEffect, useCallback, useState, useRef } from 'react';
import PropTypes from 'prop-types';
import {
  Film,
  Music,
  Grid3x3,
  FolderOpen,
  LogOut,
  Bell,
  Settings,
  User,
  Download,
  Upload,
  Undo2,
  Redo2,
} from 'lucide-react';

import { isDrumTrack, DRUM_NOTES, getNoteGroup } from './js/drumUtils';
import {
  DEFAULT_COMPOSITION_STYLE,
  DEFAULT_CLIP_STYLE,
} from './js/styleDefaults';
import InstrumentList from './components/InstrumentList/InstrumentList';
import InstrumentSidebar from './components/InstrumentSidebar/InstrumentSidebar';
import RecordingModal from './components/RecordingModal/RecordingModal';
import RightPanel from './components/RightPanel/RightPanel';

import { useMidiProcessing } from './hooks/useMidiProcessing';
import { useVideoRecording } from './hooks/useVideoRecording';
import { useAuth } from './context/AuthContext';
import { useProject } from './context/ProjectContext';
import { useUndoRedo } from './hooks/useUndoRedo';
import { useProjectSync } from './hooks/useProjectSync';
import {
  configureApiService,
  apiFetch,
  uploadClip,
  uploadProjectBackground,
  deleteProjectBackground,
  downloadProjectExport,
  importProjectFromZip,
} from './services/apiService';

// Components
import AuthPage from './components/Auth/AuthPage';
import LandingPage from './components/LandingPage';
import ProjectManager from './components/Projects/ProjectManager';
import MidiUploader from './components/MidiUploader/';
import MidiInfoDisplay from './components/MidiInfoDisplay/MidiInfoDisplay';
import RecordingSection from './components/RecordingSection/RecordingSection';
import CompositionSection from './components/CompositionSection/CompositionSection';
import MidiParser from './components/MidiParser/MidiParser';
import ProgressBar from './components/ProgressBar/ProgressBar';
import Grid from './components/Grid/Grid';
import Mixer from './components/Mixer/Mixer';
import PreviewPlayer from './components/PreviewPlayer/PreviewPlayer';
import CompositionStylePanel from './components/CompositionStylePanel/CompositionStylePanel';

// Social components
import SocialFeed from './components/Social/SocialFeed.jsx';
import CompositionDetail from './components/Social/CompositionDetail.jsx';
import UserProfile from './components/Social/UserProfile.jsx';
import Notifications from './components/Social/Notifications.jsx';
import SettingsModal from './components/Social/Settings.jsx';
import './components/Social/Social.css';

import './App.css';

const normalizeInstrumentName = (name) =>
  name.toLowerCase().replace(/\s+/g, '_');

const toInstrumentKey = (instrument) => {
  if (instrument.isDrum) {
    const name = (instrument.group || instrument.name || '')
      .toLowerCase()
      .replace(/\s+/g, '_');
    return `drum_${name}`;
  }
  return normalizeInstrumentName(instrument.name || '');
};

function base64ToFile(dataUrl, filename) {
  const [header, data] = dataUrl.split(',');
  const mime = header.match(/:(.*?);/)?.[1] || 'audio/midi';
  const bytes = atob(data);
  const arr = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
  return new File([arr], filename, { type: mime });
}

function App() {
  // All hooks must be at the top level, outside any conditionals
  const {
    user,
    isAuthenticated,
    loading: authLoading,
    token,
    logout,
  } = useAuth();
  const { currentProject, selectProject } = useProject();

  // Top-level view: 'compose' (requires project) | 'feed' (social)
  const [appView, setAppView] = useState('compose');

  // Social navigation: { page: 'feed' | 'detail' | 'profile', id: null | string }
  const [socialNav, setSocialNav] = useState({ page: 'feed', id: null });

  // Handle deep-link: ?composition=ID — open the feed and navigate to that composition
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const compositionId = params.get('composition');
    if (compositionId) {
      setAppView('feed');
      setSocialNav({ page: 'detail', id: compositionId });
      // Clean the URL so refreshing doesn't re-trigger
      window.history.replaceState({}, '', window.location.pathname);
    }
  }, []);

  // When a project is selected while on the Projects tab, auto-switch to Editor
  useEffect(() => {
    if (currentProject && appView === 'projects') {
      setAppView('compose');
    }
  }, [currentProject?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Auth modal state for guests
  const [showAuth, setShowAuth] = useState(false);

  // User avatar dropdown menu
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const userMenuRef = useRef(null);
  useEffect(() => {
    if (!userMenuOpen) return;
    const handler = (e) => {
      if (userMenuRef.current && !userMenuRef.current.contains(e.target))
        setUserMenuOpen(false);
    };
    const keyHandler = (e) => {
      if (e.key === 'Escape') setUserMenuOpen(false);
    };
    document.addEventListener('mousedown', handler);
    document.addEventListener('keydown', keyHandler);
    return () => {
      document.removeEventListener('mousedown', handler);
      document.removeEventListener('keydown', keyHandler);
    };
  }, [userMenuOpen]);

  // Notifications panel
  const [notifOpen, setNotifOpen] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);

  const fetchUnreadCount = useCallback(async () => {
    if (!isAuthenticated) return;
    try {
      const token = localStorage.getItem('auth_token');
      const res = await fetch(
        'http://localhost:3000/api/social/notifications/unread-count',
        {
          headers: { Authorization: `Bearer ${token}` },
        },
      );
      if (res.ok) {
        const data = await res.json();
        setUnreadCount(data.count || 0);
      }
    } catch {
      /* ignore */
    }
  }, [isAuthenticated]);

  useEffect(() => {
    fetchUnreadCount();
    const interval = setInterval(fetchUnreadCount, 60000);
    return () => clearInterval(interval);
  }, [fetchUnreadCount]);

  // Settings modal
  const [settingsOpen, setSettingsOpen] = useState(false);

  // Auth modal wrapper
  function AuthPageModal({ onClose }) {
    return (
      <div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          background: 'rgba(0,0,0,0.7)',
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <div
          style={{
            position: 'relative',
            background: '#181828',
            borderRadius: 12,
            padding: 32,
            minWidth: 340,
          }}
        >
          <button
            onClick={onClose}
            style={{
              position: 'absolute',
              top: 12,
              right: 12,
              background: 'none',
              border: 'none',
              color: '#fff',
              fontSize: 22,
              cursor: 'pointer',
            }}
          >
            ×
          </button>
          <AuthPage />
        </div>
      </div>
    );
  }
  AuthPageModal.propTypes = { onClose: PropTypes.func.isRequired };

  // Wire API service so all fetch helpers include auth headers + projectId.
  // Called synchronously (not in useEffect) so child effects can use apiFetch immediately.
  configureApiService({
    getToken: () => token,
    getProjectId: () => currentProject?.id ?? null,
  });

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {}, [token, currentProject]);

  // Show loading spinner while verifying stored token
  if (authLoading) {
    return (
      <div
        style={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <p style={{ color: '#666', fontSize: '1.1rem' }}>Loading…</p>
      </div>
    );
  }

  // Show landing/about page for guests
  if (!isAuthenticated) {
    return (
      <>
        <nav className='app-nav'>
          <button className='app-nav__brand' onClick={() => setShowAuth(true)}>
            <span className='app-nav__brand-icon'>
              <Film size={28} />
            </span>
            Symphovie
          </button>
          <div className='app-nav__tabs'>
            <button className='app-nav__tab app-nav__tab--active'>
              <Grid3x3 className='app-nav__tab-icon' /> Feed
            </button>
          </div>
          <div className='app-nav__right'>
            <button className='app-nav__btn' onClick={() => setShowAuth(true)}>
              Login / Sign Up
            </button>
          </div>
        </nav>
        <LandingPage onLogin={() => setShowAuth(true)} />
        {showAuth && <AuthPageModal onClose={() => setShowAuth(false)} />}
      </>
    );
  }

  const navBar = (
    <nav className='app-nav'>
      {/* Logo = Home: clicking goes back to compose root */}
      <button
        className='app-nav__brand'
        onClick={() => {
          setAppView('compose');
          setSocialNav({ page: 'feed', id: null });
        }}
      >
        <span className='app-nav__brand-icon'>
          <Film size={28} />
        </span>
        Symphovie
      </button>

      <div className='app-nav__tabs'>
        <button
          className={`app-nav__tab${appView === 'compose' ? ' app-nav__tab--active' : ''}`}
          onClick={() => setAppView('compose')}
        >
          <Music className='app-nav__tab-icon' /> Editor
        </button>
        <button
          className={`app-nav__tab${appView === 'feed' ? ' app-nav__tab--active' : ''}`}
          onClick={() => {
            setAppView('feed');
            setSocialNav({ page: 'feed', id: null });
          }}
        >
          <Grid3x3 className='app-nav__tab-icon' /> Feed
        </button>
        <button
          className={`app-nav__tab${appView === 'projects' ? ' app-nav__tab--active' : ''}`}
          onClick={() => setAppView('projects')}
        >
          <FolderOpen className='app-nav__tab-icon' /> Projects
        </button>
      </div>

      <div className='app-nav__right'>
        {appView === 'compose' && currentProject && (
          <span className='app-nav__project-name'>
            <FolderOpen size={14} /> {currentProject.name}
          </span>
        )}
        {user && (
          <>
            {/* Bell icon with unread dot */}
            <button
              className='app-nav__bell-btn'
              onClick={() => {
                setNotifOpen((v) => !v);
                setUserMenuOpen(false);
              }}
              aria-label='Notifications'
            >
              <Bell size={18} />
              {unreadCount > 0 && (
                <span className='app-nav__notif-dot'>
                  {unreadCount > 9 ? '9+' : unreadCount}
                </span>
              )}
            </button>

            <div className='app-nav__user-menu' ref={userMenuRef}>
              <button
                className='app-nav__avatar-btn'
                onClick={() => {
                  setUserMenuOpen((v) => !v);
                  setNotifOpen(false);
                }}
                aria-label='User menu'
                aria-expanded={userMenuOpen}
              >
                {user.profileImageUrl ? (
                  <img
                    src={user.profileImageUrl}
                    alt={user.username}
                    className='app-nav__avatar-img'
                  />
                ) : (
                  <span className='app-nav__avatar-initials'>
                    {user.username?.[0]?.toUpperCase() || 'U'}
                  </span>
                )}
              </button>
              {userMenuOpen && (
                <div className='app-nav__dropdown'>
                  <div className='app-nav__dropdown-header'>
                    <span className='app-nav__dropdown-username'>
                      @{user.username}
                    </span>
                  </div>
                  <button
                    className='app-nav__dropdown-item'
                    onClick={() => {
                      setAppView('feed');
                      setSocialNav({ page: 'profile', id: user.id });
                      setUserMenuOpen(false);
                    }}
                  >
                    <User size={15} /> My Profile
                  </button>
                  <button
                    className='app-nav__dropdown-item'
                    onClick={() => {
                      setSettingsOpen(true);
                      setUserMenuOpen(false);
                    }}
                  >
                    <Settings size={15} /> Settings
                  </button>
                  <div className='app-nav__dropdown-divider' />
                  <button
                    className='app-nav__dropdown-item app-nav__dropdown-item--danger'
                    onClick={() => {
                      setUserMenuOpen(false);
                      logout();
                    }}
                  >
                    <LogOut size={15} /> Sign Out
                  </button>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </nav>
  );

  const overlays = (
    <>
      {notifOpen && (
        <div className='notif-panel-wrap'>
          <Notifications
            onClose={() => {
              setNotifOpen(false);
              setUnreadCount(0);
            }}
            onSelectComposition={(id) => {
              setAppView('feed');
              setSocialNav({ page: 'detail', id });
              setNotifOpen(false);
            }}
          />
        </div>
      )}
      {settingsOpen && <SettingsModal onClose={() => setSettingsOpen(false)} />}
    </>
  );

  if (appView === 'feed') {
    return (
      <div style={{ minHeight: '100vh', background: 'var(--social-bg)' }}>
        {navBar}
        {overlays}
        {socialNav.page === 'feed' && (
          <SocialFeed
            onSelectComposition={(id) => setSocialNav({ page: 'detail', id })}
            onSelectUser={(id) => setSocialNav({ page: 'profile', id })}
            disableInteractions={!isAuthenticated}
          />
        )}
        {socialNav.page === 'detail' && (
          <div style={{ maxWidth: 1200, margin: '0 auto', padding: '0 1rem' }}>
            <CompositionDetail
              compositionId={socialNav.id}
              onBack={() => setSocialNav({ page: 'feed', id: null })}
              onSelectUser={(id) => setSocialNav({ page: 'profile', id })}
              onSelectComposition={(id) => setSocialNav({ page: 'detail', id })}
              disableInteractions={!isAuthenticated}
            />
          </div>
        )}
        {socialNav.page === 'profile' && (
          <div style={{ maxWidth: 1200, margin: '0 auto', padding: '0 1rem' }}>
            <UserProfile
              userId={socialNav.id}
              onBack={() => setSocialNav({ page: 'feed', id: null })}
              onSelectComposition={(id) => setSocialNav({ page: 'detail', id })}
              onSelectUser={(id) => setSocialNav({ page: 'profile', id })}
            />
          </div>
        )}
      </div>
    );
  }

  // Projects tab — always shows project manager
  if (appView === 'projects') {
    return (
      <div style={{ minHeight: '100vh', background: 'var(--color-bg-dark)' }}>
        {navBar}
        {overlays}
        <ProjectManager onContinue={() => setAppView('compose')} />
      </div>
    );
  }

  // Compose/Editor tab — show editor if project open, otherwise project selection
  if (appView === 'compose') {
    if (!currentProject) {
      return (
        <div style={{ minHeight: '100vh', background: 'var(--color-bg-dark)' }}>
          {navBar}
          {overlays}
          <ProjectManager onContinue={() => setAppView('compose')} />
        </div>
      );
    }
    return (
      <div
        style={{
          minHeight: '100vh',
          background: 'var(--color-bg-dark)',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {navBar}
        {overlays}
        <MainApp
          onChangeProject={() => selectProject(null)}
          onLogout={logout}
        />
      </div>
    );
  }
}

function MainApp({ onChangeProject, onLogout }) {
  const { currentProject, saveProjectState, loadProjectState } = useProject();
  const {
    // parsedMidiData,
    instruments,
    instrumentTrackMap,
    longestNotes,
    onMidiProcessed: processMidiData,
    clearMidiState,
  } = useMidiProcessing();
  const {
    videoFiles,
    setVideoFiles,
    recordedVideosCount,
    setRecordedVideosCount,
    instrumentVideos,
    setInstrumentVideos,
    isReadyToCompose,
    setIsReadyToCompose,
    isAudioContextReady,
    error,
    startAudioContext,
  } = useVideoRecording(instruments);

  const [parsedMidiData, setParsedMidiData] = useState(null);
  const [midiParseError, setMidiParseError] = useState(null);
  const [midiFile, setMidiFile] = useState(null);
  const [gridArrangement, setGridArrangement] = useState({});
  const [trackVolumes, setTrackVolumes] = useState({});
  const [muteStates, setMuteStates] = useState({});
  const [compositionStyle, setCompositionStyle] = useState(() => ({
    ...DEFAULT_COMPOSITION_STYLE,
  }));
  const [backgroundAsset, setBackgroundAsset] = useState(null);
  const [clipStyles, setClipStyles] = useState({}); // keyed by item.id (e.g. 'drum-drum_snare_drum')
  const [soloTrack, setSoloTrack] = useState(null);
  const [activeLevels, setActiveLevels] = useState({});
  const lastMeterStateRef = useRef(0);

  const [leftPanelOpen, setLeftPanelOpen] = useState(true);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  // Currently open recording modal target (instrument object or null)
  const [recordingTarget, setRecordingTarget] = useState(null);
  // Preview playback state — synced to grid video overlays
  const [isPreviewPlaying, setIsPreviewPlaying] = useState(false);

  // Track which instrument keys have already been queued for pre-caching
  // so we don't send duplicate requests on every re-render.
  const precachedKeysRef = useRef(new Set());

  // Project-scoped persistence: clip list, blob cache, state restore & save
  const { savedClipKeys, setSavedClipKeys, clipBlobCache } = useProjectSync({
    currentProject,
    instruments,
    midiFile,
    gridArrangement,
    trackVolumes,
    compositionStyle,
    clipStyles,
    loadProjectState,
    saveProjectState,
    toInstrumentKey,
    precachedKeysRef,
    setMidiFile,
    setGridArrangement,
    setTrackVolumes,
    setCompositionStyle,
    setClipStyles,
    setVideoFiles,
    setInstrumentVideos,
    setBackgroundAsset,
  });

  useEffect(
    () => () => {
      if (backgroundAsset?.url?.startsWith('blob:')) {
        try {
          URL.revokeObjectURL(backgroundAsset.url);
        } catch {
          /* ignore */
        }
      }
    },
    [backgroundAsset?.url],
  );

  // Export/import state
  const [exportLoading, setExportLoading] = useState(false);
  const [importLoading, setImportLoading] = useState(false);
  const importInputRef = useRef(null);

  // ── Undo / Redo ───────────────────────────────────────────────────────────
  const {
    snapshot: undoSnapshot,
    canUndo,
    canRedo,
    pushSnapshot,
    undo: undoHistory,
    redo: redoHistory,
    reset: resetHistory,
    isProgrammaticRef,
  } = useUndoRedo({
    gridArrangement,
    compositionStyle,
    clipStyles,
    trackVolumes,
    muteStates,
    soloTrack,
  });

  const pushUndoDebounceRef = useRef(null);

  // Push a debounced snapshot on every relevant state change.
  useEffect(() => {
    if (isProgrammaticRef.current) {
      isProgrammaticRef.current = false;
      return;
    }
    clearTimeout(pushUndoDebounceRef.current);
    pushUndoDebounceRef.current = setTimeout(() => {
      pushSnapshot({
        gridArrangement,
        compositionStyle,
        clipStyles,
        trackVolumes,
        muteStates,
        soloTrack,
      });
    }, 400);
  }, [
    gridArrangement,
    compositionStyle,
    clipStyles,
    trackVolumes,
    muteStates,
    soloTrack,
  ]); // eslint-disable-line react-hooks/exhaustive-deps

  // Apply snapshot when undo/redo changes it.
  const prevSnapshotRef = useRef(undoSnapshot);
  useEffect(() => {
    if (undoSnapshot === prevSnapshotRef.current) return;
    prevSnapshotRef.current = undoSnapshot;
    if (!isProgrammaticRef.current) return;
    setGridArrangement(undoSnapshot.gridArrangement);
    setCompositionStyle(undoSnapshot.compositionStyle);
    setClipStyles(undoSnapshot.clipStyles);
    setTrackVolumes(undoSnapshot.trackVolumes);
    setMuteStates(undoSnapshot.muteStates);
    setSoloTrack(undoSnapshot.soloTrack);
  }, [undoSnapshot]); // eslint-disable-line react-hooks/exhaustive-deps

  // Reset undo history when switching projects (prevents undo into a previous project's state).
  useEffect(() => {
    resetHistory({
      gridArrangement: {},
      compositionStyle: { ...DEFAULT_COMPOSITION_STYLE },
      clipStyles: {},
      trackVolumes: {},
      muteStates: {},
      soloTrack: null,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentProject?.id]);

  // ─────────────────────────────────────────────────────────────────────────

  // Fire-and-forget pre-cache request for one instrument's blob + MIDI notes.
  const triggerPrecache = useCallback((instrumentKey, blob, midiData) => {
    if (precachedKeysRef.current.has(instrumentKey)) return;
    precachedKeysRef.current.add(instrumentKey);

    // Collect unique MIDI notes for this instrument key
    const notes = new Set();
    midiData.tracks.forEach((track) => {
      if (isDrumTrack(track)) {
        // Match drum notes whose group maps to this key
        const expectedKey = `drum_${getNoteGroup(track.notes[0]?.midi ?? 0)
          .toLowerCase()
          .replace(/\s+/g, '_')}`;
        track.notes.forEach((note) => {
          const noteKey = `drum_${getNoteGroup(note.midi).toLowerCase().replace(/\s+/g, '_')}`;
          if (noteKey === instrumentKey) notes.add(note.midi);
        });
      } else {
        const trackKey = track.instrument?.name
          ?.toLowerCase()
          .replace(/\s+/g, '_');
        if (trackKey === instrumentKey) {
          track.notes.forEach((note) => notes.add(note.midi));
        }
      }
    });

    if (notes.size === 0) return;

    const formData = new FormData();
    formData.append('video', blob, `${instrumentKey}.mp4`);
    formData.append('midiNotes', JSON.stringify([...notes]));

    // Include auth token and project scope
    const token = localStorage.getItem('auth_token');
    const projectId = (() => {
      try {
        return JSON.parse(localStorage.getItem('current_project'))?.id;
      } catch {
        return null;
      }
    })();
    const url = projectId
      ? `http://localhost:3000/api/autotune/precache?projectId=${projectId}`
      : 'http://localhost:3000/api/autotune/precache';
    const headers = token ? { Authorization: `Bearer ${token}` } : {};

    fetch(url, { method: 'POST', headers, body: formData })
      .then((r) => {
        if (!r.ok) throw new Error(`precache HTTP ${r.status}`);
        console.log(`[precache] Queued ${instrumentKey} (${notes.size} notes)`);
      })
      .catch((err) => console.warn(`[precache] ${instrumentKey} failed:`, err));
  }, []);

  // Trigger precache whenever a new video is recorded AND MIDI is loaded,
  // or when MIDI loads after videos are already recorded.
  useEffect(() => {
    if (!parsedMidiData || Object.keys(videoFiles).length === 0) return;
    for (const [key, blob] of Object.entries(videoFiles)) {
      if (blob instanceof Blob) {
        triggerPrecache(key, blob, parsedMidiData);
      }
    }
  }, [parsedMidiData, videoFiles, triggerPrecache]);

  const handleVolumeChange = (trackKey, volume) => {
    setTrackVolumes((prev) => ({
      ...prev,
      [trackKey]: volume,
    }));
  };

  // Throttled meter update — called up to ~15 Hz from PreviewPlayer's rAF loop.
  // We gate state updates to ~10 Hz here to avoid excessive re-renders.
  const handleMeterUpdate = useCallback((levels) => {
    const now = Date.now();
    if (now - lastMeterStateRef.current < 100) return;
    lastMeterStateRef.current = now;
    setActiveLevels(levels);
  }, []);

  const handleMuteChange = (trackKey, isMuted) => {
    setMuteStates((prev) => ({ ...prev, [trackKey]: isMuted }));
  };

  const handleSoloChange = (trackKey) => {
    setSoloTrack((prev) => (prev === trackKey ? null : trackKey));
  };

  const handleMidiProcessed = (file) => {
    setMidiFile(file);
  };

  const handleParsedMidi = useCallback(
    (midiInfo) => {
      setMidiParseError(null);

      // Clear all in-memory clips so stale clips from a previous MIDI don't bleed through.
      // The instruments effect will re-populate from clipBlobCache for matching instruments.
      setInstrumentVideos((prev) => {
        Object.values(prev).forEach((url) => {
          try {
            URL.revokeObjectURL(url);
          } catch {
            /* ignore */
          }
        });
        return {};
      });
      setVideoFiles({});
      precachedKeysRef.current = new Set();

      setParsedMidiData(midiInfo);
      processMidiData(midiInfo);
    },
    [processMidiData],
  );

  const handleMidiParseError = useCallback(
    (err) => {
      console.error('[MidiParser] Parse error:', err);
      // Clear all MIDI-derived state so stale data can't be used for composition
      setParsedMidiData(null);
      setMidiFile(null);
      clearMidiState();
      setMidiParseError(err.message || 'Failed to parse MIDI file');
    },
    [clearMidiState],
  );

  // Add handleRecordingComplete function
  const handleRecordingComplete = useCallback(
    (blob, instrument) => {
      if (!(blob instanceof Blob)) {
        console.error('Invalid blob:', blob);
        return;
      }
      // toInstrumentKey already reads instrument.group for drums — no mutation needed
      const key = toInstrumentKey(instrument);

      console.log(
        'Recording complete for instrument:',
        key,
        'blob size:',
        blob.size,
      );

      setVideoFiles((prev) => ({ ...prev, [key]: blob }));
      clipBlobCache.current[key] = blob;

      // Persist clip to server for restore on next project open
      if (currentProject) {
        uploadClip(currentProject.id, key, blob)
          .then(() => setSavedClipKeys((prev) => new Set([...prev, key])))
          .catch((err) =>
            console.warn(`[clips] Failed to upload clip for ${key}:`, err),
          );
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    },
    [currentProject?.id],
  );

  const handleVideoReady = useCallback((videoUrl, instrument) => {
    const instrumentKey = toInstrumentKey(instrument);

    setInstrumentVideos((prev) => {
      // Revoke the old URL for this key before overwriting
      if (prev[instrumentKey] && prev[instrumentKey] !== videoUrl) {
        try {
          URL.revokeObjectURL(prev[instrumentKey]);
        } catch {
          /* ignore */
        }
      }
      return { ...prev, [instrumentKey]: videoUrl };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleBackgroundUpload = useCallback(
    async (file) => {
      if (!currentProject?.id) {
        throw new Error('Select a project before uploading a background');
      }

      const { background } = await uploadProjectBackground(currentProject.id, file);
      setBackgroundAsset({
        blob: file,
        url: URL.createObjectURL(file),
        kind: background.kind,
        mimeType: background.mimeType,
        originalName: background.originalName,
      });
      setCompositionStyle((prev) => ({
        ...prev,
        backgroundMode: background.kind,
        backgroundMedia: {
          ...background,
          saved: true,
        },
      }));
    },
    [currentProject?.id],
  );

  const handleBackgroundRemove = useCallback(async () => {
    if (!currentProject?.id) {
      throw new Error('Select a project before removing a background');
    }

    await deleteProjectBackground(currentProject.id);
    setBackgroundAsset(null);
    setCompositionStyle((prev) => ({
      ...prev,
      backgroundMode: 'color',
      backgroundMedia: null,
    }));
  }, [currentProject?.id]);

  // Add click handler to initialize audio context
  useEffect(() => {
    const handleClick = () => {
      if (!isAudioContextReady) {
        startAudioContext();
      }
    };

    document.addEventListener('click', handleClick);
    return () => document.removeEventListener('click', handleClick);
  }, [isAudioContextReady, startAudioContext]);

  const handleExport = useCallback(async () => {
    if (!currentProject) return;
    setExportLoading(true);
    try {
      await downloadProjectExport(currentProject.id, currentProject.name);
    } catch (err) {
      alert(`Export failed: ${err.message}`);
    } finally {
      setExportLoading(false);
    }
  }, [currentProject]);

  const handleImport = useCallback(
    async (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      e.target.value = '';
      setImportLoading(true);
      try {
        const { project } = await importProjectFromZip(file);
        onChangeProject(project);
      } catch (err) {
        alert(`Import failed: ${err.message}`);
      } finally {
        setImportLoading(false);
      }
    },
    [onChangeProject],
  );

  return (
    <div className='editor-shell'>
      {/* Non-rendering helpers always present */}
      {midiFile && (
        <MidiParser
          file={midiFile}
          onParsed={handleParsedMidi}
          onError={handleMidiParseError}
        />
      )}
      {midiParseError && (
        <div
          role='alert'
          className='mx-4 mt-2 p-3 bg-red-100 border border-red-300 text-red-800 rounded flex items-start gap-2 text-sm'
        >
          <span className='flex-1'>⚠️ MIDI parse error: {midiParseError}</span>
          <button
            onClick={() => setMidiParseError(null)}
            className='shrink-0 text-red-600 hover:text-red-900 font-bold'
            aria-label='Dismiss MIDI error'
          >
            ✕
          </button>
        </div>
      )}

      {/* ── Top bar ──────────────────────────────────────────────── */}
      <div className='editor-topbar'>
        {parsedMidiData ? (
          <>
            <span className='editor-topbar__midi-pill'>
              🎵 {midiFile?.name?.replace(/\.midi?$/i, '') || 'MIDI loaded'}
            </span>
            <div className='editor-topbar__sep' />
            <MidiUploader onMidiProcessed={handleMidiProcessed} compact />
          </>
        ) : (
          <MidiUploader onMidiProcessed={handleMidiProcessed} compact />
        )}
        <div className='editor-topbar__spacer' />

        {/* Undo / Redo */}
        <button
          className='editor-topbar__icon-btn'
          onClick={undoHistory}
          disabled={!canUndo}
          title='Undo (Ctrl+Z)'
          aria-label='Undo'
        >
          <Undo2 size={16} />
          <span className='editor-topbar__icon-label'>Undo</span>
        </button>
        <button
          className='editor-topbar__icon-btn'
          onClick={redoHistory}
          disabled={!canRedo}
          title='Redo (Ctrl+Y)'
          aria-label='Redo'
        >
          <Redo2 size={16} />
          <span className='editor-topbar__icon-label'>Redo</span>
        </button>

        {/* Export / Import */}
        {currentProject && (
          <>
            <button
              className='editor-topbar__icon-btn'
              onClick={handleExport}
              disabled={exportLoading}
              title='Export project as ZIP'
              aria-label='Export project'
            >
              <Download size={16} />
              <span className='editor-topbar__icon-label'>Export</span>
            </button>
            <button
              className='editor-topbar__icon-btn'
              onClick={() => importInputRef.current?.click()}
              disabled={importLoading}
              title='Import project from ZIP'
              aria-label='Import project'
            >
              <Upload size={16} />
              <span className='editor-topbar__icon-label'>Import</span>
            </button>
            <input
              ref={importInputRef}
              type='file'
              accept='.zip'
              style={{ display: 'none' }}
              onChange={handleImport}
            />
          </>
        )}
      </div>

      {/* ── 3-panel body ─────────────────────────────────────────── */}
      <div className='editor-body'>
        {/* LEFT: Instrument sidebar */}
        <div
          className={`editor-left${leftPanelOpen ? '' : ' editor-left--collapsed'}`}
        >
          <InstrumentSidebar
            instruments={instruments}
            instrumentVideos={instrumentVideos}
            longestNotes={longestNotes}
            onRecordClick={setRecordingTarget}
            isOpen={leftPanelOpen}
            onToggle={() => setLeftPanelOpen((v) => !v)}
          />
        </div>

        {/* CENTER: Grid canvas or empty state */}
        <div className='editor-center'>
          {parsedMidiData ? (
            <>
              <MidiInfoDisplay midiData={parsedMidiData} />

              {!isReadyToCompose && instruments.length > 0 && (
                <ProgressBar
                  current={recordedVideosCount}
                  total={instruments.length}
                />
              )}

              <Grid
                midiData={parsedMidiData}
                onArrangementChange={setGridArrangement}
                initialArrangement={gridArrangement}
                compositionStyle={compositionStyle}
                backgroundAsset={backgroundAsset}
                clipStyles={clipStyles}
                instrumentVideos={instrumentVideos}
                isPreviewPlaying={isPreviewPlaying}
                activeLevels={activeLevels}
                onClipStyleChange={(itemId, newStyle) =>
                  setClipStyles((prev) => ({
                    ...prev,
                    [itemId]: {
                      ...DEFAULT_CLIP_STYLE,
                      ...prev[itemId],
                      ...newStyle,
                    },
                  }))
                }
              />

              {instruments.length > 0 && (
                <CompositionSection
                  videoFiles={videoFiles}
                  midiData={parsedMidiData}
                  instrumentTrackMap={instrumentTrackMap}
                  gridArrangement={gridArrangement}
                  trackVolumes={trackVolumes}
                  muteStates={muteStates}
                  soloTrack={soloTrack}
                  compositionStyle={compositionStyle}
                  clipStyles={clipStyles}
                  projectName={currentProject?.name || ''}
                />
              )}
            </>
          ) : (
            <div className='editor-empty'>
              <span className='editor-empty__title'>🎵 AutoTune Syncer</span>
              <span className='editor-empty__sub'>
                Drop a MIDI file above or click &quot;Load MIDI&quot; to get started
              </span>
            </div>
          )}
        </div>

        {/* RIGHT: Style + Mix panel */}
        <RightPanel
          isOpen={rightPanelOpen}
          onToggle={() => setRightPanelOpen((v) => !v)}
          compositionStyle={compositionStyle}
          onStyleChange={setCompositionStyle}
          backgroundAsset={backgroundAsset}
          onBackgroundUpload={handleBackgroundUpload}
          onBackgroundRemove={handleBackgroundRemove}
          instruments={instruments}
          volumes={trackVolumes}
          muteStates={muteStates}
          soloTrack={soloTrack}
          onVolumeChange={handleVolumeChange}
          onMuteChange={handleMuteChange}
          onSoloChange={handleSoloChange}
          activeLevels={activeLevels}
          midiData={parsedMidiData}
          videoFiles={videoFiles}
          onMeterUpdate={handleMeterUpdate}
          onPlayStateChange={setIsPreviewPlaying}
          isPreviewPlaying={isPreviewPlaying}
        />
      </div>

      {/* Recording modal — portal rendered to document.body */}
      {recordingTarget && (
        <RecordingModal
          instrument={recordingTarget}
          instrumentVideos={instrumentVideos}
          longestNotes={longestNotes}
          midiData={parsedMidiData}
          onRecordingComplete={handleRecordingComplete}
          onVideoReady={handleVideoReady}
          onClose={() => setRecordingTarget(null)}
        />
      )}
    </div>
  );
}

MainApp.propTypes = {
  onChangeProject: PropTypes.func.isRequired,
  onLogout: PropTypes.func.isRequired,
};

export default App;
