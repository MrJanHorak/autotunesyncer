/* eslint-disable no-unused-vars */
import { useEffect, useCallback, useState, useRef } from 'react';
import PropTypes from 'prop-types';
import { Film, Music, Grid3x3, FolderOpen, LogOut } from 'lucide-react';

import { isDrumTrack, DRUM_NOTES, getNoteGroup } from './js/drumUtils';
import InstrumentList from './components/InstrumentList/InstrumentList';

import { useMidiProcessing } from './hooks/useMidiProcessing';
import { useVideoRecording } from './hooks/useVideoRecording';
import { useAuth } from './context/AuthContext';
import { useProject } from './context/ProjectContext';
import {
  configureApiService,
  apiFetch,
  uploadClip,
} from './services/apiService';

// Components
import AuthPage from './components/Auth/AuthPage';
import LandingPage from './components/LandingPage';
import ProjectManager from './components/Projects/ProjectManager';
import MidiUploader from './components/MidiUploader/';
import MidiInfoDisplay from './components/MidiInfoDisplay/MidiInfoDisplay';
import RecordingSection from './components/RecordingSection/RecordingSection';
import AudioContextInitializer from './components/AudioContextInitializer/AudioContextInitializer';
import CompositionSection from './components/CompositionSection/CompositionSection';
import MidiParser from './components/MidiParser/MidiParser';
import ProgressBar from './components/ProgressBar/ProgressBar';
import Grid from './components/Grid/Grid';
import Mixer from './components/Mixer/Mixer';
import PreviewPlayer from './components/PreviewPlayer/PreviewPlayer';

// Social components
import SocialFeed from './components/Social/SocialFeed.jsx';
import CompositionDetail from './components/Social/CompositionDetail.jsx';
import UserProfile from './components/Social/UserProfile.jsx';
import './components/Social/Social.css';

import './App.css';

const normalizeInstrumentName = (name) => name.toLowerCase().replace(/\s+/g, '_');

const toInstrumentKey = (instrument) => {
  if (instrument.isDrum) {
    const name = (instrument.group || instrument.name || '').toLowerCase().replace(/\s+/g, '_');
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

  // Auth modal state for guests
  const [showAuth, setShowAuth] = useState(false);

  // Auth modal wrapper
  function AuthPageModal({ onClose }) {
    return (
      <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', background: 'rgba(0,0,0,0.7)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ position: 'relative', background: '#181828', borderRadius: 12, padding: 32, minWidth: 340 }}>
          <button onClick={onClose} style={{ position: 'absolute', top: 12, right: 12, background: 'none', border: 'none', color: '#fff', fontSize: 22, cursor: 'pointer' }}>×</button>
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
            <span className='app-nav__brand-icon'><Film size={28} /></span>
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
        onClick={() => { setAppView('compose'); setSocialNav({ page: 'feed', id: null }); }}
      >
        <span className='app-nav__brand-icon'><Film size={28} /></span>
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
          onClick={() => { setAppView('feed'); setSocialNav({ page: 'feed', id: null }); }}
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
        {user && (
          <span className='app-nav__user'>
            <span className='app-nav__avatar'>
              {user.profileImageUrl
                ? <img src={user.profileImageUrl} alt='profile' />
                : user.username?.[0]?.toUpperCase() || 'U'
              }
            </span>
            @{user.username}
          </span>
        )}
        {appView === 'compose' && currentProject && (
          <button className='app-nav__btn' onClick={() => selectProject(null)}>
            <FolderOpen size={15} /> {currentProject.name}
          </button>
        )}
        <button className='app-nav__btn' onClick={logout}>
          <LogOut size={15} /> Sign Out
        </button>
      </div>
    </nav>
  );

  if (appView === 'feed') {
    return (
      <div style={{ minHeight: '100vh', background: 'var(--social-bg)' }}>
        {navBar}
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

  // Compose view — requires project selection
  if (!currentProject) {
    return (
      <div
        style={{
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
        }}
      >
        {navBar}
        <ProjectManager />
      </div>
    );
  }

  return (
    <div
      style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
      }}
    >
      {navBar}
      <MainApp onChangeProject={() => selectProject(null)} onLogout={logout} />
    </div>
  );
}

function MainApp({ onChangeProject, onLogout }) {
  const { currentProject, saveProjectState, loadProjectState } = useProject();
  const {
    // parsedMidiData,
    instruments,
    instrumentTrackMap,
    longestNotes,
    onMidiProcessed: processMidiData,
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
    audioContextStarted,
    isAudioContextReady,
    error,
    startAudioContext,
  } = useVideoRecording(instruments);

  const [parsedMidiData, setParsedMidiData] = useState(null);
  const [midiFile, setMidiFile] = useState(null);
  const [gridArrangement, setGridArrangement] = useState({});
  const [trackVolumes, setTrackVolumes] = useState({});
  const [muteStates, setMuteStates] = useState({});
  const [soloTrack, setSoloTrack] = useState(null);
  const [activeLevels, setActiveLevels] = useState({});
  const lastMeterStateRef = useRef(0);

  // Persisted clip keys from server (instrument keys that have saved clips)
  const [savedClipKeys, setSavedClipKeys] = useState(new Set());
  // In-memory blob cache to avoid re-fetching on MIDI change within same project
  const clipBlobCache = useRef({});
  // Version counter — incremented on project switch to discard stale fetches
  const clipsLoadingVersion = useRef(0);

  // Track which instrument keys have already been queued for pre-caching
  // so we don't send duplicate requests on every re-render.
  const precachedKeysRef = useRef(new Set());

  // ── Project clip persistence ──────────────────────────────────────────────

  // On project change: load saved clip list from server + restore MIDI from state.
  useEffect(() => {
    if (!currentProject) {
      setSavedClipKeys(new Set());
      clipBlobCache.current = {};
      return;
    }

    const version = ++clipsLoadingVersion.current;

    apiFetch(`/projects/${currentProject.id}/clips`)
      .then((r) => r.json())
      .then(({ clips }) => {
        if (clipsLoadingVersion.current !== version) return;
        setSavedClipKeys(new Set(clips.map((c) => c.instrument_key)));
      })
      .catch((err) => console.warn('[clips] Failed to load clip list:', err));

    loadProjectState(currentProject.id)
      .then((state) => {
        if (clipsLoadingVersion.current !== version) return;
        if (state?.midiFileBase64) {
          const file = base64ToFile(
            state.midiFileBase64,
            state.midiFileName || 'project.mid',
          );
          setMidiFile(file);
        }
      })
      .catch((err) =>
        console.warn('[clips] Failed to load project state:', err),
      );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentProject?.id]);

  // When instruments load or savedClipKeys changes: lazily fetch blobs for matching clips.
  useEffect(() => {
    if (!instruments.length || !savedClipKeys.size || !currentProject) return;
    const version = clipsLoadingVersion.current;
    const projectId = currentProject.id;

    for (const instrument of instruments) {
      const key = toInstrumentKey(instrument);
      if (!savedClipKeys.has(key)) continue;

      if (clipBlobCache.current[key]) {
        setVideoFiles((prev) =>
          prev[key] ? prev : { ...prev, [key]: clipBlobCache.current[key] },
        );
        setInstrumentVideos((prev) =>
          prev[key]
            ? prev
            : {
                ...prev,
                [key]: URL.createObjectURL(clipBlobCache.current[key]),
              },
        );
        continue;
      }

      apiFetch(`/projects/${projectId}/clips/${encodeURIComponent(key)}/file`)
        .then((r) => {
          if (clipsLoadingVersion.current !== version) return null;
          return r.blob();
        })
        .then((blob) => {
          if (!blob || clipsLoadingVersion.current !== version) return;
          clipBlobCache.current[key] = blob;
          setVideoFiles((prev) =>
            prev[key] ? prev : { ...prev, [key]: blob },
          );
          setInstrumentVideos((prev) =>
            prev[key] ? prev : { ...prev, [key]: URL.createObjectURL(blob) },
          );
        })
        .catch((err) =>
          console.warn(`[clips] Failed to fetch clip for ${key}:`, err),
        );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [instruments, savedClipKeys, currentProject?.id]);

  // When MIDI file changes: persist it to project state for restore on refresh.
  useEffect(() => {
    if (!midiFile || !currentProject) return;
    const projectId = currentProject.id;
    const reader = new FileReader();
    reader.onload = async () => {
      try {
        const currentState = await loadProjectState(projectId).catch(
          () => null,
        );
        await saveProjectState({
          ...(currentState || {}),
          midiFileBase64: reader.result,
          midiFileName: midiFile.name,
        });
      } catch (err) {
        console.warn('[clips] Failed to save MIDI to project state:', err);
      }
    };
    reader.readAsDataURL(midiFile);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [midiFile, currentProject?.id]);

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
      console.log('Parsed MIDI info:', midiInfo);

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

  // Add handleRecordingComplete function
  const handleRecordingComplete = useCallback(
    (blob, instrument) => {
      if (!(blob instanceof Blob)) {
        console.error('Invalid blob:', blob);
        return;
      }
      if (instrument.isDrum) {
        instrument.name = instrument.group;
      }
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

  // Add handleVideoReady function
  const handleVideoReady = useCallback((videoUrl, instrument) => {
    if (instrument.isDrum) instrument.name = instrument.group;
    const instrumentKey = toInstrumentKey(instrument);

    setInstrumentVideos((prev) => ({
      ...prev,
      [instrumentKey]: videoUrl,
    }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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

  return (
    <div className='app-container'>
      <AudioContextInitializer
        audioContextStarted={audioContextStarted}
        onInitialize={startAudioContext}
      />
      <MidiUploader onMidiProcessed={handleMidiProcessed} />

      {midiFile && <MidiParser file={midiFile} onParsed={handleParsedMidi} />}

      {parsedMidiData && (
        <>
          <MidiInfoDisplay midiData={parsedMidiData} />
          {instruments.length > 0 && (
            <InstrumentList instruments={instruments} />
          )}

          {instruments.length > 0 && (
            <div
              className='audio-control-section'
              style={{
                margin: '20px 0',
                padding: '20px',
                background: '#f5f5f5',
                borderRadius: '8px',
              }}
            >
              <Mixer
                instruments={instruments}
                volumes={trackVolumes}
                onVolumeChange={handleVolumeChange}
                muteStates={muteStates}
                soloTrack={soloTrack}
                onMuteChange={handleMuteChange}
                onSoloChange={handleSoloChange}
                activeLevels={activeLevels}
              />

              <div style={{ marginTop: '15px' }}>
                <PreviewPlayer
                  midiData={parsedMidiData}
                  videoFiles={videoFiles}
                  volumes={trackVolumes}
                  muteStates={muteStates}
                  soloTrack={soloTrack}
                  instruments={instruments}
                  onMeterUpdate={handleMeterUpdate}
                />
              </div>
            </div>
          )}

          <Grid
            midiData={parsedMidiData}
            onArrangementChange={setGridArrangement}
          />

          {!isReadyToCompose && instruments.length > 0 && (
            <ProgressBar
              current={recordedVideosCount}
              total={instruments.length}
            />
          )}

          <RecordingSection
            instruments={instruments}
            longestNotes={longestNotes}
            onRecordingComplete={handleRecordingComplete}
            onVideoReady={handleVideoReady}
            instrumentVideos={instrumentVideos}
            midiData={parsedMidiData}
          />
        </>
      )}

      {isReadyToCompose && instruments.length > 0 && (
        <CompositionSection
          videoFiles={videoFiles}
          midiData={parsedMidiData}
          instrumentTrackMap={instrumentTrackMap}
          gridArrangement={gridArrangement}
          trackVolumes={trackVolumes}
          muteStates={muteStates}
          soloTrack={soloTrack}
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
