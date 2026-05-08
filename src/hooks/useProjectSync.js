import { useState, useRef, useEffect, useCallback } from 'react';
import {
  DEFAULT_COMPOSITION_STYLE,
  DEFAULT_CLIP_STYLE,
} from '../js/styleDefaults';
import { apiFetch, fetchProjectBackgroundFile } from '../services/apiService';
import {
  hasGridArrangement,
  normalizeGridArrangement,
} from '../../shared/gridLayout.js';
import {
  DEFAULT_RENDER_PRESET,
  normalizeRenderPreset,
} from '../../shared/renderPresets.js';

const PROJECT_STATE_SCHEMA_VERSION = 2;

const normalizeSavedClipStyle = (savedStyle = {}, schemaVersion = 0) => {
  const normalizedStyle = { ...DEFAULT_CLIP_STYLE, ...savedStyle };

  if (
    Number(schemaVersion || 0) < PROJECT_STATE_SCHEMA_VERSION &&
    savedStyle?.roundedCorners === false &&
    savedStyle?.roundedCornersConfigured !== true
  ) {
    // Legacy projects rendered rounded in the grid despite the saved false
    // default, so keep that visible behavior when loading older state.
    normalizedStyle.roundedCorners = true;
  }

  return normalizedStyle;
};

/**
 * Manages all project-scoped persistence side-effects:
 *   1. On project switch: load clip list + restore saved state (MIDI, grid, style, etc.)
 *   2. When instruments/savedClipKeys change: lazily fetch clip blobs
 *   3. When midiFile changes: persist MIDI base64 to project state
 *   4. When arrangement/style changes: debounced save to project state
 *
 * Returns { savedClipKeys, setSavedClipKeys, clipBlobCache } so callers can
 * react to clip availability and share the blob cache for recording uploads.
 */
export function useProjectSync({
  currentProject,
  instruments,
  midiFile,
  gridArrangement,
  trackVolumes,
  renderPreset,
  compositionStyle,
  clipStyles,
  loadProjectState,
  saveProjectState,
  toInstrumentKey,
  precachedKeysRef,
  setMidiFile,
  setGridArrangement,
  setTrackVolumes,
  setRenderPreset,
  setCompositionStyle,
  setClipStyles,
  setVideoFiles,
  setInstrumentVideos,
  setBackgroundAsset,
  onProjectConflict,
}) {
  const [savedClipKeys, setSavedClipKeys] = useState(new Set());
  const clipBlobCache = useRef({});
  const clipsLoadingVersion = useRef(0);
  const saveArrangementTimeoutRef = useRef(null);
  // In-memory shadow of the last saved project state — avoids GET-before-POST on every save
  const shadowStateRef = useRef(null);
  const suspendPersistenceUntilRef = useRef(0);

  const suspendPersistence = useCallback((durationMs = 1500) => {
    suspendPersistenceUntilRef.current = Date.now() + durationMs;
  }, []);

  const isPersistenceSuspended = useCallback(
    () => Date.now() < suspendPersistenceUntilRef.current,
    [],
  );

  const applyLoadedState = useCallback(
    async (state, version) => {
      suspendPersistence();
      shadowStateRef.current = state || {};

      const stateSchemaVersion = Number(state?.schemaVersion || 0);
      setRenderPreset(normalizeRenderPreset(state?.renderPreset));

      if (state?.midiFileBase64) {
        const [header, data] = state.midiFileBase64.split(',');
        const mime = header.match(/:(.*?);/)?.[1] || 'audio/midi';
        const bytes = atob(data);
        const arr = new Uint8Array(bytes.length);
        for (let index = 0; index < bytes.length; index += 1) {
          arr[index] = bytes.charCodeAt(index);
        }
        const file = new File([arr], state.midiFileName || 'project.mid', {
          type: mime,
        });
        setMidiFile(file);
      } else {
        setMidiFile(null);
      }

      if (hasGridArrangement(state?.gridArrangement)) {
        setGridArrangement(normalizeGridArrangement(state.gridArrangement));
      } else {
        setGridArrangement({});
      }

      setTrackVolumes(
        state?.trackVolumes && Object.keys(state.trackVolumes).length > 0
          ? state.trackVolumes
          : {},
      );

      setCompositionStyle({
        ...DEFAULT_COMPOSITION_STYLE,
        ...(state?.compositionStyle || {}),
      });

      const savedBackground = state?.compositionStyle?.backgroundMedia;
      const backgroundMode = state?.compositionStyle?.backgroundMode;
      if (
        savedBackground?.saved &&
        backgroundMode &&
        backgroundMode !== 'color'
      ) {
        try {
          const blob = await fetchProjectBackgroundFile(currentProject.id);
          if (!blob || clipsLoadingVersion.current !== version) return state;
          setBackgroundAsset({
            blob,
            url: URL.createObjectURL(blob),
            kind:
              savedBackground.kind ||
              (blob.type.startsWith('video/') ? 'video' : 'image'),
            mimeType: savedBackground.mimeType || blob.type,
            originalName: savedBackground.originalName || 'project background',
          });
        } catch (err) {
          console.warn('[background] Failed to load project background:', err);
          setBackgroundAsset(null);
        }
      } else {
        setBackgroundAsset(null);
      }

      if (state?.clipStyles && Object.keys(state.clipStyles).length > 0) {
        setClipStyles(
          Object.fromEntries(
            Object.entries(state.clipStyles).map(([id, saved]) => [
              id,
              normalizeSavedClipStyle(saved, stateSchemaVersion),
            ]),
          ),
        );
      } else {
        setClipStyles({});
      }

      return state;
    },
    [
      currentProject?.id,
      setBackgroundAsset,
      setClipStyles,
      setCompositionStyle,
      setGridArrangement,
      setMidiFile,
      setRenderPreset,
      setTrackVolumes,
      suspendPersistence,
    ],
  );

  const refreshSavedClipKeys = useCallback(async (projectId, version) => {
    try {
      const response = await apiFetch(`/projects/${projectId}/clips`);
      const { clips } = await response.json();
      if (clipsLoadingVersion.current !== version) return;
      setSavedClipKeys(new Set(clips.map((clip) => clip.instrument_key)));
    } catch (err) {
      console.warn('[clips] Failed to load clip list:', err);
    }
  }, []);

  const reloadProjectState = useCallback(async () => {
    if (!currentProject) return null;

    const version = clipsLoadingVersion.current;
    await refreshSavedClipKeys(currentProject.id, version);

    const state = await loadProjectState(currentProject.id);
    if (clipsLoadingVersion.current !== version) return null;

    return applyLoadedState(state, version);
  }, [
    applyLoadedState,
    currentProject,
    loadProjectState,
    refreshSavedClipKeys,
  ]);

  // ── 1. On project switch: load clip list + restore state ─────────────────
  useEffect(() => {
    const version = ++clipsLoadingVersion.current;

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
    setBackgroundAsset(null);
    if (precachedKeysRef) precachedKeysRef.current = new Set();
    shadowStateRef.current = null; // reset shadow on project switch

    if (!currentProject) {
      setSavedClipKeys(new Set());
      clipBlobCache.current = {};
      setRenderPreset(DEFAULT_RENDER_PRESET);
      return;
    }

    refreshSavedClipKeys(currentProject.id, version);

    loadProjectState(currentProject.id)
      .then((state) => {
        if (clipsLoadingVersion.current !== version) return;
        void applyLoadedState(state, version);
      })
      .catch((err) =>
        console.warn('[clips] Failed to load project state:', err),
      );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentProject?.id]);

  // ── 2. Lazily fetch blobs for instruments with saved clips ────────────────
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

  // ── 3. Persist MIDI file to project state when it changes ─────────────────
  useEffect(() => {
    if (!midiFile || !currentProject) return;
    if (isPersistenceSuspended()) return;
    const reader = new FileReader();
    reader.onload = async () => {
      try {
        const patch = {
          midiFileBase64: reader.result,
          midiFileName: midiFile.name,
        };
        shadowStateRef.current = {
          ...(shadowStateRef.current || {}),
          ...patch,
        };
        await saveProjectState(shadowStateRef.current);
      } catch (err) {
        if (err?.code === 'PROJECT_CONFLICT' && onProjectConflict) {
          onProjectConflict(err);
          return;
        }
        console.warn('[clips] Failed to save MIDI to project state:', err);
      }
    };
    reader.readAsDataURL(midiFile);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [midiFile, currentProject?.id]);

  // ── 4. Debounced save of arrangement, volumes, and style ──────────────────
  useEffect(() => {
    if (!currentProject || !hasGridArrangement(gridArrangement)) return;
    if (isPersistenceSuspended()) return;
    clearTimeout(saveArrangementTimeoutRef.current);
    saveArrangementTimeoutRef.current = setTimeout(async () => {
      try {
        const patch = {
          schemaVersion: PROJECT_STATE_SCHEMA_VERSION,
          gridArrangement: normalizeGridArrangement(gridArrangement),
          trackVolumes,
          renderPreset: normalizeRenderPreset(renderPreset),
          compositionStyle,
          clipStyles,
        };
        shadowStateRef.current = {
          ...(shadowStateRef.current || {}),
          ...patch,
        };
        await saveProjectState(shadowStateRef.current);
      } catch (err) {
        if (err?.code === 'PROJECT_CONFLICT' && onProjectConflict) {
          onProjectConflict(err);
          return;
        }
        console.warn('[state] Failed to save arrangement:', err);
      }
    }, 1500);
  }, [
    gridArrangement,
    trackVolumes,
    renderPreset,
    compositionStyle,
    clipStyles,
    currentProject?.id,
  ]); // eslint-disable-line react-hooks/exhaustive-deps

  return { savedClipKeys, setSavedClipKeys, clipBlobCache, reloadProjectState };
}
