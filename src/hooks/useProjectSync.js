import { useState, useRef, useEffect } from 'react';
import {
  DEFAULT_COMPOSITION_STYLE,
  DEFAULT_CLIP_STYLE,
} from '../js/styleDefaults';
import { apiFetch, fetchProjectBackgroundFile } from '../services/apiService';
import {
  hasGridArrangement,
  normalizeGridArrangement,
} from '../../shared/gridLayout.js';

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
}) {
  const [savedClipKeys, setSavedClipKeys] = useState(new Set());
  const clipBlobCache = useRef({});
  const clipsLoadingVersion = useRef(0);
  const saveArrangementTimeoutRef = useRef(null);
  // In-memory shadow of the last saved project state — avoids GET-before-POST on every save
  const shadowStateRef = useRef(null);

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
      return;
    }

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
        shadowStateRef.current = state || {}; // seed shadow from server
        if (state?.midiFileBase64) {
          const [header, data] = state.midiFileBase64.split(',');
          const mime = header.match(/:(.*?);/)?.[1] || 'audio/midi';
          const bytes = atob(data);
          const arr = new Uint8Array(bytes.length);
          for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
          const file = new File([arr], state.midiFileName || 'project.mid', {
            type: mime,
          });
          setMidiFile(file);
        }
        if (hasGridArrangement(state?.gridArrangement)) {
          setGridArrangement(normalizeGridArrangement(state.gridArrangement));
        }
        if (state?.trackVolumes && Object.keys(state.trackVolumes).length > 0)
          setTrackVolumes(state.trackVolumes);
        if (state?.compositionStyle)
          setCompositionStyle((prev) => ({
            ...DEFAULT_COMPOSITION_STYLE,
            ...prev,
            ...state.compositionStyle,
          }));
        const savedBackground = state?.compositionStyle?.backgroundMedia;
        const backgroundMode = state?.compositionStyle?.backgroundMode;
        if (savedBackground?.saved && backgroundMode && backgroundMode !== 'color') {
          fetchProjectBackgroundFile(currentProject.id)
            .then((blob) => {
              if (!blob || clipsLoadingVersion.current !== version) return;
              setBackgroundAsset({
                blob,
                url: URL.createObjectURL(blob),
                kind:
                  savedBackground.kind ||
                  (blob.type.startsWith('video/') ? 'video' : 'image'),
                mimeType: savedBackground.mimeType || blob.type,
                originalName:
                  savedBackground.originalName || 'project background',
              });
            })
            .catch((err) =>
              console.warn('[background] Failed to load project background:', err),
            );
        }
        if (state?.clipStyles && Object.keys(state.clipStyles).length > 0) {
          setClipStyles(
            Object.fromEntries(
              Object.entries(state.clipStyles).map(([id, saved]) => [
                id,
                { ...DEFAULT_CLIP_STYLE, ...saved },
              ]),
            ),
          );
        }
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
        console.warn('[clips] Failed to save MIDI to project state:', err);
      }
    };
    reader.readAsDataURL(midiFile);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [midiFile, currentProject?.id]);

  // ── 4. Debounced save of arrangement, volumes, and style ──────────────────
  useEffect(() => {
    if (!currentProject || !hasGridArrangement(gridArrangement)) return;
    clearTimeout(saveArrangementTimeoutRef.current);
    saveArrangementTimeoutRef.current = setTimeout(async () => {
      try {
        const patch = {
          gridArrangement: normalizeGridArrangement(gridArrangement),
          trackVolumes,
          compositionStyle,
          clipStyles,
        };
        shadowStateRef.current = {
          ...(shadowStateRef.current || {}),
          ...patch,
        };
        await saveProjectState(shadowStateRef.current);
      } catch (err) {
        console.warn('[state] Failed to save arrangement:', err);
      }
    }, 1500);
  }, [
    gridArrangement,
    trackVolumes,
    compositionStyle,
    clipStyles,
    currentProject?.id,
  ]); // eslint-disable-line react-hooks/exhaustive-deps

  return { savedClipKeys, setSavedClipKeys, clipBlobCache };
}
