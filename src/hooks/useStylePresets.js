import { useState, useCallback } from 'react';
import { DEFAULT_COMPOSITION_STYLE } from '../js/styleDefaults';

const STORAGE_KEY = 'ats_style_presets';
const MAX_PRESETS = 50;

function loadFromStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((p) => p && typeof p.name === 'string' && p.style);
  } catch {
    return [];
  }
}

function saveToStorage(presets) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(presets));
  } catch {
    console.warn('[presets] localStorage write failed');
  }
}

/**
 * Hook for managing named composition style presets in localStorage.
 *
 * @returns {{
 *   presets: Array<{name: string, style: object}>,
 *   savePreset: (name: string, style: object) => void,
 *   applyPreset: (name: string) => object | null,
 *   deletePreset: (name: string) => void,
 * }}
 */
export function useStylePresets() {
  const [presets, setPresets] = useState(loadFromStorage);

  const savePreset = useCallback((name, style) => {
    setPresets((prev) => {
      const filtered = prev.filter((p) => p.name !== name);
      const next = [...filtered, { name, style }].slice(-MAX_PRESETS);
      saveToStorage(next);
      return next;
    });
  }, []);

  const applyPreset = useCallback(
    (name) => {
      const preset = presets.find((p) => p.name === name);
      if (!preset) return null;
      // Merge with defaults so any new fields added since save are populated.
      return { ...DEFAULT_COMPOSITION_STYLE, ...preset.style };
    },
    [presets],
  );

  const deletePreset = useCallback((name) => {
    setPresets((prev) => {
      const next = prev.filter((p) => p.name !== name);
      saveToStorage(next);
      return next;
    });
  }, []);

  return { presets, savePreset, applyPreset, deletePreset };
}
