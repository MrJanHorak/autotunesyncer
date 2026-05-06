import { useState, useMemo, useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import GridLayout from 'react-grid-layout';
import { GridClipItem } from './SortableItems';
import { DEFAULT_COMPOSITION_STYLE } from '../../js/styleDefaults';
import { isDrumTrack, getDrumName } from '../../js/drumUtils';
import {
  getGridArrangementBounds,
  hasGridArrangement,
  normalizeGridArrangement,
} from '../../../shared/gridLayout.js';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';
import './Grid.css';

const FONT_FAMILY_MAP = {
  default: 'inherit',
  arial: 'Arial, sans-serif',
  verdana: 'Verdana, sans-serif',
  impact: 'Impact, sans-serif',
  courier: '"Courier New", monospace',
  times: '"Times New Roman", serif',
  georgia: 'Georgia, serif',
  trebuchet: '"Trebuchet MS", sans-serif',
  comic: '"Comic Sans MS", cursive',
};
const getFontFamily = (font) => FONT_FAMILY_MAP[font] || 'inherit';
const BASE_EDITOR_GRID_UNITS = 12;
const EDITOR_GRID_SNAP_FACTOR = 2;
const EDITOR_GRID_UNITS = BASE_EDITOR_GRID_UNITS * EDITOR_GRID_SNAP_FACTOR;
const DEFAULT_STAGE_WIDTH = 960;
const MIN_TILE_SPAN = 2 * EDITOR_GRID_SNAP_FACTOR;
const MAX_TILE_SPAN = 6 * EDITOR_GRID_SNAP_FACTOR;
const getArrangementId = (item) => item.id.replace(/^(track-|drum-)/, '');

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

const sortEditorLayout = (a, b) => {
  if (a.y !== b.y) return a.y - b.y;
  if (a.x !== b.x) return a.x - b.x;
  return String(a.i).localeCompare(String(b.i));
};

const getOverlapArea = (left, right) => {
  const overlapWidth =
    Math.min(left.x + left.w, right.x + right.w) - Math.max(left.x, right.x);
  const overlapHeight =
    Math.min(left.y + left.h, right.y + right.h) - Math.max(left.y, right.y);

  if (overlapWidth <= 0 || overlapHeight <= 0) return 0;
  return overlapWidth * overlapHeight;
};

const isOverlapping = (left, right) => getOverlapArea(left, right) > 0;

const canPlaceEditorItem = (candidate, occupiedItems) =>
  occupiedItems.every(
    (item) => item.i === candidate.i || !isOverlapping(candidate, item),
  );

const getPositionDistance = (left, right) =>
  Math.abs(left.x - right.x) + Math.abs(left.y - right.y);

const getShrinkCandidates = (item, blocker) => {
  const itemRight = item.x + item.w;
  const itemBottom = item.y + item.h;
  const blockerRight = blocker.x + blocker.w;
  const blockerBottom = blocker.y + blocker.h;

  return [
    {
      x: item.x,
      y: item.y,
      size: Math.min(Math.max(0, blocker.x - item.x), item.h),
    },
    {
      x: blockerRight,
      y: item.y,
      size: Math.min(Math.max(0, itemRight - blockerRight), item.h),
    },
    {
      x: item.x,
      y: item.y,
      size: Math.min(item.w, Math.max(0, blocker.y - item.y)),
    },
    {
      x: item.x,
      y: blockerBottom,
      size: Math.min(item.w, Math.max(0, itemBottom - blockerBottom)),
    },
  ]
    .filter((candidate) => candidate.size >= MIN_TILE_SPAN)
    .map((candidate) => ({
      i: item.i,
      x: candidate.x,
      y: candidate.y,
      w: candidate.size,
      h: candidate.size,
    }))
    .sort((left, right) => {
      if (right.w !== left.w) return right.w - left.w;
      return getPositionDistance(left, item) - getPositionDistance(right, item);
    });
};

const findNearestOpenSlot = (item, occupiedItems) => {
  for (let size = item.w; size >= MIN_TILE_SPAN; size -= 1) {
    const maxOrigin = EDITOR_GRID_UNITS - size;
    const positions = [];

    for (let y = 0; y <= maxOrigin; y += 1) {
      for (let x = 0; x <= maxOrigin; x += 1) {
        positions.push({ x, y });
      }
    }

    positions.sort((left, right) => {
      const leftDistance = getPositionDistance(left, item);
      const rightDistance = getPositionDistance(right, item);
      if (leftDistance !== rightDistance) return leftDistance - rightDistance;
      if (left.y !== right.y) return left.y - right.y;
      return left.x - right.x;
    });

    for (const position of positions) {
      const candidate = {
        i: item.i,
        x: position.x,
        y: position.y,
        w: size,
        h: size,
      };

      if (canPlaceEditorItem(candidate, occupiedItems)) {
        return candidate;
      }
    }
  }

  return item;
};

const findSwapTarget = (layout, draggedItem) => {
  if (!draggedItem?.i) return null;

  let bestMatch = null;
  let bestArea = 0;

  layout.forEach((item) => {
    if (!item || item.i === draggedItem.i) return;
    const overlapArea = getOverlapArea(item, draggedItem);
    if (overlapArea > bestArea) {
      bestArea = overlapArea;
      bestMatch = item;
    }
  });

  return bestMatch;
};

const swapLayoutItems = (layout, firstId, secondId) => {
  const swappedLayout = normalizeEditorLayout(layout).map((item) => ({
    ...item,
  }));
  const firstItem = swappedLayout.find((item) => item.i === firstId);
  const secondItem = swappedLayout.find((item) => item.i === secondId);

  if (!firstItem || !secondItem) {
    return swappedLayout;
  }

  const firstRect = {
    x: firstItem.x,
    y: firstItem.y,
    w: firstItem.w,
    h: firstItem.h,
  };

  firstItem.x = secondItem.x;
  firstItem.y = secondItem.y;
  firstItem.w = secondItem.w;
  firstItem.h = secondItem.h;

  secondItem.x = firstRect.x;
  secondItem.y = firstRect.y;
  secondItem.w = firstRect.w;
  secondItem.h = firstRect.h;

  return normalizeEditorLayout(swappedLayout);
};

const getDefaultTileSpan = (itemCount) => {
  if (itemCount <= 4) return 6 * EDITOR_GRID_SNAP_FACTOR;
  if (itemCount <= 9) return 4 * EDITOR_GRID_SNAP_FACTOR;
  if (itemCount <= 16) return 3 * EDITOR_GRID_SNAP_FACTOR;
  return 2 * EDITOR_GRID_SNAP_FACTOR;
};

const normalizeEditorLayout = (layout) =>
  [...(layout || [])]
    .map((entry) => {
      const size = clamp(
        Math.round(
          Math.max(
            Number(entry?.w) || MIN_TILE_SPAN,
            Number(entry?.h) || MIN_TILE_SPAN,
          ),
        ),
        MIN_TILE_SPAN,
        MAX_TILE_SPAN,
      );
      const maxOrigin = EDITOR_GRID_UNITS - size;

      return {
        i: entry.i,
        x: clamp(Math.round(Number(entry?.x) || 0), 0, maxOrigin),
        y: clamp(Math.round(Number(entry?.y) || 0), 0, maxOrigin),
        w: size,
        h: size,
      };
    })
    .sort(sortEditorLayout);

const resolveResizeCollisions = (layout, resizedItemId) => {
  const normalizedLayout = normalizeEditorLayout(layout);
  if (!resizedItemId) return normalizedLayout;

  const resizedItem = normalizedLayout.find((item) => item.i === resizedItemId);
  if (!resizedItem) return normalizedLayout;

  const fixedItems = [];
  const overlappingItems = [];

  normalizedLayout.forEach((item) => {
    if (item.i === resizedItemId) return;

    if (isOverlapping(item, resizedItem)) {
      overlappingItems.push(item);
      return;
    }

    fixedItems.push(item);
  });

  if (!overlappingItems.length) return normalizedLayout;

  const occupiedItems = [resizedItem, ...fixedItems];
  const resolvedItems = overlappingItems
    .sort((left, right) => {
      const overlapDiff =
        getOverlapArea(right, resizedItem) - getOverlapArea(left, resizedItem);
      if (overlapDiff !== 0) return overlapDiff;
      return sortEditorLayout(left, right);
    })
    .map((item) => {
      const shrinkCandidate = getShrinkCandidates(item, resizedItem).find(
        (candidate) => canPlaceEditorItem(candidate, occupiedItems),
      );
      const resolvedItem =
        shrinkCandidate || findNearestOpenSlot(item, occupiedItems);
      occupiedItems.push(resolvedItem);
      return resolvedItem;
    });

  return normalizeEditorLayout([resizedItem, ...fixedItems, ...resolvedItems]);
};

const areEditorLayoutsEqual = (left, right) => {
  const leftLayout = normalizeEditorLayout(left);
  const rightLayout = normalizeEditorLayout(right);
  if (leftLayout.length !== rightLayout.length) return false;

  return leftLayout.every((item, index) => {
    const other = rightLayout[index];
    return (
      item.i === other.i &&
      item.x === other.x &&
      item.y === other.y &&
      item.w === other.w &&
      item.h === other.h
    );
  });
};

const buildDefaultEditorLayout = (processedItems) => {
  const span = getDefaultTileSpan(processedItems.length);
  const itemsPerRow = Math.max(1, Math.floor(EDITOR_GRID_UNITS / span));

  return normalizeEditorLayout(
    processedItems.map((item, index) => ({
      i: item.id,
      x: (index % itemsPerRow) * span,
      y: Math.floor(index / itemsPerRow) * span,
      w: span,
      h: span,
    })),
  );
};

const buildEditorLayoutFromArrangement = (processedItems, arrangement) => {
  const normalizedArrangement = normalizeGridArrangement(arrangement, {
    defaultColumns: EDITOR_GRID_UNITS,
    defaultRows: EDITOR_GRID_UNITS,
  });

  if (!hasGridArrangement(normalizedArrangement)) {
    return buildDefaultEditorLayout(processedItems);
  }

  const arrangementIdToItemId = new Map(
    processedItems.map((item) => [getArrangementId(item), item.id]),
  );
  const matchingEntries = Object.entries(normalizedArrangement.items).filter(
    ([arrangementId]) => arrangementIdToItemId.has(arrangementId),
  );

  if (matchingEntries.length !== processedItems.length) {
    return buildDefaultEditorLayout(processedItems);
  }

  const sourceBounds = getGridArrangementBounds(normalizedArrangement);
  const shouldScale = normalizedArrangement.columns !== EDITOR_GRID_UNITS;
  const scale = shouldScale
    ? Math.max(
        1,
        Math.floor(
          EDITOR_GRID_UNITS /
            Math.max(
              1,
              normalizedArrangement.columns,
              sourceBounds.columnCount,
              sourceBounds.rowCount,
            ),
        ),
      )
    : 1;

  return normalizeEditorLayout(
    matchingEntries.map(([arrangementId, item]) => {
      const size = shouldScale
        ? Math.max(item.w * scale, item.h * scale, scale)
        : Math.max(item.w, item.h);

      return {
        i: arrangementIdToItemId.get(arrangementId),
        x: item.x * scale,
        y: item.y * scale,
        w: size,
        h: size,
      };
    }),
  );
};

const buildArrangementFromEditorLayout = (layout) => {
  const items = Object.fromEntries(
    normalizeEditorLayout(layout).map((item, index) => [
      item.i.replace(/^(track-|drum-)/, ''),
      {
        x: item.x,
        y: item.y,
        w: item.w,
        h: item.h,
        position: index,
        type: item.i.startsWith('drum-') ? 'drum' : 'track',
      },
    ]),
  );

  return normalizeGridArrangement(
    {
      version: 2,
      columns: EDITOR_GRID_UNITS,
      rows: EDITOR_GRID_UNITS,
      items,
    },
    {
      defaultColumns: EDITOR_GRID_UNITS,
      defaultRows: EDITOR_GRID_UNITS,
    },
  );
};

const hexToRgba = (hex, alpha = 1) => {
  const normalized = (hex || '').replace('#', '');
  const safe =
    normalized.length === 3
      ? normalized
          .split('')
          .map((char) => char + char)
          .join('')
      : normalized.padEnd(6, '0').slice(0, 6);
  const value = Number.parseInt(safe, 16);
  const red = (value >> 16) & 255;
  const green = (value >> 8) & 255;
  const blue = value & 255;
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
};

const Grid = ({
  midiData,
  onArrangementChange,
  initialArrangement,
  clipStyles,
  onClipStyleChange,
  instrumentVideos,
  isPreviewPlaying,
  activeLevels,
  compositionStyle,
  backgroundAsset,
}) => {
  const previewStyle = {
    ...DEFAULT_COMPOSITION_STYLE,
    ...(compositionStyle || {}),
  };
  const backgroundMode = previewStyle.backgroundMode || 'color';
  const showBackgroundMedia =
    backgroundMode !== 'color' &&
    backgroundAsset?.url &&
    backgroundAsset?.kind === backgroundMode;
  const titleText = previewStyle.titleText?.trim() || '';
  const titleSubtitleText = previewStyle.titleSubtitleText?.trim() || '';
  const introTitleText = previewStyle.introCardText?.trim() || titleText;
  const introTitleSubtitleText =
    previewStyle.introCardSubtext?.trim() || titleSubtitleText;
  const taglineText = previewStyle.taglineText?.trim() || '';
  const titleFont =
    previewStyle.titleFont || previewStyle.introCardFont || 'default';
  const titleColor =
    previewStyle.titleColor || previewStyle.introCardTextColor || '#ffffff';
  const titleCardBg =
    previewStyle.titleBackgroundColor || previewStyle.introCardBg || '#120b24';
  const titleCardOpacity = previewStyle.titleBackgroundOpacity ?? 0.82;
  const titleUsesCard = Boolean(previewStyle.titleBackgroundEnabled);
  const titleUsesFullscreenBackground =
    titleUsesCard && previewStyle.titleBackgroundMode === 'fullscreen';
  const introDuration = previewStyle.introCardEnabled
    ? (previewStyle.introCardDuration ?? 3)
    : 0;
  const previewDuration = useMemo(() => {
    const tracks = Array.isArray(midiData?.tracks) ? midiData.tracks : [];
    let maxTime = 0;

    tracks.forEach((track) => {
      track?.notes?.forEach((note) => {
        const start = Number(
          note?.time ?? note?.start ?? note?.startTime ?? note?.ticks ?? 0,
        );
        const duration = Number(note?.duration ?? 0);
        const end = Number(note?.end ?? note?.endTime ?? start + duration);
        maxTime = Math.max(maxTime, Number.isFinite(end) ? end : 0);
      });
    });

    return maxTime;
  }, [midiData]);

  const autoTransitionIntervalSeconds = useMemo(() => {
    const tracks = Array.isArray(midiData?.tracks) ? midiData.tracks : [];
    const totalNotes = tracks.reduce(
      (acc, track) =>
        acc + (Array.isArray(track?.notes) ? track.notes.length : 0),
      0,
    );
    const safeDuration = Math.max(1, Number(previewDuration || 0));
    const noteDensity = totalNotes / safeDuration;

    if (noteDensity >= 12) return 2.5;
    if (noteDensity >= 8) return 3.5;
    if (noteDensity >= 4) return 5;
    if (noteDensity >= 2) return 6.5;
    return 8;
  }, [midiData, previewDuration]);

  const beatSyncPulseMs = useMemo(() => {
    const sensitivity = previewStyle.beatSyncSensitivity || 'medium';
    if (sensitivity === 'high') return 450;
    if (sensitivity === 'low') return 900;
    return 650;
  }, [previewStyle.beatSyncSensitivity]);

  // 1. Process MIDI data first
  const processedData = useMemo(() => {
    const trackData = [];
    const drumData = new Map();

    midiData.tracks.forEach((track, trackIndex) => {
      if (!track.notes?.length) return;

      if (isDrumTrack(track)) {
        track.notes.forEach((note) => {
          const drumName = getDrumName(note.midi);
          const key = `drum_${drumName.toLowerCase().replace(/\s+/g, '_')}`;
          if (!drumData.has(key)) {
            drumData.set(key, {
              id: `drum-${key}`,
              name: drumName,
              count: 0,
            });
          }
          drumData.get(key).count++;
        });
      } else {
        trackData.push({
          id: `track-${trackIndex}`,
          name: track.instrument.name,
          count: track.notes.length,
        });
      }
    });

    return [...trackData, ...Array.from(drumData.values())];
  }, [midiData]);

  const initialEditorLayout = useMemo(
    () => buildEditorLayoutFromArrangement(processedData, initialArrangement),
    [processedData, initialArrangement],
  );
  const stageRef = useRef(null);
  const dragStartLayoutRef = useRef(null);
  const resizeItemIdRef = useRef(null);
  const ignoreNextLayoutChangeRef = useRef(false);
  const committedLayoutRef = useRef(initialEditorLayout);
  const [stageWidth, setStageWidth] = useState(DEFAULT_STAGE_WIDTH);
  const [editorLayout, setEditorLayout] = useState(initialEditorLayout);
  const gridHeight = useMemo(() => (stageWidth * 9) / 16, [stageWidth]);
  const rowHeight = useMemo(() => gridHeight / EDITOR_GRID_UNITS, [gridHeight]);

  // Intro card preview playback state
  const [showIntroCard, setShowIntroCard] = useState(false);
  const [introCardFadingOut, setIntroCardFadingOut] = useState(false);
  const [previewTransitionActive, setPreviewTransitionActive] = useState(false);
  const [previewOutroActive, setPreviewOutroActive] = useState(false);
  const [beatPulseActive, setBeatPulseActive] = useState(false);
  const [waveformPhase, setWaveformPhase] = useState(0);
  const [previewPlayCycle, setPreviewPlayCycle] = useState(0);
  const introTimerRef = useRef(null);
  const fadeOutTimerRef = useRef(null);
  const previewTransitionTimerRef = useRef(null);
  const previewTransitionIntervalRef = useRef(null);
  const previewOutroStartTimerRef = useRef(null);
  const beatPulseIntervalRef = useRef(null);
  const beatPulseDecayTimerRef = useRef(null);
  const wasPreviewPlayingRef = useRef(false);

  const waveformLevel = useMemo(() => {
    const values = Object.values(activeLevels || {}).filter((v) =>
      Number.isFinite(v),
    );
    if (!values.length) return 0;

    const normalizeDb = (db) => {
      const clamped = Math.max(-60, Math.min(0, Number(db)));
      return (clamped + 60) / 60;
    };

    const peak = Math.max(...values.map(normalizeDb));
    const avg =
      values.reduce((sum, value) => sum + normalizeDb(value), 0) /
      values.length;
    return Math.max(0, Math.min(1, peak * 0.72 + avg * 0.28));
  }, [activeLevels]);

  const waveformBars = useMemo(() => {
    const count = 52;
    return Array.from({ length: count }, (_, index) => {
      const centerDistance = Math.abs(index - (count - 1) / 2) / (count / 2);
      const centerBoost = 1 - centerDistance * 0.72;
      const oscA = Math.sin(waveformPhase * 1.1 + index * 0.62) * 0.24;
      const oscB = Math.cos(waveformPhase * 0.78 - index * 0.41) * 0.18;
      const dynamic = Math.max(0.06, waveformLevel * centerBoost + oscA + oscB);
      const height = Math.max(0.08, Math.min(1, dynamic));
      return `${(height * 100).toFixed(2)}%`;
    });
  }, [waveformLevel, waveformPhase]);

  useEffect(() => {
    if (isPreviewPlaying && !wasPreviewPlayingRef.current) {
      setPreviewPlayCycle((value) => value + 1);
    }
    wasPreviewPlayingRef.current = isPreviewPlaying;
  }, [isPreviewPlaying]);

  useEffect(() => {
    let timerId;
    if (isPreviewPlaying && previewStyle.waveformEnabled) {
      timerId = setInterval(() => {
        setWaveformPhase((phase) => phase + 0.33);
      }, 85);
    }
    return () => clearInterval(timerId);
  }, [isPreviewPlaying, previewStyle.waveformEnabled]);

  useEffect(() => {
    if (
      isPreviewPlaying &&
      previewStyle.titleEnabled &&
      previewStyle.introCardEnabled &&
      introTitleText
    ) {
      setIntroCardFadingOut(false);
      setShowIntroCard(true);
      introTimerRef.current = setTimeout(() => {
        // Start fade-out animation
        setIntroCardFadingOut(true);
        // Remove after fade-out completes (500ms matches CSS animation)
        fadeOutTimerRef.current = setTimeout(() => {
          setShowIntroCard(false);
          setIntroCardFadingOut(false);
        }, 500);
      }, introDuration * 1000);
    } else if (!isPreviewPlaying) {
      clearTimeout(introTimerRef.current);
      clearTimeout(fadeOutTimerRef.current);
      setShowIntroCard(false);
      setIntroCardFadingOut(false);
    }
    return () => {
      clearTimeout(introTimerRef.current);
      clearTimeout(fadeOutTimerRef.current);
    };
  }, [
    introTitleText,
    introDuration,
    isPreviewPlaying,
    previewStyle.titleEnabled,
    previewStyle.introCardEnabled,
  ]);

  useEffect(() => {
    clearTimeout(previewTransitionTimerRef.current);
    clearInterval(previewTransitionIntervalRef.current);

    const triggerPreviewTransition = () => {
      setPreviewTransitionActive(true);
      const transitionMs =
        Math.max(0.2, Number(previewStyle.transitionDuration ?? 0.6)) * 1000;
      clearTimeout(previewTransitionTimerRef.current);
      previewTransitionTimerRef.current = setTimeout(
        () => {
          setPreviewTransitionActive(false);
        },
        Math.round(transitionMs + 120),
      );
    };

    if (
      isPreviewPlaying &&
      previewStyle.transitionEnabled &&
      (previewStyle.transitionPreset || 'none') !== 'none'
    ) {
      const transitionOnRaw = previewStyle.transitionOn || 'start';
      const transitionOn =
        transitionOnRaw === 'sections'
          ? 'section'
          : transitionOnRaw === 'interval'
            ? 'section'
            : transitionOnRaw === 'auto'
              ? 'phrase'
              : transitionOnRaw;
      if (transitionOn === 'section' || transitionOn === 'phrase') {
        const cadenceSeconds =
          transitionOn === 'phrase'
            ? autoTransitionIntervalSeconds
            : Math.max(2, Number(previewStyle.transitionSectionInterval ?? 8));
        const sectionMs = cadenceSeconds * 1000;
        triggerPreviewTransition();
        previewTransitionIntervalRef.current = setInterval(
          triggerPreviewTransition,
          Math.round(sectionMs),
        );
      } else {
        triggerPreviewTransition();
      }
    } else {
      setPreviewTransitionActive(false);
    }
    return () => {
      clearTimeout(previewTransitionTimerRef.current);
      clearInterval(previewTransitionIntervalRef.current);
    };
  }, [
    isPreviewPlaying,
    previewStyle.transitionEnabled,
    previewStyle.transitionPreset,
    previewStyle.transitionDuration,
    previewStyle.transitionOn,
    previewStyle.transitionSectionInterval,
    autoTransitionIntervalSeconds,
  ]);

  useEffect(() => {
    clearTimeout(previewOutroStartTimerRef.current);
    if (
      isPreviewPlaying &&
      previewStyle.outroEffectEnabled &&
      previewDuration > 0
    ) {
      const outroDuration = Math.max(
        0.4,
        Number(previewStyle.outroEffectDuration ?? 1.2),
      );
      const startMs = Math.max(0, (previewDuration - outroDuration) * 1000);
      previewOutroStartTimerRef.current = setTimeout(() => {
        setPreviewOutroActive(true);
      }, Math.round(startMs));
    } else {
      setPreviewOutroActive(false);
    }
    return () => clearTimeout(previewOutroStartTimerRef.current);
  }, [
    isPreviewPlaying,
    previewDuration,
    previewStyle.outroEffectEnabled,
    previewStyle.outroEffectDuration,
  ]);

  useEffect(() => {
    if (!isPreviewPlaying) {
      setPreviewOutroActive(false);
    }
  }, [isPreviewPlaying]);

  useEffect(() => {
    clearInterval(beatPulseIntervalRef.current);
    clearTimeout(beatPulseDecayTimerRef.current);

    if (isPreviewPlaying && previewStyle.beatSyncEnabled) {
      const triggerPulse = () => {
        setBeatPulseActive(true);
        clearTimeout(beatPulseDecayTimerRef.current);
        beatPulseDecayTimerRef.current = setTimeout(
          () => {
            setBeatPulseActive(false);
          },
          Math.round(Math.max(120, beatSyncPulseMs * 0.32)),
        );
      };

      triggerPulse();
      beatPulseIntervalRef.current = setInterval(
        triggerPulse,
        Math.round(beatSyncPulseMs),
      );
    } else {
      setBeatPulseActive(false);
    }

    return () => {
      clearInterval(beatPulseIntervalRef.current);
      clearTimeout(beatPulseDecayTimerRef.current);
    };
  }, [
    isPreviewPlaying,
    previewStyle.beatSyncEnabled,
    previewStyle.beatSyncSensitivity,
    beatSyncPulseMs,
  ]);

  useEffect(() => {
    setEditorLayout((currentLayout) =>
      areEditorLayoutsEqual(currentLayout, initialEditorLayout)
        ? currentLayout
        : initialEditorLayout,
    );
    committedLayoutRef.current = initialEditorLayout;
  }, [initialEditorLayout]);

  useEffect(() => {
    const stageNode = stageRef.current;
    if (!stageNode) return undefined;

    const updateWidth = () => {
      const nextWidth = Math.max(
        stageNode.clientWidth || DEFAULT_STAGE_WIDTH,
        320,
      );
      setStageWidth((currentWidth) =>
        Math.abs(currentWidth - nextWidth) < 1 ? currentWidth : nextWidth,
      );
    };

    updateWidth();

    if (typeof ResizeObserver === 'undefined') return undefined;

    const observer = new ResizeObserver(() => updateWidth());
    observer.observe(stageNode);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!processedData.length) return;
    if (hasGridArrangement(initialArrangement)) return;
    onArrangementChange(buildArrangementFromEditorLayout(initialEditorLayout));
  }, [
    processedData.length,
    initialArrangement,
    initialEditorLayout,
    onArrangementChange,
  ]);

  const updateEditorLayout = (nextLayout) => {
    if (ignoreNextLayoutChangeRef.current) {
      ignoreNextLayoutChangeRef.current = false;
      return committedLayoutRef.current;
    }

    const normalizedLayout = resizeItemIdRef.current
      ? resolveResizeCollisions(nextLayout, resizeItemIdRef.current)
      : normalizeEditorLayout(nextLayout);
    setEditorLayout((currentLayout) =>
      areEditorLayoutsEqual(currentLayout, normalizedLayout)
        ? currentLayout
        : normalizedLayout,
    );
    return normalizedLayout;
  };

  const applyResolvedEditorLayout = (nextLayout) => {
    const normalizedLayout = normalizeEditorLayout(nextLayout);
    setEditorLayout((currentLayout) =>
      areEditorLayoutsEqual(currentLayout, normalizedLayout)
        ? currentLayout
        : normalizedLayout,
    );
    committedLayoutRef.current = normalizedLayout;
    onArrangementChange(buildArrangementFromEditorLayout(normalizedLayout));
    return normalizedLayout;
  };

  const commitEditorLayout = (nextLayout) => {
    const normalizedLayout = updateEditorLayout(nextLayout);
    committedLayoutRef.current = normalizedLayout;
    onArrangementChange(buildArrangementFromEditorLayout(normalizedLayout));
  };

  const handleDragStart = (nextLayout) => {
    resizeItemIdRef.current = null;
    dragStartLayoutRef.current = normalizeEditorLayout(nextLayout);
  };

  const handleDragStop = (nextLayout, oldItem, newItem) => {
    const startingLayout =
      dragStartLayoutRef.current || committedLayoutRef.current;
    dragStartLayoutRef.current = null;

    const swapTarget = findSwapTarget(startingLayout, newItem || oldItem);
    if (swapTarget) {
      ignoreNextLayoutChangeRef.current = true;
      const swappedLayout = swapLayoutItems(
        startingLayout,
        (newItem || oldItem).i,
        swapTarget.i,
      );
      applyResolvedEditorLayout(swappedLayout);
      return;
    }

    commitEditorLayout(nextLayout);
  };

  const handleResizeStart = (nextLayout, oldItem, newItem) => {
    resizeItemIdRef.current = (newItem || oldItem)?.i || null;
  };

  const handleResize = (nextLayout, oldItem, newItem) => {
    resizeItemIdRef.current =
      (newItem || oldItem)?.i || resizeItemIdRef.current;
    updateEditorLayout(nextLayout);
  };

  const handleResizeStop = (nextLayout, oldItem, newItem) => {
    resizeItemIdRef.current =
      (newItem || oldItem)?.i || resizeItemIdRef.current;
    commitEditorLayout(nextLayout);
    resizeItemIdRef.current = null;
  };

  const handleResetLayout = () => {
    resizeItemIdRef.current = null;
    const nextLayout = buildDefaultEditorLayout(processedData);
    setEditorLayout(nextLayout);
    committedLayoutRef.current = nextLayout;
    onArrangementChange(buildArrangementFromEditorLayout(nextLayout));
  };

  // Heat map calculations with modern spectrum gradient - 10 tier system for maximum distinction
  const getHeatColor = (intensity) => {
    // Creates smooth gradient backgrounds through the full thermal spectrum
    // Blue (cold) → Cyan → Green → Yellow → Orange → Red (hot)
    // Maximum visual distinction across entire activity range
    if (intensity < 0.1) {
      // Extreme low (0-10%) - pale blue
      return 'linear-gradient(135deg, #bfdbfe 0%, #93c5fd 100%)';
    } else if (intensity < 0.2) {
      // Very minimal (10-20%) - sky blue
      return 'linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%)';
    } else if (intensity < 0.3) {
      // Minimal (20-30%) - bright blue
      return 'linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%)';
    } else if (intensity < 0.4) {
      // Very low (30-40%) - cyan
      return 'linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%)';
    } else if (intensity < 0.5) {
      // Low (40-50%) - cyan-green
      return 'linear-gradient(135deg, #14b8a6 0%, #10b981 100%)';
    } else if (intensity < 0.6) {
      // Medium-low (50-60%) - green
      return 'linear-gradient(135deg, #22c55e 0%, #84cc16 100%)';
    } else if (intensity < 0.7) {
      // Medium (60-70%) - yellow-green
      return 'linear-gradient(135deg, #84cc16 0%, #eab308 100%)';
    } else if (intensity < 0.8) {
      // Medium-high (70-80%) - yellow-orange
      return 'linear-gradient(135deg, #eab308 0%, #f59e0b 100%)';
    } else if (intensity < 0.9) {
      // High (80-90%) - orange
      return 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)';
    } else {
      // Extreme high (90-100%) - orange to red
      return 'linear-gradient(135deg, #ea580c 0%, #dc2626 100%)';
    }
  };

  const getHeatIntensity = (count) => {
    const maxCount = Math.max(...processedData.map((item) => item.count || 0));
    return maxCount > 0 ? count / maxCount : 0;
  };

  // Get accent color for text - always white for readability
  const getAccentColor = () => {
    return '#ffffff'; // Always white for best readability
  };
  const getTitlePositionStyle = () => {
    if (titleUsesFullscreenBackground) {
      return {
        inset: '0',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
      };
    }

    switch (previewStyle.titlePosition) {
      case 'bottom-center':
        return {
          left: '50%',
          bottom: '16px',
          transform: 'translateX(-50%)',
        };
      case 'center':
        return {
          left: '50%',
          top: '50%',
          transform: 'translate(-50%, -50%)',
        };
      default:
        return { left: '50%', top: '14px', transform: 'translateX(-50%)' };
    }
  };

  const getTaglinePositionStyle = () => {
    const width = `${Math.max(20, Math.min(100, previewStyle.taglineWidth ?? 72))}%`;
    const verticalOffset = Math.max(
      -160,
      Math.min(160, Number(previewStyle.taglineVerticalOffset ?? 0)),
    );
    const bottom = `${14 + verticalOffset}px`;

    switch (previewStyle.taglinePosition) {
      case 'bottom-left':
        return {
          left: '14px',
          bottom,
          transform: 'none',
          width,
        };
      case 'bottom-right':
        return {
          right: '14px',
          bottom,
          left: 'auto',
          transform: 'none',
          width,
        };
      default:
        return {
          left: '50%',
          bottom,
          transform: 'translateX(-50%)',
          width,
        };
    }
  };

  const getTaglineSurfaceStyle = () => {
    if (!previewStyle.taglineBackgroundEnabled) return {};

    const shape = previewStyle.taglineShape || 'rounded';
    const accent = previewStyle.taglineAccentColor || '#ff4db8';
    const bg = hexToRgba(
      previewStyle.taglineBackgroundColor || '#0c1220',
      previewStyle.taglineBackgroundOpacity ?? 0.72,
    );

    const base = {
      background: bg,
      boxShadow: '0 -10px 32px rgba(0, 0, 0, 0.22)',
      padding: '0.65rem 1rem 0.7rem',
      borderTop: `3px solid ${accent}`,
      borderRadius: '14px 14px 0 0',
    };

    if (shape === 'pill') {
      return {
        ...base,
        borderTop: 'none',
        border: `1px solid ${hexToRgba(accent, 0.55)}`,
        borderRadius: '999px',
        padding: '0.58rem 1.25rem',
      };
    }

    if (shape === 'square') {
      return {
        ...base,
        borderRadius: '0',
      };
    }

    if (shape === 'outline') {
      return {
        ...base,
        borderTop: 'none',
        border: `2px solid ${hexToRgba(accent, 0.78)}`,
        background: hexToRgba(
          previewStyle.taglineBackgroundColor || '#0c1220',
          Math.max(
            0.18,
            (previewStyle.taglineBackgroundOpacity ?? 0.72) * 0.58,
          ),
        ),
      };
    }

    if (shape === 'accent-left') {
      return {
        ...base,
        borderTop: 'none',
        borderLeft: `4px solid ${accent}`,
        borderRadius: '12px',
      };
    }

    return base;
  };

  const getTaglineAnimationStyle = () => {
    if (!isPreviewPlaying || previewDuration <= 0) {
      return {
        className: '',
        style: {
          '--grid-tagline-transform':
            getTaglinePositionStyle().transform || 'none',
        },
      };
    }

    const fadeInDuration = Math.max(
      0,
      Number(previewStyle.taglineFadeInDuration ?? 0.5),
    );
    const fadeOutDuration = Math.max(
      0,
      Number(previewStyle.taglineFadeOutDuration ?? 0.5),
    );
    const startAt = previewStyle.introCardEnabled ? introDuration : 0;
    const fadeOutStart = Math.max(
      startAt + fadeInDuration,
      previewDuration - fadeOutDuration,
    );

    return {
      className: [
        fadeInDuration > 0 ? 'grid-preview-tagline--fade-in' : '',
        fadeOutDuration > 0 && previewDuration > startAt
          ? 'grid-preview-tagline--fade-out'
          : '',
      ]
        .filter(Boolean)
        .join(' '),
      style: {
        '--grid-tagline-transform':
          getTaglinePositionStyle().transform || 'none',
        '--tagline-fade-in-delay': `${startAt}s`,
        '--tagline-fade-in-duration': `${fadeInDuration}s`,
        '--tagline-fade-out-delay': `${fadeOutStart}s`,
        '--tagline-fade-out-duration': `${fadeOutDuration}s`,
      },
    };
  };

  const getTitleAnimationStyle = () => {
    const preset = previewStyle.titleAnimationPreset || 'fade';
    const intensity = previewStyle.titleAnimIntensity || 'medium';
    const direction = previewStyle.titleAnimDirection || 'left';

    const intensityMap = {
      low: 0.7,
      medium: 1,
      high: 1.35,
    };
    const dirSign = direction === 'right' ? 1 : -1;
    const factor = intensityMap[intensity] || 1;
    const duration = Math.max(
      0.3,
      Number(previewStyle.titleAnimDuration ?? 0.7),
    );
    const delay = Math.max(0, Number(previewStyle.titleAnimDelay ?? 0));
    const titleLength = Math.max(1, String(titleText || '').trim().length);
    const typewriterSteps = Math.min(60, Math.max(8, titleLength));
    const easing =
      previewStyle.titleAnimEasing || 'cubic-bezier(0.22, 1, 0.36, 1)';

    const presetClassMap = {
      fade: 'grid-preview-title--anim-fade',
      'scroll-up': 'grid-preview-title--anim-scroll-up',
      'scroll-left': 'grid-preview-title--anim-scroll-left',
      bounce: 'grid-preview-title--anim-bounce',
      'spin-soft': 'grid-preview-title--anim-spin-soft',
      'blur-focus': 'grid-preview-title--anim-blur-focus',
      typewriter: 'grid-preview-title--anim-typewriter',
    };

    const entryEasing =
      preset === 'typewriter' ? `steps(${typewriterSteps}, end)` : easing;

    return {
      className: presetClassMap[preset] || presetClassMap.fade,
      style: {
        '--title-entry-animation': `grid-title-${preset}`,
        '--title-entry-duration': `${duration}s`,
        '--title-entry-delay': `${delay}s`,
        '--title-entry-easing': entryEasing,
        '--title-motion-y': `${Math.round(-38 * factor)}px`,
        '--title-motion-x': `${Math.round(46 * factor * dirSign)}px`,
        '--title-rotate-start': `${(dirSign * 7 * factor).toFixed(2)}deg`,
        '--title-scale-start': `${(0.94 - (factor - 1) * 0.02).toFixed(3)}`,
        '--title-blur-start': `${(6 * factor).toFixed(1)}px`,
      },
      delay,
    };
  };

  const getWatermarkPositionStyle = () => {
    switch (previewStyle.watermarkPosition) {
      case 'bottom-left':
        return { left: '12px', bottom: '12px' };
      case 'top-right':
        return { right: '12px', top: '12px' };
      case 'top-left':
        return { left: '12px', top: '12px' };
      default:
        return { right: '12px', bottom: '12px' };
    }
  };

  const getPreviewTransitionStyle = () => {
    const presetRaw = previewStyle.transitionPreset || 'none';
    const preset =
      presetRaw === 'slide-left'
        ? 'push-left'
        : presetRaw === 'slide-right'
          ? 'push-right'
          : presetRaw === 'zoom-in'
            ? 'zoom'
            : presetRaw;
    const strength = previewStyle.transitionStrength || 'medium';
    const duration = Math.max(
      0.2,
      Number(previewStyle.transitionDuration ?? 0.6),
    );
    const strengthMap = { low: 0.55, medium: 0.8, high: 1 };
    const timingModeRaw = previewStyle.transitionOn || 'start';
    const timingMode =
      timingModeRaw === 'sections'
        ? 'section'
        : timingModeRaw === 'interval'
          ? 'section'
          : timingModeRaw === 'auto'
            ? 'phrase'
            : timingModeRaw;
    const repeatByInterval =
      timingMode === 'section' || timingMode === 'phrase';

    return {
      className:
        previewStyle.transitionEnabled &&
        previewTransitionActive &&
        preset !== 'none'
          ? `grid-preview-stage--transition grid-preview-stage--transition-${preset}`
          : '',
      style: {
        '--stage-transition-duration': `${duration}s`,
        '--stage-transition-strength': `${strengthMap[strength] || 0.8}`,
        '--stage-transition-repeat': repeatByInterval ? '1' : '0',
        '--stage-transition-auto-interval': `${autoTransitionIntervalSeconds}s`,
      },
    };
  };

  const getPreviewOutroStyle = () => {
    const preset = previewStyle.outroEffectPreset || 'fade-black';
    const strength = previewStyle.outroEffectStrength || 'medium';
    const duration = Math.max(
      0.4,
      Number(previewStyle.outroEffectDuration ?? 1.2),
    );
    const strengthMap = { low: 0.55, medium: 0.8, high: 1 };

    return {
      className:
        previewStyle.outroEffectEnabled && previewOutroActive
          ? `grid-preview-stage--outro grid-preview-stage--outro-${preset}`
          : '',
      style: {
        '--outro-duration': `${duration}s`,
        '--outro-strength': `${strengthMap[strength] || 0.8}`,
      },
    };
  };

  const titleAnimationStyle = getTitleAnimationStyle();
  const stageTransitionStyle = getPreviewTransitionStyle();
  const stageOutroStyle = getPreviewOutroStyle();
  const beatSyncTargets = Array.isArray(previewStyle.beatSyncTargets)
    ? previewStyle.beatSyncTargets
    : [];
  const beatPulseMode = previewStyle.beatPulseMode || 'scale';
  const beatPulseClass = beatPulseActive
    ? `grid-beat-pulse--${beatPulseMode}`
    : '';
  const hasBeatTarget = (target) =>
    previewStyle.beatSyncEnabled && beatSyncTargets.includes(target);
  const overlayBeatPulseClass = hasBeatTarget('overlays') ? beatPulseClass : '';
  const titleBeatPulseClass = hasBeatTarget('title') ? beatPulseClass : '';
  const taglineBeatPulseClass = hasBeatTarget('tagline') ? beatPulseClass : '';
  const trackCellsBeatPulseClass = hasBeatTarget('track-cells')
    ? beatPulseClass
    : '';
  const canEditLayout = !isPreviewPlaying && processedData.length > 0;

  return (
    <div className='grid-container'>
      {canEditLayout && (
        <div className='grid-layout-toolbar'>
          <span className='grid-layout-note'>
            Drag from the grip to move clips. Resize from the lower-right corner
            to spotlight instruments. Drop onto another clip to swap positions.
          </span>
          <button className='grid-layout-reset' onClick={handleResetLayout}>
            Reset Layout
          </button>
        </div>
      )}

      <div
        ref={stageRef}
        className={[
          'grid-preview-stage',
          stageTransitionStyle.className,
          stageOutroStyle.className,
        ]
          .filter(Boolean)
          .join(' ')}
        style={{
          background: previewStyle.backgroundColor || '#0a0a0f',
          height: `${gridHeight}px`,
          ...stageTransitionStyle.style,
          ...stageOutroStyle.style,
        }}
      >
        {showBackgroundMedia && backgroundAsset.kind === 'image' && (
          <img
            src={backgroundAsset.url}
            alt=''
            aria-hidden='true'
            className='grid-preview-stage__background-media'
          />
        )}
        {showBackgroundMedia && backgroundAsset.kind === 'video' && (
          <video
            src={backgroundAsset.url}
            muted
            loop
            autoPlay
            playsInline
            preload='auto'
            aria-hidden='true'
            className='grid-preview-stage__background-media'
          />
        )}
        <GridLayout
          className='grid'
          layout={editorLayout}
          cols={EDITOR_GRID_UNITS}
          rowHeight={rowHeight}
          width={stageWidth}
          autoSize={false}
          margin={[0, 0]}
          containerPadding={[0, 0]}
          compactType={null}
          resizeHandles={['se']}
          draggableHandle='.grid-cell__drag-handle'
          draggableCancel='.cell-style-btn,.clip-style-popover,.clip-style-popover *,input,button,select,label'
          isDraggable={!isPreviewPlaying}
          isResizable={!isPreviewPlaying}
          allowOverlap
          isBounded
          maxRows={EDITOR_GRID_UNITS}
          style={{ height: `${gridHeight}px` }}
          onLayoutChange={updateEditorLayout}
          onDragStart={handleDragStart}
          onDragStop={handleDragStop}
          onResizeStart={handleResizeStart}
          onResize={handleResize}
          onResizeStop={handleResizeStop}
        >
          {processedData.map((item) => {
            const intensity = getHeatIntensity(item.count);
            const videoKey = item.id.startsWith('drum-')
              ? item.id.replace('drum-', '')
              : (item.name || '').toLowerCase().replace(/\s+/g, '_');
            const videoUrl = instrumentVideos?.[videoKey] || null;

            return (
              <div key={item.id} className='grid-layout-item'>
                <GridClipItem
                  id={item.id}
                  item={item}
                  getHeatColor={getHeatColor(intensity)}
                  accentColor={getAccentColor(intensity)}
                  isEmpty={false}
                  clipStyle={clipStyles?.[item.id]}
                  onClipStyleChange={(newStyle) =>
                    onClipStyleChange?.(item.id, newStyle)
                  }
                  videoUrl={videoUrl}
                  isPreviewPlaying={isPreviewPlaying}
                  hasStageBackgroundMedia={showBackgroundMedia}
                  activeLevel={activeLevels?.[videoKey]}
                  beatPulseClass={trackCellsBeatPulseClass}
                  isEditable={canEditLayout}
                  fillParent
                />
              </div>
            );
          })}
        </GridLayout>

        <div
          className={['grid-preview-overlay', overlayBeatPulseClass]
            .filter(Boolean)
            .join(' ')}
          aria-hidden='true'
        >
          {previewStyle.vignetteEnabled && (
            <div
              className='grid-preview-vignette'
              style={{
                opacity: Math.min(
                  Math.max(previewStyle.vignetteStrength ?? 0.5, 0.1),
                  1,
                ),
              }}
            />
          )}

          {previewStyle.glitchEnabled && (
            <div
              className={`grid-preview-glitch grid-preview-glitch--${previewStyle.glitchIntensity || 'subtle'}`}
            />
          )}

          {previewStyle.waveformEnabled && (
            <div
              className='grid-preview-waveform'
              style={{
                '--waveform-height': `${Math.max(16, Math.min(220, Number(previewStyle.waveformHeight ?? 60)))}px`,
                '--waveform-color': previewStyle.waveformColor || '#00ff88',
                '--waveform-active': isPreviewPlaying ? '1' : '0.58',
                '--waveform-level': waveformLevel.toFixed(3),
              }}
            >
              {waveformBars.map((height, index) => (
                <span
                  key={`wave-${index}`}
                  className='grid-preview-waveform__bar'
                  style={{ height }}
                />
              ))}
            </div>
          )}

          {previewStyle.titleEnabled && titleText && !showIntroCard && (
            <div
              key={`title-${previewPlayCycle}`}
              data-title-animation={previewStyle.titleAnimationPreset || 'fade'}
              className={[
                titleUsesFullscreenBackground
                  ? 'grid-preview-title grid-preview-title--fullscreen'
                  : titleUsesCard
                    ? 'grid-preview-title-card'
                    : 'grid-preview-title',
                previewStyle.titleAnimated && isPreviewPlaying
                  ? [
                      'grid-preview-title--fade-in',
                      titleAnimationStyle.className,
                      titleBeatPulseClass,
                    ]
                      .filter(Boolean)
                      .join(' ')
                  : titleBeatPulseClass,
                (previewStyle.titleDuration ?? 0) > 0 && isPreviewPlaying
                  ? 'grid-preview-title--fade-out'
                  : '',
              ]
                .filter(Boolean)
                .join(' ')}
              style={{
                ...getTitlePositionStyle(),
                '--grid-title-transform': titleUsesFullscreenBackground
                  ? 'translateY(0)'
                  : getTitlePositionStyle().transform || 'translateX(-50%)',
                ...(previewStyle.titleAnimated
                  ? titleAnimationStyle.style
                  : {}),
                color: titleColor,
                fontSize: `${previewStyle.titleFontSize}px`,
                fontFamily: getFontFamily(titleFont),
                ...(previewStyle.titleGlowEnabled && {
                  textShadow: `0 0 ${previewStyle.titleGlowSize || 8}px ${previewStyle.titleGlowColor || '#ffffff'}, 
                                 0 ${previewStyle.titleShadowSize || 2}px ${previewStyle.titleShadowSize || 2}px rgba(0,0,0,0.5)`,
                }),
                ...(!previewStyle.titleGlowEnabled &&
                  previewStyle.titleShadowEnabled && {
                    textShadow: `0 ${previewStyle.titleShadowSize || 2}px ${previewStyle.titleShadowSize || 2}px rgba(0,0,0,0.55)`,
                  }),
                ...(titleUsesCard &&
                  !titleUsesFullscreenBackground && {
                    background: hexToRgba(titleCardBg, titleCardOpacity),
                    border: '1px solid rgba(255,255,255,0.16)',
                    borderRadius: '18px',
                    padding: '0.75rem 1.1rem',
                    boxShadow: '0 18px 48px rgba(0,0,0,0.28)',
                    backdropFilter: 'blur(14px)',
                  }),
                ...(titleUsesFullscreenBackground && {
                  background: hexToRgba(titleCardBg, titleCardOpacity),
                  padding: '2rem',
                  backdropFilter: 'blur(14px)',
                }),
                ...((previewStyle.titleDuration ?? 0) > 0 && {
                  '--title-fade-out-delay': `${(previewStyle.titleDuration ?? 0) + (previewStyle.titleAnimated ? titleAnimationStyle.delay : 0)}s`,
                }),
              }}
            >
              <span className='grid-preview-title__text'>{titleText}</span>
              {titleSubtitleText && (
                <span
                  className='grid-preview-title__subtext'
                  style={{
                    color: previewStyle.titleSubtitleColor || '#d8d8e6',
                    fontSize: `${previewStyle.titleSubtitleFontSize ?? Math.max(14, Math.round(previewStyle.titleFontSize * 0.43))}px`,
                    fontFamily: getFontFamily(titleFont),
                  }}
                >
                  {titleSubtitleText}
                </span>
              )}
            </div>
          )}

          {previewStyle.taglineEnabled && taglineText && !showIntroCard && (
            <div
              className={[
                'grid-preview-tagline',
                getTaglineAnimationStyle().className,
                taglineBeatPulseClass,
              ]
                .filter(Boolean)
                .join(' ')}
              style={{
                ...getTaglinePositionStyle(),
                ...getTaglineAnimationStyle().style,
                color: previewStyle.taglineColor,
                fontSize: `${previewStyle.taglineFontSize}px`,
                fontFamily: getFontFamily(previewStyle.taglineFont),
                textAlign: previewStyle.taglineAlignment || 'center',
                maxWidth: 'calc(100% - 28px)',
                ...(previewStyle.taglineShadowEnabled && {
                  textShadow: `0 ${previewStyle.taglineShadowSize || 2}px ${previewStyle.taglineShadowSize || 2}px ${hexToRgba(previewStyle.taglineShadowColor || '#000000', 0.55)}`,
                }),
                ...getTaglineSurfaceStyle(),
              }}
            >
              {taglineText}
            </div>
          )}

          {previewStyle.watermarkEnabled &&
            previewStyle.watermarkText?.trim() && (
              <div
                className='grid-preview-watermark'
                style={{
                  ...getWatermarkPositionStyle(),
                  color: previewStyle.watermarkColor,
                  fontSize: `${previewStyle.watermarkFontSize}px`,
                  fontFamily: getFontFamily(previewStyle.watermarkFont),
                  opacity: Math.min(
                    Math.max(previewStyle.watermarkOpacity ?? 0.5, 0.1),
                    1,
                  ),
                }}
              >
                {previewStyle.watermarkText}
              </div>
            )}

          {/* Intro card: full-screen overlay at start of preview playback */}
          {showIntroCard && (
            <div
              className={`grid-intro-card${previewStyle.introCardAnimated ? ' grid-intro-card--animated' : ''}${introCardFadingOut ? ' grid-intro-card--hiding' : ''}${titleUsesFullscreenBackground ? ' grid-intro-card--fullscreen' : ' grid-intro-card--panel'}`}
              style={{
                background: titleUsesFullscreenBackground
                  ? hexToRgba(
                      titleCardBg,
                      Math.min(titleCardOpacity + 0.13, 0.95),
                    )
                  : 'transparent',
              }}
            >
              <div
                className={
                  titleUsesFullscreenBackground
                    ? 'grid-intro-card__content'
                    : 'grid-intro-card__content grid-intro-card__content--panel'
                }
                style={
                  titleUsesFullscreenBackground
                    ? undefined
                    : {
                        background: hexToRgba(
                          titleCardBg,
                          Math.min(titleCardOpacity + 0.13, 0.95),
                        ),
                      }
                }
              >
                <p
                  className='grid-intro-card__title'
                  style={{
                    color: titleColor,
                    fontFamily: getFontFamily(titleFont),
                  }}
                >
                  {introTitleText || 'Untitled'}
                </p>
                {introTitleSubtitleText && (
                  <p
                    className='grid-intro-card__subtext'
                    style={{
                      color: previewStyle.titleSubtitleColor || '#d8d8e6',
                      fontFamily: getFontFamily(titleFont),
                      fontSize: `${previewStyle.titleSubtitleFontSize ?? 24}px`,
                    }}
                  >
                    {introTitleSubtitleText}
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

Grid.propTypes = {
  midiData: PropTypes.shape({
    tracks: PropTypes.arrayOf(
      PropTypes.shape({
        notes: PropTypes.array,
        instrument: PropTypes.shape({
          name: PropTypes.string,
        }),
      }),
    ),
  }).isRequired,
  onArrangementChange: PropTypes.func.isRequired,
  initialArrangement: PropTypes.object,
  clipStyles: PropTypes.object,
  onClipStyleChange: PropTypes.func,
  instrumentVideos: PropTypes.object,
  isPreviewPlaying: PropTypes.bool,
  activeLevels: PropTypes.object,
  compositionStyle: PropTypes.object,
  backgroundAsset: PropTypes.object,
};

export default Grid;
