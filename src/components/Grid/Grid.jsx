import { useState, useMemo, useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
} from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  rectSortingStrategy,
} from '@dnd-kit/sortable';
import { SortableItem } from './SortableItems';
import { DEFAULT_COMPOSITION_STYLE } from '../../js/styleDefaults';
import { isDrumTrack, getDrumName } from '../../js/drumUtils';
import './Grid.css';

const FONT_FAMILY_MAP = {
  default:   'inherit',
  arial:     'Arial, sans-serif',
  verdana:   'Verdana, sans-serif',
  impact:    'Impact, sans-serif',
  courier:   '"Courier New", monospace',
  times:     '"Times New Roman", serif',
  georgia:   'Georgia, serif',
  trebuchet: '"Trebuchet MS", sans-serif',
  comic:     '"Comic Sans MS", cursive',
};
const getFontFamily = (font) => FONT_FAMILY_MAP[font] || 'inherit';
const hexToRgba = (hex, alpha = 1) => {
  const normalized = (hex || '').replace('#', '');
  const safe = normalized.length === 3
    ? normalized.split('').map((char) => char + char).join('')
    : normalized.padEnd(6, '0').slice(0, 6);
  const value = Number.parseInt(safe, 16);
  const red = (value >> 16) & 255;
  const green = (value >> 8) & 255;
  const blue = value & 255;
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
};

const Grid = ({ midiData, onArrangementChange, initialArrangement, clipStyles, onClipStyleChange, instrumentVideos, isPreviewPlaying, activeLevels, compositionStyle }) => {
  const previewStyle = { ...DEFAULT_COMPOSITION_STYLE, ...(compositionStyle || {}) };
  const titleText = previewStyle.titleText?.trim() || previewStyle.introCardText?.trim() || '';
  const titleSubtitleText = previewStyle.titleSubtitleText?.trim() || previewStyle.introCardSubtext?.trim() || '';
  const taglineText = previewStyle.taglineText?.trim() || '';
  const titleFont = previewStyle.titleFont || previewStyle.introCardFont || 'default';
  const titleColor = previewStyle.titleColor || previewStyle.introCardTextColor || '#ffffff';
  const titleCardBg = previewStyle.titleBackgroundColor || previewStyle.introCardBg || '#120b24';
  const titleCardOpacity = previewStyle.titleBackgroundOpacity ?? 0.82;
  const titleUsesCard = Boolean(previewStyle.titleBackgroundEnabled);
  const introDuration = previewStyle.introCardEnabled ? (previewStyle.introCardDuration ?? 3) : 0;

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

  // 2. Calculate optimal columns based on processed data
  const calculateOptimalColumns = useMemo(() => {
    const itemCount = processedData.length;
    const optimalColumns = [];

    for (let cols = 1; cols <= Math.min(5, itemCount); cols++) {
      const rows = Math.ceil(itemCount / cols);
      const gridWidth = cols * 16;
      const gridHeight = rows * 9;
      const gridAspectRatio = gridWidth / gridHeight;

      const isViable =
        gridAspectRatio >= 1 &&
        gridAspectRatio <= 2 &&
        rows <= 3 &&
        rows * cols >= itemCount;

      if (isViable) {
        optimalColumns.push({
          cols,
          ratio: Math.abs(1.7777 - gridAspectRatio),
        });
      }
    }

    optimalColumns.sort((a, b) => a.ratio - b.ratio);
    return optimalColumns.length > 0
      ? optimalColumns.map((col) => col.cols)
      : [Math.ceil(Math.sqrt(itemCount))];
  }, [processedData.length]);

  // 3. Create initial grid data with empty spaces
  const initialGridData = useMemo(() => {
    const totalColumns = calculateOptimalColumns[0] || 4;
    const totalRows = Math.ceil(processedData.length / totalColumns);
    const totalSpaces = totalRows * totalColumns;

    const emptySpaces = Array.from(
      { length: totalSpaces - processedData.length },
      (_, index) => ({
        id: `empty-${index}`,
        name: '',
        count: 0,
        isEmpty: true,
      })
    );

    return [...processedData, ...emptySpaces];
  }, [processedData, calculateOptimalColumns]);

  // 4. Initialize state
  const [items, setItems] = useState(initialGridData);
  const [columnCount, setColumnCount] = useState(
    calculateOptimalColumns[0] || 4
  );
  const arrangementRestoredRef = useRef(false);

  // Intro card preview playback state
  const [showIntroCard, setShowIntroCard] = useState(false);
  const introTimerRef = useRef(null);

  useEffect(() => {
    if (isPreviewPlaying && previewStyle.introCardEnabled && titleText) {
      setShowIntroCard(true);
      introTimerRef.current = setTimeout(() => {
        setShowIntroCard(false);
      }, introDuration * 1000);
    } else if (!isPreviewPlaying) {
      clearTimeout(introTimerRef.current);
      setShowIntroCard(false);
    }
    return () => clearTimeout(introTimerRef.current);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [introDuration, isPreviewPlaying, previewStyle.introCardEnabled, titleText]);

  // Restore saved drag order once (on first non-empty initialArrangement)
  useEffect(() => {
    if (arrangementRestoredRef.current) return;
    if (!initialArrangement || Object.keys(initialArrangement).length === 0) return;
    arrangementRestoredRef.current = true;
    setItems((current) =>
      [...current].sort((a, b) => {
        const idA = a.id.replace(/^(track-|drum-)/, '');
        const idB = b.id.replace(/^(track-|drum-)/, '');
        const posA = initialArrangement[idA]?.position ?? Infinity;
        const posB = initialArrangement[idB]?.position ?? Infinity;
        return posA - posB;
      })
    );
  }, [initialArrangement]);

  // DND setup
  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  // Handlers
  // const handleDragEnd = (event) => {
  //   const { active, over } = event;

  //   if (active.id !== over.id) {
  //     setItems((items) => {
  //       const oldIndex = items.findIndex((item) => item.id === active.id);
  //       const newIndex = items.findIndex((item) => item.id === over.id);

  //       if (oldIndex !== -1 && newIndex !== -1) {
  //         return arrayMove(items, oldIndex, newIndex);
  //       }
  //       return items;
  //     });
  //   }
  // };

  const handleDragEnd = (event) => {
    const { active, over } = event;
    if (active.id !== over.id) {
      setItems((items) => {
        const oldIndex = items.findIndex((item) => item.id === active.id);
        const newIndex = items.findIndex((item) => item.id === over.id);
        const newItems = arrayMove(items, oldIndex, newIndex);

        const arrangement = newItems.reduce((acc, item, index) => {
          const id = item.isEmpty
            ? item.id
            : item.id.replace(/^(track-|drum-)/, '');
          acc[id] = {
            position: index,
            row: Math.floor(index / columnCount),
            column: index % columnCount,
            type: item.isEmpty ? 'empty' : (item.id.startsWith('drum-') ? 'drum' : 'track'),
            isEmpty: item.isEmpty || false,
          };
          return acc;
        }, {});
        onArrangementChange(arrangement);
        return newItems;
      });
    }
  };

  const handleColumnChange = (event) => {
    const newColumnCount = parseInt(event.target.value);
    setColumnCount(newColumnCount);
    document.documentElement.style.setProperty(
      '--column-count',
      newColumnCount
    );
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
    const maxCount = Math.max(...items.map((item) => item.count || 0));
    return maxCount > 0 ? count / maxCount : 0;
  };

  // Get accent color for text - always white for readability
  const getAccentColor = () => {
    return '#ffffff'; // Always white for best readability
  };
  useEffect(() => {
    if (calculateOptimalColumns.length > 0) {
      const optimalColumnCount = calculateOptimalColumns[0];
      setColumnCount(optimalColumnCount);
      document.documentElement.style.setProperty(
        '--column-count',
        optimalColumnCount
      );
    }
  }, [calculateOptimalColumns]);

  // Initialize arrangement on load
  useEffect(() => {
    if (items.length > 0) {
      const arrangement = items.reduce((acc, item, index) => {
        const id = item.isEmpty
          ? item.id
          : item.id.replace(/^(track-|drum-)/, '');
        acc[id] = {
          position: index,
          row: Math.floor(index / columnCount),
          column: index % columnCount,
          type: item.isEmpty ? 'empty' : (item.id.startsWith('drum-') ? 'drum' : 'track'),
          isEmpty: item.isEmpty || false,
        };
        return acc;
      }, {});
      onArrangementChange(arrangement);
    }
  }, [items, columnCount, onArrangementChange]);

  const getTitlePositionStyle = () => {
    switch (previewStyle.titlePosition) {
      case 'bottom-center':
        return { left: '50%', bottom: '16px', transform: 'translateX(-50%)' };
      case 'center':
        return { left: '50%', top: '50%', transform: 'translate(-50%, -50%)' };
      default:
        return { left: '50%', top: '14px', transform: 'translateX(-50%)' };
    }
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

  return (
    <div className='grid-container'>
      <DndContext
        sensors={sensors}
        collisionDetection={closestCenter}
        onDragEnd={handleDragEnd}
      >
        <div
          className='grid-preview-stage'
          style={{ background: previewStyle.backgroundColor || '#0a0a0f' }}
        >
          <div
            className='grid'
            style={{
              display: 'grid',
              gridTemplateColumns: `repeat(${columnCount}, 1fr)`,
              gap: '8px',
              aspectRatio: '16/9',
              width: '100%',
              height: 'auto',
              maxHeight: '100%',
            }}
          >
            <SortableContext items={items} strategy={rectSortingStrategy}>
              {items.map((item) => {
                const intensity = getHeatIntensity(item.count);
                // Derive the instrumentVideos key from the item id/name
                const videoKey = item.id.startsWith('drum-')
                  ? item.id.replace('drum-', '')
                  : (item.name || '').toLowerCase().replace(/\s+/g, '_');
                const videoUrl = instrumentVideos?.[videoKey] || null;
                return (
                  <SortableItem
                    key={item.id}
                    id={item.id}
                    item={item}
                    getHeatColor={
                      item.isEmpty
                        ? 'transparent'
                        : getHeatColor(intensity)
                    }
                    accentColor={
                      item.isEmpty
                        ? '#9ca3af'
                        : getAccentColor(intensity)
                    }
                    isEmpty={item.isEmpty}
                    clipStyle={clipStyles?.[item.id]}
                    onClipStyleChange={(newStyle) => onClipStyleChange?.(item.id, newStyle)}
                    videoUrl={videoUrl}
                    isPreviewPlaying={isPreviewPlaying}
                    activeLevel={activeLevels?.[videoKey]}
                  />
                );
              })}
            </SortableContext>
          </div>

          <div className='grid-preview-overlay' aria-hidden='true'>
            {previewStyle.vignetteEnabled && (
              <div
                className='grid-preview-vignette'
                style={{ opacity: Math.min(Math.max(previewStyle.vignetteStrength ?? 0.5, 0.1), 1) }}
              />
            )}

            {previewStyle.glitchEnabled && (
              <div
                className={`grid-preview-glitch grid-preview-glitch--${previewStyle.glitchIntensity || 'subtle'}`}
              />
            )}

            {previewStyle.titleEnabled && titleText && !showIntroCard && (
              <div
                className={[
                  titleUsesCard ? 'grid-preview-title-card' : 'grid-preview-title',
                  previewStyle.titleAnimated ? 'grid-preview-title--fade-in' : '',
                  (previewStyle.titleDuration ?? 0) > 0 ? 'grid-preview-title--fade-out' : '',
                ].filter(Boolean).join(' ')}
                style={{
                  ...getTitlePositionStyle(),
                  color: titleColor,
                  fontSize: `${previewStyle.titleFontSize}px`,
                  fontFamily: getFontFamily(titleFont),
                  ...(titleUsesCard && {
                    background: hexToRgba(titleCardBg, titleCardOpacity),
                    border: '1px solid rgba(255,255,255,0.16)',
                    borderRadius: '18px',
                    padding: '0.75rem 1.1rem',
                    boxShadow: '0 18px 48px rgba(0,0,0,0.28)',
                    backdropFilter: 'blur(14px)',
                  }),
                  ...((previewStyle.titleDuration ?? 0) > 0 && {
                    '--title-fade-out-delay': `${introDuration + previewStyle.titleDuration}s`,
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
                className='grid-preview-tagline'
                style={{
                  color: previewStyle.taglineColor,
                  fontSize: `${previewStyle.taglineFontSize}px`,
                  fontFamily: getFontFamily(previewStyle.taglineFont),
                  ...(previewStyle.taglineBackgroundEnabled && {
                    background: hexToRgba(previewStyle.taglineBackgroundColor || '#0c1220', previewStyle.taglineBackgroundOpacity ?? 0.72),
                    borderTop: `3px solid ${previewStyle.taglineAccentColor || '#ff4db8'}`,
                    padding: '0.65rem 1rem 0.7rem',
                    borderRadius: '14px 14px 0 0',
                    minWidth: 'min(72%, 720px)',
                    textAlign: 'center',
                    boxShadow: '0 -10px 32px rgba(0, 0, 0, 0.22)',
                  }),
                }}
              >
                {taglineText}
              </div>
            )}

            {previewStyle.watermarkEnabled && previewStyle.watermarkText?.trim() && (
              <div
                className='grid-preview-watermark'
                style={{
                  ...getWatermarkPositionStyle(),
                  color: previewStyle.watermarkColor,
                  fontSize: `${previewStyle.watermarkFontSize}px`,
                  fontFamily: getFontFamily(previewStyle.watermarkFont),
                  opacity: Math.min(Math.max(previewStyle.watermarkOpacity ?? 0.5, 0.1), 1),
                }}
              >
                {previewStyle.watermarkText}
              </div>
            )}

            {/* Intro card: full-screen overlay at start of preview playback */}
            {showIntroCard && (
              <div
                className={`grid-intro-card${previewStyle.introCardAnimated ? ' grid-intro-card--animated' : ''}`}
                style={{ background: hexToRgba(titleCardBg, Math.min(titleCardOpacity + 0.13, 0.95)) }}
              >
                <p
                  className='grid-intro-card__title'
                  style={{
                    color: titleColor,
                    fontFamily: getFontFamily(titleFont),
                  }}
                >
                  {titleText || 'Untitled'}
                </p>
                {titleSubtitleText && (
                  <p
                    className='grid-intro-card__subtext'
                    style={{
                      color: previewStyle.titleSubtitleColor || '#d8d8e6',
                      fontFamily: getFontFamily(titleFont),
                      fontSize: `${previewStyle.titleSubtitleFontSize ?? 24}px`,
                    }}
                  >
                    {titleSubtitleText}
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </DndContext>
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
      })
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
};

export default Grid;
