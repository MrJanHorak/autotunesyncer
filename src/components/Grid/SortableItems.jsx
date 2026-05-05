/* eslint-disable react/prop-types */
import { useState, useRef, useEffect, useId, memo } from 'react';
import { createPortal } from 'react-dom';
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { DEFAULT_CLIP_STYLE, COLOR_GRADE_LABELS } from '../../js/styleDefaults';

const SVG_COLOR_GRADES = new Set([
  'warm',
  'cool',
  'vintage',
  'cyberpunk',
  'vivid',
]);

const getClipColorFilter = (colorGrade, filterIds) => {
  switch (colorGrade) {
    case 'warm':
      return `url(#${filterIds.warm})`;
    case 'cool':
      return `url(#${filterIds.cool})`;
    case 'vintage':
      return `url(#${filterIds.vintage})`;
    case 'cyberpunk':
      return `url(#${filterIds.cyberpunk})`;
    case 'bw':
      return 'grayscale(1)';
    case 'vivid':
      return `url(#${filterIds.vivid})`;
    default:
      return 'none';
  }
};

const ClipColorGradeFilterDefs = memo(function ClipColorGradeFilterDefs({
  filterIds,
}) {
  return (
    <svg
      aria-hidden='true'
      focusable='false'
      width='0'
      height='0'
      style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}
    >
      <defs>
        <filter id={filterIds.warm} colorInterpolationFilters='sRGB'>
          <feColorMatrix type='saturate' values='1.2' />
          <feComponentTransfer>
            <feFuncR type='gamma' amplitude='1' exponent='0.9091' offset='0' />
            <feFuncG type='identity' />
            <feFuncB type='gamma' amplitude='1' exponent='1.1364' offset='0' />
            <feFuncA type='identity' />
          </feComponentTransfer>
        </filter>

        <filter id={filterIds.cool} colorInterpolationFilters='sRGB'>
          <feColorMatrix type='saturate' values='1.1' />
          <feComponentTransfer>
            <feFuncR type='gamma' amplitude='1' exponent='1.1364' offset='0' />
            <feFuncG type='identity' />
            <feFuncB type='gamma' amplitude='1' exponent='0.8696' offset='0' />
            <feFuncA type='identity' />
          </feComponentTransfer>
        </filter>

        <filter id={filterIds.vintage} colorInterpolationFilters='sRGB'>
          <feColorMatrix
            type='matrix'
            values='
              0.393 0.769 0.189 0 0
              0.349 0.686 0.168 0 0
              0.272 0.534 0.131 0 0
              0     0     0     1 0
            '
          />
        </filter>

        <filter id={filterIds.cyberpunk} colorInterpolationFilters='sRGB'>
          <feColorMatrix type='saturate' values='1.6' />
          <feComponentTransfer>
            <feFuncR type='linear' slope='1.1' intercept='-0.05' />
            <feFuncG type='linear' slope='1.1' intercept='-0.05' />
            <feFuncB type='linear' slope='1.1' intercept='-0.05' />
            <feFuncA type='identity' />
          </feComponentTransfer>
          <feColorMatrix
            type='matrix'
            values='
              1.15 0    0.15 0 0
              0    1.15 0.15 0 0
              0.25 0    1    0 0
              0    0    0    1 0
            '
          />
        </filter>

        <filter id={filterIds.vivid} colorInterpolationFilters='sRGB'>
          <feColorMatrix type='saturate' values='1.8' />
          <feComponentTransfer>
            <feFuncR type='linear' slope='1.1' intercept='-0.01' />
            <feFuncG type='linear' slope='1.1' intercept='-0.01' />
            <feFuncB type='linear' slope='1.1' intercept='-0.01' />
            <feFuncA type='identity' />
          </feComponentTransfer>
        </filter>
      </defs>
    </svg>
  );
});

const ClipStylePopover = ({
  style,
  onChange,
  onClose,
  instrumentName,
  anchorRef,
}) => {
  const set = (k, v) => onChange({ ...style, [k]: v });
  const popoverRef = useRef(null);
  const [pos, setPos] = useState(null);

  // Compute fixed position from the anchor button's screen coordinates
  useEffect(() => {
    if (anchorRef?.current) {
      const rect = anchorRef.current.getBoundingClientRect();
      const popoverWidth = 260;
      const left = Math.max(
        8,
        Math.min(
          rect.right - popoverWidth,
          window.innerWidth - popoverWidth - 8,
        ),
      );
      setPos({ top: rect.bottom + 4, left });
    }
  }, [anchorRef]);

  useEffect(() => {
    const handler = (e) => {
      if (
        popoverRef.current &&
        !popoverRef.current.contains(e.target) &&
        !anchorRef?.current?.contains(e.target)
      ) {
        onClose();
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [onClose, anchorRef]);

  if (!pos) return null;

  return createPortal(
    <div
      ref={popoverRef}
      className='clip-style-popover'
      style={{ position: 'fixed', top: pos.top, left: pos.left, zIndex: 9999 }}
      onMouseDown={(e) => e.stopPropagation()}
      onPointerDown={(e) => e.stopPropagation()}
    >
      <div className='clip-style-popover__header'>
        <span className='clip-style-popover__title'>🎨 {instrumentName}</span>
        <button className='clip-style-popover__close' onClick={onClose}>
          ✕
        </button>
      </div>

      {/* Border */}
      <div className='clip-style-row'>
        <label>Border</label>
        <div className='clip-style-row__controls'>
          <input
            type='color'
            value={style.borderColor}
            onChange={(e) => set('borderColor', e.target.value)}
            title='Border color'
          />
          <input
            type='range'
            min={0}
            max={8}
            value={style.borderWidth}
            onChange={(e) => set('borderWidth', +e.target.value)}
            title='Border width (0 = off)'
          />
          <span className='clip-style-hint'>{style.borderWidth}px</span>
        </div>
      </div>

      {/* Gap-fill / bg color */}
      <div className='clip-style-row'>
        <label>Gap Fill</label>
        <div className='clip-style-row__controls'>
          <label className='csp-toggle' style={{ marginRight: '0.5rem' }}>
            <input
              type='checkbox'
              checked={!!style.bgColorEnabled}
              onChange={(e) => set('bgColorEnabled', e.target.checked)}
              title='Enable custom idle background color'
            />
            <span className='csp-toggle__slider' />
          </label>
          {style.bgColorEnabled ? (
            <>
              <input
                type='color'
                value={style.bgColor || '#1a1a2e'}
                onChange={(e) => set('bgColor', e.target.value)}
                title='Background color when clip is idle'
              />
              <span
                className='clip-style-hint'
                style={{
                  color: 'var(--color-text-muted)',
                  fontSize: '0.72rem',
                }}
              >
                when idle
              </span>
            </>
          ) : (
            <span
              className='clip-style-hint'
              style={{ color: 'var(--color-text-muted)', fontSize: '0.72rem' }}
            >
              transparent (global bg)
            </span>
          )}
        </div>
      </div>

      {/* Rounded corners */}
      <div className='clip-style-row'>
        <label>Rounded</label>
        <div className='clip-style-row__controls'>
          <label className='csp-toggle' style={{ marginRight: '0.5rem' }}>
            <input
              type='checkbox'
              checked={style.roundedCorners}
              onChange={(e) => set('roundedCorners', e.target.checked)}
            />
            <span className='csp-toggle__slider' />
          </label>
          {style.roundedCorners && (
            <>
              <input
                type='range'
                min={4}
                max={32}
                value={style.cornerRadius}
                onChange={(e) => set('cornerRadius', +e.target.value)}
              />
              <span className='clip-style-hint'>{style.cornerRadius}px</span>
            </>
          )}
        </div>
      </div>

      {/* Color grade */}
      <div className='clip-style-row'>
        <label>Color Grade</label>
        <div className='clip-style-row__controls'>
          <select
            className='clip-style-select'
            value={style.colorGrade}
            onChange={(e) => set('colorGrade', e.target.value)}
          >
            {Object.entries(COLOR_GRADE_LABELS).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Label */}
      <div className='clip-style-row'>
        <label>Label</label>
        <div className='clip-style-row__controls'>
          <label className='csp-toggle' style={{ marginRight: '0.5rem' }}>
            <input
              type='checkbox'
              checked={style.labelEnabled}
              onChange={(e) => set('labelEnabled', e.target.checked)}
            />
            <span className='csp-toggle__slider' />
          </label>
          {style.labelEnabled && (
            <>
              <input
                className='clip-style-text-input'
                type='text'
                value={style.labelText}
                onChange={(e) => set('labelText', e.target.value)}
                placeholder={instrumentName}
                maxLength={30}
              />
              <input
                type='color'
                value={style.labelColor}
                onChange={(e) => set('labelColor', e.target.value)}
                title='Label color'
              />
            </>
          )}
        </div>
      </div>

      {/* Beat flash */}
      <div className='clip-style-row'>
        <label>Beat Flash</label>
        <div className='clip-style-row__controls'>
          <label className='csp-toggle' style={{ marginRight: '0.5rem' }}>
            <input
              type='checkbox'
              checked={style.beatFlashEnabled}
              onChange={(e) => set('beatFlashEnabled', e.target.checked)}
            />
            <span className='csp-toggle__slider' />
          </label>
          {style.beatFlashEnabled && (
            <>
              <input
                type='color'
                value={style.beatFlashColor}
                onChange={(e) => set('beatFlashColor', e.target.value)}
                title='Flash color'
              />
              <input
                type='range'
                min={0.1}
                max={1}
                step={0.05}
                value={style.beatFlashIntensity}
                onChange={(e) => set('beatFlashIntensity', +e.target.value)}
              />
              <span className='clip-style-hint'>
                {Math.round(style.beatFlashIntensity * 100)}%
              </span>
            </>
          )}
        </div>
      </div>

      {/* Fade */}
      <div className='clip-style-row'>
        <label>Clip Fade</label>
        <div className='clip-style-row__controls'>
          <label className='csp-toggle'>
            <input
              type='checkbox'
              checked={style.fadeEnabled}
              onChange={(e) => set('fadeEnabled', e.target.checked)}
            />
            <span className='csp-toggle__slider' />
          </label>
        </div>
      </div>

      {/* Transparent background */}
      <div className='clip-style-row'>
        <label>Transparent Bg</label>
        <div className='clip-style-row__controls'>
          <label className='csp-toggle'>
            <input
              type='checkbox'
              checked={style.transparentBg}
              onChange={(e) => set('transparentBg', e.target.checked)}
            />
            <span className='csp-toggle__slider' />
          </label>
          <span
            className='clip-style-hint'
            style={{ color: 'var(--color-text-muted)', fontSize: '0.72rem' }}
          >
            {style.transparentBg ? 'global bg' : 'clip bg color'}
          </span>
        </div>
      </div>

      <button
        className='clip-style-reset'
        onClick={() => onChange({ ...DEFAULT_CLIP_STYLE })}
      >
        ↺ Reset clip style
      </button>
    </div>,
    document.body,
  );
};

export const SortableItem = memo(function SortableItem({
  id,
  item,
  getHeatColor,
  accentColor,
  isEmpty,
  clipStyle,
  onClipStyleChange,
  videoUrl,
  isPreviewPlaying,
  activeLevel,
  beatPulseClass,
  isEditable = true,
}) {
  const { attributes, listeners, setNodeRef, transform, transition } =
    useSortable({ id });

  return (
    <GridClipItem
      id={id}
      item={item}
      getHeatColor={getHeatColor}
      accentColor={accentColor}
      isEmpty={isEmpty}
      clipStyle={clipStyle}
      onClipStyleChange={onClipStyleChange}
      videoUrl={videoUrl}
      isPreviewPlaying={isPreviewPlaying}
      activeLevel={activeLevel}
      beatPulseClass={beatPulseClass}
      isEditable={isEditable}
      containerRef={setNodeRef}
      containerProps={{ ...attributes, ...listeners }}
      containerStyle={{
        transform: transform ? CSS.Transform.toString(transform) : '',
        transition: isPreviewPlaying ? 'none' : transition,
      }}
      fillParent={false}
    />
  );
});

export const GridClipItem = memo(function GridClipItem({
  id,
  item,
  getHeatColor,
  accentColor,
  isEmpty,
  clipStyle,
  onClipStyleChange,
  videoUrl,
  isPreviewPlaying,
  activeLevel,
  beatPulseClass,
  isEditable = true,
  containerRef = null,
  containerProps = {},
  containerStyle = null,
  fillParent = true,
}) {
  const [showStylePicker, setShowStylePicker] = useState(false);
  const videoRef = useRef(null);
  const wasActiveRef = useRef(false);
  const btnRef = useRef(null);
  const gradeFilterIdSeed = useId().replace(/[^a-zA-Z0-9_-]/g, '');

  // Opacity logic:
  //   idle (no preview)      → 0.35, looping
  //   preview + note active  → 0.78, visible
  //   preview + note silent  → 0, hidden (video paused too, so no frozen frame)
  const ACTIVE_THRESHOLD_DB = -45;
  const isInstrumentActive = isPreviewPlaying
    ? activeLevel !== undefined && activeLevel > ACTIVE_THRESHOLD_DB
    : false;

  // Not playing → dim idle loop (0.35)
  // Preview + active → full brightness (1)
  // Preview + silent → hidden (0)
  const videoOpacity = !isPreviewPlaying ? 0.35 : isInstrumentActive ? 1 : 0;

  // Start idle loop on initial mount (once video is ready)
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !videoUrl || isPreviewPlaying) return;
    const start = () => video.play().catch(() => {});
    if (video.readyState >= 2) {
      start();
    } else {
      video.addEventListener('canplay', start, { once: true });
      return () => video.removeEventListener('canplay', start);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoUrl]);

  // When preview toggles: restore idle loop or pause+hide to wait for first note.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !videoUrl) return;
    wasActiveRef.current = false;
    if (!isPreviewPlaying) {
      video.play().catch(() => {});
    } else {
      video.pause();
      video.currentTime = 0;
    }
  }, [isPreviewPlaying, videoUrl]);

  // Note onset/release during preview.
  // On note start: jump to beginning and play. On note end: pause and reset so
  // no frozen frame is visible when opacity drops back to 0.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !videoUrl || !isPreviewPlaying) return;

    const isActive =
      activeLevel !== undefined && activeLevel > ACTIVE_THRESHOLD_DB;

    if (isActive && !wasActiveRef.current) {
      video.currentTime = 0;
      video.play().catch(() => {});
    } else if (!isActive && wasActiveRef.current) {
      video.pause();
      video.currentTime = 0;
    }

    wasActiveRef.current = isActive;
  }, [activeLevel, isPreviewPlaying, videoUrl]);

  const cs = clipStyle || DEFAULT_CLIP_STYLE;
  const gradeFilterIds = {
    warm: `clip-grade-${gradeFilterIdSeed}-warm`,
    cool: `clip-grade-${gradeFilterIdSeed}-cool`,
    vintage: `clip-grade-${gradeFilterIdSeed}-vintage`,
    cyberpunk: `clip-grade-${gradeFilterIdSeed}-cyberpunk`,
    vivid: `clip-grade-${gradeFilterIdSeed}-vivid`,
  };
  const usesSvgColorGrade = SVG_COLOR_GRADES.has(cs.colorGrade);
  const videoFilter = getClipColorFilter(cs.colorGrade, gradeFilterIds);
  const clipBackground =
    !cs.transparentBg && cs.bgColorEnabled && cs.bgColor
      ? cs.bgColor
      : 'transparent';
  const fadeDuration = Math.max(cs.fadeDuration ?? 0.15, 0.05);
  const videoVisible =
    !isPreviewPlaying || isInstrumentActive || cs.fadeEnabled;
  const showClipBorder =
    cs.borderWidth > 0 && (!isPreviewPlaying || isInstrumentActive);
  const previewLabelText = (cs.labelText || item.name || '').trim();
  const videoEffectFilter = [
    videoFilter !== 'none' ? videoFilter : '',
    cs.fadeEnabled && isPreviewPlaying && isInstrumentActive
      ? 'brightness(1.12)'
      : '',
  ]
    .filter(Boolean)
    .join(' ');
  const beatFlashOpacity =
    isPreviewPlaying && isInstrumentActive && cs.beatFlashEnabled
      ? Math.min(Math.max(cs.beatFlashIntensity ?? 0.4, 0), 1)
      : 0;

  const cellStyle = {
    background: isPreviewPlaying
      ? isEmpty
        ? 'transparent'
        : clipBackground
      : isEmpty
        ? '#f3f4f6'
        : getHeatColor,
    borderRadius: cs.roundedCorners ? `${cs.cornerRadius}px` : '12px',
    ...(fillParent
      ? { width: '100%', height: '100%' }
      : { aspectRatio: '16/9' }),
    border: showClipBorder
      ? `${cs.borderWidth}px solid ${cs.borderColor}`
      : 'none',
    boxSizing: 'border-box',
    position: 'relative',
    overflow: 'hidden',
    ...(containerStyle || {}),
  };

  const cellContentStyle = {
    '--accent-color': accentColor,
  };

  return (
    <div
      ref={containerRef}
      style={cellStyle}
      className={`grid-cell ${isEmpty ? 'empty' : ''} ${isPreviewPlaying ? 'preview-active' : ''} ${beatPulseClass || ''}`}
      {...containerProps}
    >
      {!isEmpty && usesSvgColorGrade && (
        <ClipColorGradeFilterDefs filterIds={gradeFilterIds} />
      )}

      {!isEmpty && !isPreviewPlaying && isEditable && (
        <div
          className='grid-cell__drag-handle'
          title='Drag clip to reposition'
          aria-hidden='true'
        >
          <span className='grid-cell__drag-dot' />
          <span className='grid-cell__drag-dot' />
          <span className='grid-cell__drag-dot' />
          <span className='grid-cell__drag-dot' />
        </div>
      )}

      {/* Video — use display:none (not opacity:0) when hidden so the browser's
           default black video background can't bleed through transparent cells */}
      {!isEmpty && videoUrl && (
        <video
          ref={videoRef}
          src={videoUrl}
          loop
          muted
          playsInline
          preload='auto'
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            display: videoVisible ? 'block' : 'none',
            opacity: !isPreviewPlaying ? 0.35 : isInstrumentActive ? 1 : 0,
            transition: cs.fadeEnabled
              ? `opacity ${fadeDuration}s ease, filter ${fadeDuration}s ease`
              : 'opacity 0.08s ease',
            pointerEvents: 'none',
            zIndex: 0,
            borderRadius: 'inherit',
            filter: videoEffectFilter || 'none',
          }}
        />
      )}

      {!isEmpty && (
        <div
          className='clip-beat-flash'
          style={{
            background: cs.beatFlashColor || '#ffffff',
            opacity: beatFlashOpacity,
          }}
        />
      )}

      {!isEmpty && isPreviewPlaying && cs.labelEnabled && previewLabelText && (
        <div
          style={{
            position: 'absolute',
            left: '6px',
            bottom: '6px',
            zIndex: 2,
            maxWidth: 'calc(100% - 12px)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            padding: '3px 6px',
            borderRadius: '6px',
            background: 'rgba(0, 0, 0, 0.45)',
            color: cs.labelColor || '#ffffff',
            fontSize: `${cs.labelFontSize ?? 14}px`,
            fontWeight: 600,
            lineHeight: 1.1,
            pointerEvents: 'none',
            textShadow: '0 1px 2px rgba(0, 0, 0, 0.45)',
          }}
        >
          {previewLabelText}
        </div>
      )}

      {!isEmpty && (
        <>
          {/* Text and style button — hidden during preview for a clean stage look */}
          {!isPreviewPlaying && (
            <>
              <div
                className='cell-content'
                style={{ ...cellContentStyle, position: 'relative', zIndex: 1 }}
              >
                <span className='cell-name'>{item.name}</span>
                <span className='cell-count'>{item.count} notes</span>
              </div>

              {/* Palette button — stops drag propagation */}
              <button
                ref={btnRef}
                className='cell-style-btn'
                title='Style this clip'
                style={{ position: 'relative', zIndex: 2 }}
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  setShowStylePicker((s) => !s);
                }}
              >
                🎨
              </button>
            </>
          )}

          {showStylePicker && (
            <ClipStylePopover
              style={cs}
              onChange={onClipStyleChange}
              onClose={() => setShowStylePicker(false)}
              instrumentName={item.name}
              anchorRef={btnRef}
            />
          )}
        </>
      )}
    </div>
  );
});
