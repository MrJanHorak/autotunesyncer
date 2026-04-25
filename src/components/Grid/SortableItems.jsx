/* eslint-disable react/prop-types */
import { useState, useRef, useEffect, memo } from 'react';
import { createPortal } from 'react-dom';
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { DEFAULT_CLIP_STYLE, COLOR_GRADE_LABELS } from '../../js/styleDefaults';

const ClipStylePopover = ({ style, onChange, onClose, instrumentName, anchorRef }) => {
  const set = (k, v) => onChange({ ...style, [k]: v });
  const popoverRef = useRef(null);
  const [pos, setPos] = useState(null);

  // Compute fixed position from the anchor button's screen coordinates
  useEffect(() => {
    if (anchorRef?.current) {
      const rect = anchorRef.current.getBoundingClientRect();
      const popoverWidth = 260;
      const left = Math.max(8, Math.min(rect.right - popoverWidth, window.innerWidth - popoverWidth - 8));
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
        <button className='clip-style-popover__close' onClick={onClose}>✕</button>
      </div>

      {/* Border */}
      <div className='clip-style-row'>
        <label>Border</label>
        <div className='clip-style-row__controls'>
          <input type='color' value={style.borderColor} onChange={(e) => set('borderColor', e.target.value)} title='Border color' />
          <input type='range' min={0} max={8} value={style.borderWidth} onChange={(e) => set('borderWidth', +e.target.value)} title='Border width (0 = off)' />
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
              checked={style.bgColor != null}
              onChange={(e) => set('bgColor', e.target.checked ? '#1a1a2e' : null)}
              title='Enable custom idle background color'
            />
            <span className='csp-toggle__slider' />
          </label>
          {style.bgColor != null ? (
            <>
              <input type='color' value={style.bgColor} onChange={(e) => set('bgColor', e.target.value)} title='Background color when clip is idle' />
              <span className='clip-style-hint' style={{ color: 'var(--color-text-muted)', fontSize: '0.72rem' }}>when idle</span>
            </>
          ) : (
            <span className='clip-style-hint' style={{ color: 'var(--color-text-muted)', fontSize: '0.72rem' }}>transparent (global bg)</span>
          )}
        </div>
      </div>

      {/* Rounded corners */}
      <div className='clip-style-row'>
        <label>Rounded</label>
        <div className='clip-style-row__controls'>
          <label className='csp-toggle' style={{ marginRight: '0.5rem' }}>
            <input type='checkbox' checked={style.roundedCorners} onChange={(e) => set('roundedCorners', e.target.checked)} />
            <span className='csp-toggle__slider' />
          </label>
          {style.roundedCorners && (
            <>
              <input type='range' min={4} max={32} value={style.cornerRadius} onChange={(e) => set('cornerRadius', +e.target.value)} />
              <span className='clip-style-hint'>{style.cornerRadius}px</span>
            </>
          )}
        </div>
      </div>

      {/* Color grade */}
      <div className='clip-style-row'>
        <label>Color Grade</label>
        <div className='clip-style-row__controls'>
          <select className='clip-style-select' value={style.colorGrade} onChange={(e) => set('colorGrade', e.target.value)}>
            {Object.entries(COLOR_GRADE_LABELS).map(([k, v]) => (
              <option key={k} value={k}>{v}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Label */}
      <div className='clip-style-row'>
        <label>Label</label>
        <div className='clip-style-row__controls'>
          <label className='csp-toggle' style={{ marginRight: '0.5rem' }}>
            <input type='checkbox' checked={style.labelEnabled} onChange={(e) => set('labelEnabled', e.target.checked)} />
            <span className='csp-toggle__slider' />
          </label>
          {style.labelEnabled && (
            <>
              <input className='clip-style-text-input' type='text' value={style.labelText} onChange={(e) => set('labelText', e.target.value)} placeholder={instrumentName} maxLength={30} />
              <input type='color' value={style.labelColor} onChange={(e) => set('labelColor', e.target.value)} title='Label color' />
            </>
          )}
        </div>
      </div>

      {/* Beat flash */}
      <div className='clip-style-row'>
        <label>Beat Flash</label>
        <div className='clip-style-row__controls'>
          <label className='csp-toggle' style={{ marginRight: '0.5rem' }}>
            <input type='checkbox' checked={style.beatFlashEnabled} onChange={(e) => set('beatFlashEnabled', e.target.checked)} />
            <span className='csp-toggle__slider' />
          </label>
          {style.beatFlashEnabled && (
            <>
              <input type='color' value={style.beatFlashColor} onChange={(e) => set('beatFlashColor', e.target.value)} title='Flash color' />
              <input type='range' min={0.1} max={1} step={0.05} value={style.beatFlashIntensity} onChange={(e) => set('beatFlashIntensity', +e.target.value)} />
              <span className='clip-style-hint'>{Math.round(style.beatFlashIntensity * 100)}%</span>
            </>
          )}
        </div>
      </div>

      {/* Fade */}
      <div className='clip-style-row'>
        <label>Clip Fade</label>
        <div className='clip-style-row__controls'>
          <label className='csp-toggle'>
            <input type='checkbox' checked={style.fadeEnabled} onChange={(e) => set('fadeEnabled', e.target.checked)} />
            <span className='csp-toggle__slider' />
          </label>
        </div>
      </div>

      {/* Transparent background */}
      <div className='clip-style-row'>
        <label>Transparent Bg</label>
        <div className='clip-style-row__controls'>
          <label className='csp-toggle'>
            <input type='checkbox' checked={style.transparentBg} onChange={(e) => set('transparentBg', e.target.checked)} />
            <span className='csp-toggle__slider' />
          </label>
          <span className='clip-style-hint' style={{ color: 'var(--color-text-muted)', fontSize: '0.72rem' }}>
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
    document.body
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
}) {
  const [showStylePicker, setShowStylePicker] = useState(false);
  const videoRef = useRef(null);
  const wasActiveRef = useRef(false);
  const btnRef = useRef(null);

  const { attributes, listeners, setNodeRef, transform, transition } =
    useSortable({ id });

  // Opacity logic:
  //   idle (no preview)      → 0.35, looping
  //   preview + note active  → 0.7, playing from note start
  //   preview + note silent  → 0, hidden
  const ACTIVE_THRESHOLD_DB = -45;
  const isInstrumentActive = isPreviewPlaying
    ? (activeLevel !== undefined && activeLevel > ACTIVE_THRESHOLD_DB)
    : false;

  const videoOpacity = !isPreviewPlaying ? 0.35 : isInstrumentActive ? 0.7 : 0;

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

  // When preview toggles: restore idle loop or pause to wait for first note
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !videoUrl) return;
    if (!isPreviewPlaying) {
      wasActiveRef.current = false;
      video.play().catch(() => {});
    } else {
      video.pause();
      wasActiveRef.current = false;
    }
  }, [isPreviewPlaying, videoUrl]);

  // Note onset/release during preview — reset to 0 on each new note
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !videoUrl || !isPreviewPlaying) return;

    const isActive = activeLevel !== undefined && activeLevel > ACTIVE_THRESHOLD_DB;

    if (isActive && !wasActiveRef.current) {
      // Note just started — jump to beginning and play
      video.currentTime = 0;
      video.play().catch(() => {});
    } else if (!isActive && wasActiveRef.current) {
      // Note just ended — pause the clip
      video.pause();
    }

    wasActiveRef.current = isActive;
  }, [activeLevel, isPreviewPlaying, videoUrl]);

  const cs = clipStyle || DEFAULT_CLIP_STYLE;

  const cellStyle = {
    transform: transform ? CSS.Transform.toString(transform) : '',
    transition,
    background: isEmpty ? '#f3f4f6' : getHeatColor,
    borderRadius: cs.roundedCorners ? `${cs.cornerRadius}px` : '12px',
    aspectRatio: '16/9',
    border: cs.borderWidth > 0 ? `${cs.borderWidth}px solid ${cs.borderColor}` : 'none',
    boxSizing: 'border-box',
    position: 'relative',
    overflow: 'hidden',
  };

  const cellContentStyle = {
    '--accent-color': accentColor,
  };

  return (
    <div
      ref={setNodeRef}
      style={cellStyle}
      className={`grid-cell ${isEmpty ? 'empty' : ''}`}
      {...attributes}
      {...listeners}
    >
      {/* Semi-transparent video overlay — pointer events disabled so DnD works */}
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
            opacity: videoOpacity,
            transition: 'opacity 0.08s ease',
            pointerEvents: 'none',
            zIndex: 0,
            borderRadius: 'inherit',
          }}
        />
      )}

      {!isEmpty && (
        <>
          <div className='cell-content' style={{ ...cellContentStyle, position: 'relative', zIndex: 1 }}>
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
            onClick={(e) => { e.stopPropagation(); setShowStylePicker((s) => !s); }}
          >
            🎨
          </button>

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

