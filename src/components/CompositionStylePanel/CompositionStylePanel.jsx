import { useState, useId, cloneElement, isValidElement } from 'react';
import PropTypes from 'prop-types';
import {
  COLOR_THEMES,
  DEFAULT_COMPOSITION_STYLE,
  FONT_OPTIONS,
} from '../../js/styleDefaults';
import { useStylePresets } from '../../hooks/useStylePresets';
import './CompositionStylePanel.css';

const Section = ({ title, icon, children, defaultOpen = false }) => {
  const [open, setOpen] = useState(defaultOpen);
  const bodyId = `csp-section-${title.toLowerCase().replace(/\s+/g, '-')}`;
  return (
    <div className='csp-section'>
      <button
        className='csp-section__header'
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-controls={bodyId}
      >
        <span className='csp-section__icon'>{icon}</span>
        <span className='csp-section__title'>{title}</span>
        <span className='csp-section__chevron'>{open ? '▾' : '▸'}</span>
      </button>
      {open && (
        <div id={bodyId} className='csp-section__body'>
          {children}
        </div>
      )}
    </div>
  );
};

Section.propTypes = {
  title: PropTypes.string.isRequired,
  icon: PropTypes.string.isRequired,
  children: PropTypes.node.isRequired,
  defaultOpen: PropTypes.bool,
};

const Field = ({ label, children }) => {
  const id = useId();
  // Pass the generated id down to the single child control (Toggle, FontSelect,
  // <select>, <input>) so the <label htmlFor> association is always correct.
  const control = isValidElement(children)
    ? cloneElement(children, { id })
    : children;
  return (
    <div className='csp-field'>
      <label className='csp-field__label' htmlFor={id}>
        {label}
      </label>
      <div className='csp-field__control'>{control}</div>
    </div>
  );
};

Field.propTypes = {
  label: PropTypes.string.isRequired,
  children: PropTypes.node.isRequired,
};

const Toggle = ({ id, checked, onChange }) => (
  <label className='csp-toggle'>
    <input
      id={id}
      type='checkbox'
      checked={checked}
      onChange={(e) => onChange(e.target.checked)}
    />
    <span className='csp-toggle__slider' />
  </label>
);
Toggle.propTypes = {
  id: PropTypes.string,
  checked: PropTypes.bool.isRequired,
  onChange: PropTypes.func.isRequired,
};

const FontSelect = ({ id, value, onChange }) => (
  <select
    id={id}
    className='csp-select'
    value={value || 'default'}
    onChange={(e) => onChange(e.target.value)}
  >
    {FONT_OPTIONS.map((f) => (
      <option key={f.value} value={f.value}>
        {f.label}
      </option>
    ))}
  </select>
);
FontSelect.propTypes = {
  id: PropTypes.string,
  value: PropTypes.string,
  onChange: PropTypes.func.isRequired,
};

const CompositionStylePanel = ({
  style,
  onChange,
  autoTransitionIntervalSeconds,
  autoTransitionReason,
}) => {
  const set = (key, val) => onChange({ ...style, [key]: val });
  const { presets, savePreset, applyPreset, deletePreset } = useStylePresets();
  const [selectedPreset, setSelectedPreset] = useState('');

  const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
  const rand = (min, max, step = 1) => {
    const steps = Math.round((max - min) / step);
    return Number(
      (min + Math.floor(Math.random() * (steps + 1)) * step).toFixed(3),
    );
  };

  const applyTheme = (theme) => {
    const t = COLOR_THEMES[theme];
    if (!t) return;
    onChange({
      ...style,
      colorTheme: theme,
      backgroundColor: t.backgroundColor,
      titleColor: t.titleColor,
      taglineColor: t.taglineColor,
      watermarkColor: t.watermarkColor,
    });
  };

  const resetToDefaults = () => onChange({ ...DEFAULT_COMPOSITION_STYLE });

  const handleSavePreset = () => {
    const name = window.prompt('Preset name:')?.trim();
    if (!name) return;
    savePreset(name, style);
    setSelectedPreset(name);
  };

  const handleApplyPreset = () => {
    const applied = applyPreset(selectedPreset);
    if (applied) onChange(applied);
  };

  const handleDeletePreset = () => {
    if (!selectedPreset) return;
    if (!window.confirm(`Delete preset "${selectedPreset}"?`)) return;
    deletePreset(selectedPreset);
    setSelectedPreset('');
  };

  const handleRemixStyle = () => {
    const titlePresets = [
      'fade',
      'scroll-up',
      'scroll-left',
      'bounce',
      'spin-soft',
      'blur-focus',
    ];
    const titleDurations = [0, 4, 5, 6, 8, 10, 12];
    const glitchModes = ['subtle', 'medium', 'heavy'];
    const transitionModes = [
      'crossfade',
      'dip-black',
      'dip-white',
      'push-left',
      'push-right',
      'zoom',
      'glitch-cut',
    ];
    const intensities = ['low', 'medium', 'high'];
    const transitionTimingModes = ['start', 'interval', 'auto'];
    const directions = ['left', 'right'];
    const easings = [
      'linear',
      'ease-out',
      'ease-in-out',
      'cubic-bezier(0.22, 1, 0.36, 1)',
    ];

    onChange({
      ...style,
      titleEnabled: true,
      titleAnimated: true,
      titleAnimationPreset: pick(titlePresets),
      titleAnimDuration: rand(0.4, 1.8, 0.1),
      titleAnimDelay: rand(0, 0.9, 0.1),
      titleAnimIntensity: pick(intensities),
      titleAnimDirection: pick(directions),
      titleAnimEasing: pick(easings),
      titleDuration: pick(titleDurations),
      titleBackgroundEnabled: Math.random() > 0.42,
      titleBackgroundMode: Math.random() > 0.65 ? 'fullscreen' : 'card',
      titleBackgroundOpacity: rand(0.25, 0.9, 0.05),
      titleGlowEnabled: Math.random() > 0.45,
      titleGlowSize: rand(4, 16, 1),
      titleShadowEnabled: true,
      titleShadowSize: rand(1, 5, 0.5),
      vignetteEnabled: Math.random() > 0.52,
      vignetteStrength: rand(0.2, 0.75, 0.05),
      glitchEnabled: Math.random() > 0.62,
      glitchIntensity: pick(glitchModes),
      transitionEnabled: Math.random() > 0.25,
      transitionPreset: pick(transitionModes),
      transitionDuration: rand(0.25, 1.2, 0.05),
      transitionStrength: pick(intensities),
      transitionOn: pick(transitionTimingModes),
      transitionSectionInterval: rand(4, 14, 1),
      beatSyncEnabled: Math.random() > 0.55,
      beatSyncSensitivity: pick(intensities),
      beatSyncTargets: [
        'title',
        'tagline',
        ...(Math.random() > 0.6 ? ['track-cells'] : []),
      ],
      beatPulseMode: pick(['scale', 'glow', 'shake-lite']),
      outroEffectEnabled: Math.random() > 0.45,
      outroEffectPreset: pick([
        'fade-black',
        'fade-white',
        'glitch-out',
        'zoom-out',
      ]),
      outroEffectDuration: rand(0.6, 2.8, 0.1),
      outroEffectStrength: pick(intensities),
      taglineEnabled: Math.random() > 0.4,
      taglineShape: pick(['rounded', 'pill', 'outline', 'accent-left']),
      taglineBackgroundEnabled: Math.random() > 0.5,
      taglineWidth: rand(56, 88, 2),
      taglineFadeInDuration: rand(0.2, 1.1, 0.1),
      taglineFadeOutDuration: rand(0.2, 1.2, 0.1),
    });
  };

  const titleIntensityLabelMap = {
    low: 'Subtle',
    medium: 'Medium',
    high: 'Bold',
  };
  const livePreviewTitleIntensity =
    titleIntensityLabelMap[style.titleAnimIntensity || 'medium'] || 'Medium';
  const transitionPresetValue =
    style.transitionPreset === 'slide-left'
      ? 'push-left'
      : style.transitionPreset === 'slide-right'
        ? 'push-right'
        : style.transitionPreset === 'zoom-in'
          ? 'zoom'
          : style.transitionPreset || 'crossfade';
  const transitionOnValue =
    style.transitionOn === 'interval' || style.transitionOn === 'sections'
      ? 'section'
      : style.transitionOn === 'auto'
        ? 'phrase'
        : style.transitionOn || 'start';

  return (
    <div className='csp'>
      <div className='csp__header'>
        <h3 className='csp__title'>🎨 Composition Style</h3>
        <div className='csp__header-actions'>
          <button
            className='csp__reset csp__remix'
            onClick={handleRemixStyle}
            title='Randomize creative style settings'
          >
            Remix
          </button>
          <button
            className='csp__reset'
            onClick={resetToDefaults}
            title='Reset all to defaults'
          >
            ↺ Reset
          </button>
        </div>
      </div>

      {/* Presets */}
      <Section title='Style Presets' icon='💾'>
        <div className='csp-presets'>
          <select
            className='csp-select csp-presets__select'
            value={selectedPreset}
            onChange={(e) => setSelectedPreset(e.target.value)}
            aria-label='Select style preset'
          >
            <option value=''>— select preset —</option>
            {presets.map((p) => (
              <option key={p.name} value={p.name}>
                {p.name}
              </option>
            ))}
          </select>
          <div className='csp-presets__actions'>
            <button
              className='csp-btn csp-btn--sm'
              onClick={handleApplyPreset}
              disabled={!selectedPreset}
              title='Apply selected preset'
            >
              Apply
            </button>
            <button
              className='csp-btn csp-btn--sm'
              onClick={handleSavePreset}
              title='Save current style as a new preset'
            >
              Save as…
            </button>
            <button
              className='csp-btn csp-btn--sm csp-btn--danger'
              onClick={handleDeletePreset}
              disabled={!selectedPreset}
              title='Delete selected preset'
            >
              Delete
            </button>
          </div>
        </div>
      </Section>

      {/* Theme Picker */}
      <Section title='Color Theme' icon='🎭' defaultOpen>
        <div className='csp-theme-grid'>
          {Object.keys(COLOR_THEMES).map((t) => (
            <button
              key={t}
              className={`csp-theme-btn${style.colorTheme === t ? ' csp-theme-btn--active' : ''}`}
              style={{
                background: COLOR_THEMES[t].backgroundColor,
                borderColor: COLOR_THEMES[t].clipDefaults.borderColor,
              }}
              onClick={() => applyTheme(t)}
            >
              <span
                className='csp-theme-btn__dot'
                style={{ background: COLOR_THEMES[t].clipDefaults.borderColor }}
              />
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>
        <Field label='Canvas Background'>
          <input
            type='color'
            value={style.backgroundColor}
            onChange={(e) => set('backgroundColor', e.target.value)}
          />
          <span className='csp-color-hex'>{style.backgroundColor}</span>
        </Field>
      </Section>

      {/* Title */}
      <Section title='Title / Intro Card' icon='🎬'>
        <Field label='Enable'>
          <Toggle
            checked={style.titleEnabled}
            onChange={(v) => set('titleEnabled', v)}
          />
        </Field>
        {style.titleEnabled && (
          <>
            <Field label='Text'>
              <input
                className='csp-input'
                type='text'
                value={style.titleText}
                onChange={(e) => set('titleText', e.target.value)}
                placeholder='Your Song Title'
                maxLength={80}
              />
            </Field>
            <Field label='Position'>
              <select
                className='csp-select'
                value={style.titlePosition}
                onChange={(e) => set('titlePosition', e.target.value)}
              >
                <option value='top-center'>Top Center</option>
                <option value='bottom-center'>Bottom Center</option>
                <option value='center'>Center</option>
              </select>
            </Field>
            <Field label='Font Size'>
              <input
                className='csp-range'
                type='range'
                min={24}
                max={96}
                value={style.titleFontSize}
                onChange={(e) => set('titleFontSize', +e.target.value)}
              />
              <span className='csp-range-val'>{style.titleFontSize}px</span>
            </Field>
            <Field label='Font'>
              <FontSelect
                value={style.titleFont}
                onChange={(v) => set('titleFont', v)}
              />
            </Field>
            <Field label='Color'>
              <input
                type='color'
                value={style.titleColor}
                onChange={(e) => set('titleColor', e.target.value)}
              />
              <span className='csp-color-hex'>{style.titleColor}</span>
            </Field>
            <Field label='Subtitle'>
              <input
                className='csp-input'
                type='text'
                value={style.titleSubtitleText ?? ''}
                onChange={(e) => set('titleSubtitleText', e.target.value)}
                placeholder='Optional line below the title'
                maxLength={120}
              />
            </Field>
            {Boolean(style.titleSubtitleText?.trim()) && (
              <>
                <Field label='Subtitle Size'>
                  <input
                    className='csp-range'
                    type='range'
                    min={14}
                    max={40}
                    value={style.titleSubtitleFontSize ?? 24}
                    onChange={(e) =>
                      set('titleSubtitleFontSize', +e.target.value)
                    }
                  />
                  <span className='csp-range-val'>
                    {style.titleSubtitleFontSize ?? 24}px
                  </span>
                </Field>
                <Field label='Subtitle Color'>
                  <input
                    type='color'
                    value={style.titleSubtitleColor || '#d8d8e6'}
                    onChange={(e) => set('titleSubtitleColor', e.target.value)}
                  />
                  <span className='csp-color-hex'>
                    {style.titleSubtitleColor || '#d8d8e6'}
                  </span>
                </Field>
              </>
            )}
            <Field label='Background'>
              <Toggle
                checked={style.titleBackgroundEnabled ?? false}
                onChange={(v) => set('titleBackgroundEnabled', v)}
              />
            </Field>
            {style.titleBackgroundEnabled && (
              <>
                <Field label='Background Style'>
                  <select
                    className='csp-select'
                    value={style.titleBackgroundMode || 'card'}
                    onChange={(e) => set('titleBackgroundMode', e.target.value)}
                  >
                    <option value='card'>Local Card</option>
                    <option value='fullscreen'>Full Screen</option>
                  </select>
                </Field>
                <Field label='Background Color'>
                  <input
                    type='color'
                    value={style.titleBackgroundColor || '#120b24'}
                    onChange={(e) =>
                      set('titleBackgroundColor', e.target.value)
                    }
                  />
                  <span className='csp-color-hex'>
                    {style.titleBackgroundColor || '#120b24'}
                  </span>
                </Field>
                <Field label='Card Opacity'>
                  <input
                    className='csp-range'
                    type='range'
                    min={0.1}
                    max={1}
                    step={0.05}
                    value={style.titleBackgroundOpacity ?? 0.82}
                    onChange={(e) =>
                      set('titleBackgroundOpacity', +e.target.value)
                    }
                  />
                  <span className='csp-range-val'>
                    {Math.round((style.titleBackgroundOpacity ?? 0.82) * 100)}%
                  </span>
                </Field>
              </>
            )}
            <Field label='Show at Start'>
              <Toggle
                checked={style.introCardEnabled}
                onChange={(v) => set('introCardEnabled', v)}
              />
            </Field>
            {style.introCardEnabled && (
              <Field label='Card Duration'>
                <input
                  className='csp-range'
                  type='range'
                  min={1}
                  max={8}
                  step={0.5}
                  value={style.introCardDuration}
                  onChange={(e) => set('introCardDuration', +e.target.value)}
                />
                <span className='csp-range-val'>
                  {style.introCardDuration}s
                </span>
              </Field>
            )}
            <Field label='Animate'>
              <Toggle
                checked={style.titleAnimated}
                onChange={(v) => set('titleAnimated', v)}
              />
            </Field>
            {style.titleAnimated && (
              <>
                <Field label='Motion Preset'>
                  <select
                    className='csp-select'
                    value={style.titleAnimationPreset || 'fade'}
                    onChange={(e) =>
                      set('titleAnimationPreset', e.target.value)
                    }
                  >
                    <option value='fade'>Fade In</option>
                    <option value='scroll-up'>Scroll Up</option>
                    <option value='scroll-left'>Scroll Left</option>
                    <option value='bounce'>Bounce In</option>
                    <option value='spin-soft'>Spin Soft</option>
                    <option value='blur-focus'>Blur Focus</option>
                    <option value='typewriter'>Typewriter</option>
                  </select>
                </Field>
                <Field label='Motion Duration'>
                  <input
                    className='csp-range'
                    type='range'
                    min={0.3}
                    max={2.5}
                    step={0.1}
                    value={style.titleAnimDuration ?? 0.7}
                    onChange={(e) => set('titleAnimDuration', +e.target.value)}
                  />
                  <span className='csp-range-val'>
                    {(style.titleAnimDuration ?? 0.7).toFixed(1)}s
                  </span>
                </Field>
                <Field label='Motion Delay'>
                  <input
                    className='csp-range'
                    type='range'
                    min={0}
                    max={2}
                    step={0.1}
                    value={style.titleAnimDelay ?? 0}
                    onChange={(e) => set('titleAnimDelay', +e.target.value)}
                  />
                  <span className='csp-range-val'>
                    {(style.titleAnimDelay ?? 0).toFixed(1)}s
                  </span>
                </Field>
                <Field label='Motion Feel'>
                  <select
                    className='csp-select'
                    value={style.titleAnimEasing || 'ease-out'}
                    onChange={(e) => set('titleAnimEasing', e.target.value)}
                  >
                    <option value='linear'>Linear</option>
                    <option value='ease-out'>Ease Out</option>
                    <option value='ease-in-out'>Ease In/Out</option>
                    <option value='cubic-bezier(0.22, 1, 0.36, 1)'>
                      Cinematic
                    </option>
                  </select>
                </Field>
                <Field label='Motion Intensity'>
                  <select
                    className='csp-select'
                    value={style.titleAnimIntensity || 'medium'}
                    onChange={(e) => set('titleAnimIntensity', e.target.value)}
                  >
                    <option value='low'>Low</option>
                    <option value='medium'>Medium</option>
                    <option value='high'>High</option>
                  </select>
                </Field>
                <p className='csp-field__hint'>
                  Live Preview: {livePreviewTitleIntensity}
                </p>
                {(style.titleAnimationPreset === 'scroll-left' ||
                  style.titleAnimationPreset === 'spin-soft') && (
                  <Field label='Direction'>
                    <select
                      className='csp-select'
                      value={style.titleAnimDirection || 'left'}
                      onChange={(e) =>
                        set('titleAnimDirection', e.target.value)
                      }
                    >
                      <option value='left'>Left</option>
                      <option value='right'>Right</option>
                    </select>
                  </Field>
                )}
              </>
            )}
            <Field label='Overlay Duration'>
              <input
                className='csp-range'
                type='range'
                min={0}
                max={15}
                step={0.5}
                value={style.titleDuration ?? 0}
                onChange={(e) => set('titleDuration', +e.target.value)}
              />
              <span className='csp-range-val'>
                {(style.titleDuration ?? 0) === 0
                  ? 'Permanent'
                  : `${style.titleDuration}s`}
              </span>
            </Field>
          </>
        )}
      </Section>

      <Section title='Transitions' icon='🎞️'>
        <Field label='Enable'>
          <Toggle
            checked={style.transitionEnabled ?? false}
            onChange={(v) => set('transitionEnabled', v)}
          />
        </Field>
        {style.transitionEnabled && (
          <>
            <Field label='Preset'>
              <select
                className='csp-select'
                value={transitionPresetValue}
                onChange={(e) => set('transitionPreset', e.target.value)}
              >
                <option value='crossfade'>Crossfade</option>
                <option value='dip-black'>Dip to Black</option>
                <option value='dip-white'>Dip to White</option>
                <option value='push-left'>Push Left</option>
                <option value='push-right'>Push Right</option>
                <option value='zoom'>Zoom Cross</option>
                <option value='glitch-cut'>Glitch Cut</option>
              </select>
            </Field>
            <Field
              label={`Duration (${(style.transitionDuration ?? 0.6).toFixed(2)}s)`}
            >
              <input
                className='csp-range'
                type='range'
                min={0.2}
                max={2}
                step={0.05}
                value={style.transitionDuration ?? 0.6}
                onChange={(e) => set('transitionDuration', +e.target.value)}
              />
              <span className='csp-range-val'>
                {(style.transitionDuration ?? 0.6).toFixed(2)}s
              </span>
            </Field>
            <Field label='Strength'>
              <select
                className='csp-select'
                value={style.transitionStrength || 'medium'}
                onChange={(e) => set('transitionStrength', e.target.value)}
              >
                <option value='low'>Low</option>
                <option value='medium'>Medium</option>
                <option value='high'>High</option>
              </select>
            </Field>
            <Field label='Apply On'>
              <select
                className='csp-select'
                value={transitionOnValue}
                onChange={(e) => set('transitionOn', e.target.value)}
              >
                <option value='start'>Video Start</option>
                <option value='section'>Every X Seconds</option>
                <option value='phrase'>Auto (Song Pace)</option>
                <option value='manual-marker'>
                  Manual Marker (Single Trigger)
                </option>
              </select>
            </Field>
            {transitionOnValue === 'section' && (
              <Field
                label={`Repeat Every (${(style.transitionSectionInterval ?? 8).toFixed(1)}s)`}
              >
                <input
                  className='csp-range'
                  type='range'
                  min={2}
                  max={20}
                  step={0.5}
                  value={style.transitionSectionInterval ?? 8}
                  onChange={(e) =>
                    set('transitionSectionInterval', +e.target.value)
                  }
                />
                <span className='csp-range-val'>
                  {(style.transitionSectionInterval ?? 8).toFixed(1)}s
                </span>
              </Field>
            )}
            {transitionOnValue === 'phrase' && (
              <>
                <p className='csp-field__hint'>
                  Auto cadence:{' '}
                  {Number(autoTransitionIntervalSeconds ?? 8).toFixed(1)}s{' '}
                  between transitions (derived from MIDI density).
                </p>
                <p className='csp-field__hint'>
                  Reason:{' '}
                  {autoTransitionReason || 'Estimated from song complexity'}.
                </p>
              </>
            )}
            <p className='csp-field__hint'>
              Use Video Start for a cinematic open, or Every X Seconds to add
              recurring transitions throughout the song. Auto derives a repeat
              cadence from MIDI note density.
            </p>
          </>
        )}
      </Section>

      <Section title='Beat Sync' icon='🥁'>
        <Field label='Enable'>
          <Toggle
            checked={style.beatSyncEnabled ?? false}
            onChange={(v) => set('beatSyncEnabled', v)}
          />
        </Field>
        {style.beatSyncEnabled && (
          <>
            <Field label='Sensitivity'>
              <select
                className='csp-select'
                value={style.beatSyncSensitivity || 'medium'}
                onChange={(e) => set('beatSyncSensitivity', e.target.value)}
              >
                <option value='low'>Low</option>
                <option value='medium'>Medium</option>
                <option value='high'>High</option>
              </select>
            </Field>
            <Field label='Pulse Mode'>
              <select
                className='csp-select'
                value={style.beatPulseMode || 'scale'}
                onChange={(e) => set('beatPulseMode', e.target.value)}
              >
                <option value='scale'>Scale</option>
                <option value='glow'>Glow</option>
                <option value='shake-lite'>Shake Lite</option>
              </select>
            </Field>
            <div className='csp-field'>
              <span className='csp-field__label'>Targets</span>
              <div className='csp-field__control csp-checkbox-row'>
                {[
                  ['title', 'Title'],
                  ['tagline', 'Tagline'],
                  ['track-cells', 'Track Cells'],
                  ['overlays', 'Overlays'],
                ].map(([value, label]) => {
                  const selected = Array.isArray(style.beatSyncTargets)
                    ? style.beatSyncTargets
                    : [];
                  const checked = selected.includes(value);
                  return (
                    <label key={value} className='csp-checkbox'>
                      <input
                        type='checkbox'
                        checked={checked}
                        onChange={(e) => {
                          const next = e.target.checked
                            ? [...new Set([...selected, value])]
                            : selected.filter((v) => v !== value);
                          set('beatSyncTargets', next);
                        }}
                      />
                      <span>{label}</span>
                    </label>
                  );
                })}
              </div>
            </div>
            <p className='csp-field__hint'>
              Beat Sync now exports subtle pulses for title, tagline,
              track-cells, and overlays.
            </p>
          </>
        )}
      </Section>

      <Section title='Ending Effects' icon='🏁'>
        <Field label='Enable'>
          <Toggle
            checked={style.outroEffectEnabled ?? false}
            onChange={(v) => set('outroEffectEnabled', v)}
          />
        </Field>
        {style.outroEffectEnabled && (
          <>
            <Field label='Preset'>
              <select
                className='csp-select'
                value={style.outroEffectPreset || 'fade-black'}
                onChange={(e) => set('outroEffectPreset', e.target.value)}
              >
                <option value='fade-black'>Fade to Black</option>
                <option value='fade-white'>Fade to White</option>
                <option value='glitch-out'>Glitch Out</option>
                <option value='zoom-out'>Zoom Out</option>
              </select>
            </Field>
            <Field
              label={`Duration (${(style.outroEffectDuration ?? 1.2).toFixed(1)}s)`}
            >
              <input
                className='csp-range'
                type='range'
                min={0.4}
                max={4}
                step={0.1}
                value={style.outroEffectDuration ?? 1.2}
                onChange={(e) => set('outroEffectDuration', +e.target.value)}
              />
              <span className='csp-range-val'>
                {(style.outroEffectDuration ?? 1.2).toFixed(1)}s
              </span>
            </Field>
            <Field label='Strength'>
              <select
                className='csp-select'
                value={style.outroEffectStrength || 'medium'}
                onChange={(e) => set('outroEffectStrength', e.target.value)}
              >
                <option value='low'>Low</option>
                <option value='medium'>Medium</option>
                <option value='high'>High</option>
              </select>
            </Field>
            <p className='csp-field__hint'>
              Ending effects are applied in the final seconds of the preview and
              exported video.
            </p>
          </>
        )}
      </Section>

      {/* Tagline */}
      <Section title='Tagline / Info Bar' icon='💬'>
        <Field label='Enable'>
          <Toggle
            checked={style.taglineEnabled}
            onChange={(v) => set('taglineEnabled', v)}
          />
        </Field>
        {style.taglineEnabled && (
          <>
            <Field label='Tagline'>
              <input
                className='csp-input'
                type='text'
                value={style.taglineText}
                onChange={(e) => set('taglineText', e.target.value)}
                placeholder='Persistent lower-third info bar text…'
                maxLength={120}
              />
            </Field>
            <Field label='Font Size'>
              <input
                className='csp-range'
                type='range'
                min={14}
                max={48}
                value={style.taglineFontSize}
                onChange={(e) => set('taglineFontSize', +e.target.value)}
              />
              <span className='csp-range-val'>{style.taglineFontSize}px</span>
            </Field>
            <Field label='Font'>
              <FontSelect
                value={style.taglineFont}
                onChange={(v) => set('taglineFont', v)}
              />
            </Field>
            <Field label='Color'>
              <input
                type='color'
                value={style.taglineColor}
                onChange={(e) => set('taglineColor', e.target.value)}
              />
            </Field>
            <Field label='Background'>
              <Toggle
                checked={style.taglineBackgroundEnabled ?? false}
                onChange={(v) => set('taglineBackgroundEnabled', v)}
              />
            </Field>
            {style.taglineBackgroundEnabled && (
              <>
                <Field label='Bar Color'>
                  <input
                    type='color'
                    value={style.taglineBackgroundColor || '#0c1220'}
                    onChange={(e) =>
                      set('taglineBackgroundColor', e.target.value)
                    }
                  />
                  <span className='csp-color-hex'>
                    {style.taglineBackgroundColor || '#0c1220'}
                  </span>
                </Field>
                <Field label='Bar Opacity'>
                  <input
                    className='csp-range'
                    type='range'
                    min={0.1}
                    max={1}
                    step={0.05}
                    value={style.taglineBackgroundOpacity ?? 0.72}
                    onChange={(e) =>
                      set('taglineBackgroundOpacity', +e.target.value)
                    }
                  />
                  <span className='csp-range-val'>
                    {Math.round((style.taglineBackgroundOpacity ?? 0.72) * 100)}
                    %
                  </span>
                </Field>
                <Field label='Accent Line'>
                  <input
                    type='color'
                    value={style.taglineAccentColor || '#ff4db8'}
                    onChange={(e) => set('taglineAccentColor', e.target.value)}
                  />
                  <span className='csp-color-hex'>
                    {style.taglineAccentColor || '#ff4db8'}
                  </span>
                </Field>
                <Field label='Shape'>
                  <select
                    className='csp-select'
                    value={style.taglineShape || 'rounded'}
                    onChange={(e) => set('taglineShape', e.target.value)}
                  >
                    <option value='rounded'>Rounded</option>
                    <option value='pill'>Pill</option>
                    <option value='square'>Square</option>
                    <option value='outline'>Outline</option>
                    <option value='accent-left'>Accent Left</option>
                  </select>
                </Field>
              </>
            )}
            <Field label='Width'>
              <input
                className='csp-range'
                type='range'
                min={20}
                max={100}
                step={1}
                value={style.taglineWidth || 72}
                onChange={(e) => set('taglineWidth', +e.target.value)}
              />
              <span className='csp-range-val'>{style.taglineWidth || 72}%</span>
            </Field>
            <Field label='Vertical Offset'>
              <input
                className='csp-range'
                type='range'
                min={-120}
                max={120}
                step={1}
                value={style.taglineVerticalOffset ?? 0}
                onChange={(e) => set('taglineVerticalOffset', +e.target.value)}
              />
              <span className='csp-range-val'>
                {(style.taglineVerticalOffset ?? 0) > 0
                  ? `+${style.taglineVerticalOffset}px up`
                  : (style.taglineVerticalOffset ?? 0) < 0
                    ? `${Math.abs(style.taglineVerticalOffset)}px down`
                    : '0px'}
              </span>
            </Field>
            <Field label='Position'>
              <select
                className='csp-select'
                value={style.taglinePosition || 'bottom-center'}
                onChange={(e) => set('taglinePosition', e.target.value)}
              >
                <option value='bottom-left'>Bottom Left</option>
                <option value='bottom-center'>Bottom Center</option>
                <option value='bottom-right'>Bottom Right</option>
              </select>
            </Field>
            <Field label='Alignment'>
              <select
                className='csp-select'
                value={style.taglineAlignment || 'center'}
                onChange={(e) => set('taglineAlignment', e.target.value)}
              >
                <option value='left'>Left</option>
                <option value='center'>Center</option>
                <option value='right'>Right</option>
              </select>
            </Field>
            <Field label='Text Shadow'>
              <Toggle
                checked={style.taglineShadowEnabled ?? true}
                onChange={(v) => set('taglineShadowEnabled', v)}
              />
            </Field>
            {style.taglineShadowEnabled && (
              <>
                <Field label='Shadow Size'>
                  <input
                    className='csp-range'
                    type='range'
                    min={0}
                    max={6}
                    step={0.5}
                    value={style.taglineShadowSize ?? 2}
                    onChange={(e) => set('taglineShadowSize', +e.target.value)}
                  />
                  <span className='csp-range-val'>
                    {style.taglineShadowSize ?? 2}px
                  </span>
                </Field>
                <Field label='Shadow Color'>
                  <input
                    type='color'
                    value={style.taglineShadowColor || '#000000'}
                    onChange={(e) => set('taglineShadowColor', e.target.value)}
                  />
                  <span className='csp-color-hex'>
                    {style.taglineShadowColor || '#000000'}
                  </span>
                </Field>
              </>
            )}
            <Field label='Fade In Duration'>
              <input
                className='csp-range'
                type='range'
                min={0}
                max={3}
                step={0.1}
                value={style.taglineFadeInDuration ?? 0.5}
                onChange={(e) => set('taglineFadeInDuration', +e.target.value)}
              />
              <span className='csp-range-val'>
                {(style.taglineFadeInDuration ?? 0.5).toFixed(1)}s
              </span>
            </Field>
            <Field label='Fade Out Duration'>
              <input
                className='csp-range'
                type='range'
                min={0}
                max={3}
                step={0.1}
                value={style.taglineFadeOutDuration ?? 0.5}
                onChange={(e) => set('taglineFadeOutDuration', +e.target.value)}
              />
              <span className='csp-range-val'>
                {(style.taglineFadeOutDuration ?? 0.5).toFixed(1)}s
              </span>
            </Field>
          </>
        )}
      </Section>

      {/* Title Effects */}
      <Section title='Title Effects' icon='✨'>
        {style.titleEnabled && (
          <>
            <Field label='Glow Effect'>
              <Toggle
                checked={style.titleGlowEnabled ?? false}
                onChange={(v) => set('titleGlowEnabled', v)}
              />
            </Field>
            {style.titleGlowEnabled && (
              <>
                <Field label='Glow Size'>
                  <input
                    className='csp-range'
                    type='range'
                    min={2}
                    max={20}
                    step={1}
                    value={style.titleGlowSize ?? 8}
                    onChange={(e) => set('titleGlowSize', +e.target.value)}
                  />
                  <span className='csp-range-val'>
                    {style.titleGlowSize ?? 8}px
                  </span>
                </Field>
                <Field label='Glow Color'>
                  <input
                    type='color'
                    value={style.titleGlowColor || '#ffffff'}
                    onChange={(e) => set('titleGlowColor', e.target.value)}
                  />
                  <span className='csp-color-hex'>
                    {style.titleGlowColor || '#ffffff'}
                  </span>
                </Field>
              </>
            )}
            <Field label='Text Shadow'>
              <Toggle
                checked={style.titleShadowEnabled ?? true}
                onChange={(v) => set('titleShadowEnabled', v)}
              />
            </Field>
            {style.titleShadowEnabled && (
              <>
                <Field label='Shadow Size'>
                  <input
                    className='csp-range'
                    type='range'
                    min={0}
                    max={8}
                    step={0.5}
                    value={style.titleShadowSize ?? 2}
                    onChange={(e) => set('titleShadowSize', +e.target.value)}
                  />
                  <span className='csp-range-val'>
                    {style.titleShadowSize ?? 2}px
                  </span>
                </Field>
                <Field label='Shadow Color'>
                  <input
                    type='color'
                    value={style.titleShadowColor || '#000000'}
                    onChange={(e) => set('titleShadowColor', e.target.value)}
                  />
                  <span className='csp-color-hex'>
                    {style.titleShadowColor || '#000000'}
                  </span>
                </Field>
              </>
            )}
          </>
        )}
        {!style.titleEnabled && (
          <p style={{ color: '#999', fontSize: '0.9rem', margin: 0 }}>
            Enable Title above to use effects.
          </p>
        )}
      </Section>

      <Section title='Watermark' icon='🔖'>
        <Field label='Enable'>
          <Toggle
            checked={style.watermarkEnabled}
            onChange={(v) => set('watermarkEnabled', v)}
          />
        </Field>
        {style.watermarkEnabled && (
          <>
            <Field label='Text'>
              <input
                className='csp-input'
                type='text'
                value={style.watermarkText}
                onChange={(e) => set('watermarkText', e.target.value)}
                placeholder='@yourhandle'
                maxLength={60}
              />
            </Field>
            <Field label='Position'>
              <select
                className='csp-select'
                value={style.watermarkPosition}
                onChange={(e) => set('watermarkPosition', e.target.value)}
              >
                <option value='bottom-right'>Bottom Right</option>
                <option value='bottom-left'>Bottom Left</option>
                <option value='top-right'>Top Right</option>
                <option value='top-left'>Top Left</option>
              </select>
            </Field>
            <Field label='Opacity'>
              <input
                className='csp-range'
                type='range'
                min={0.1}
                max={1}
                step={0.05}
                value={style.watermarkOpacity}
                onChange={(e) => set('watermarkOpacity', +e.target.value)}
              />
              <span className='csp-range-val'>
                {Math.round(style.watermarkOpacity * 100)}%
              </span>
            </Field>
            <Field label='Font Size'>
              <input
                className='csp-range'
                type='range'
                min={10}
                max={36}
                value={style.watermarkFontSize}
                onChange={(e) => set('watermarkFontSize', +e.target.value)}
              />
              <span className='csp-range-val'>{style.watermarkFontSize}px</span>
            </Field>
            <Field label='Font'>
              <FontSelect
                value={style.watermarkFont}
                onChange={(v) => set('watermarkFont', v)}
              />
            </Field>
            <Field label='Color'>
              <input
                type='color'
                value={style.watermarkColor}
                onChange={(e) => set('watermarkColor', e.target.value)}
              />
            </Field>
          </>
        )}
      </Section>

      {/* Waveform */}
      <Section title='Waveform Bar' icon='🌊'>
        <Field label='Enable'>
          <Toggle
            checked={style.waveformEnabled}
            onChange={(v) => set('waveformEnabled', v)}
          />
        </Field>
        {style.waveformEnabled && (
          <>
            <Field label='Color'>
              <input
                type='color'
                value={style.waveformColor}
                onChange={(e) => set('waveformColor', e.target.value)}
              />
            </Field>
            <Field label='Height'>
              <input
                className='csp-range'
                type='range'
                min={30}
                max={120}
                value={style.waveformHeight}
                onChange={(e) => set('waveformHeight', +e.target.value)}
              />
              <span className='csp-range-val'>{style.waveformHeight}px</span>
            </Field>
          </>
        )}
      </Section>

      {/* Visual Effects */}
      <Section title='Visual Effects' icon='✨'>
        <Field label='Vignette'>
          <Toggle
            checked={style.vignetteEnabled}
            onChange={(v) => set('vignetteEnabled', v)}
          />
        </Field>
        {style.vignetteEnabled && (
          <Field label='Strength'>
            <input
              className='csp-range'
              type='range'
              min={0.1}
              max={1}
              step={0.05}
              value={style.vignetteStrength}
              onChange={(e) => set('vignetteStrength', +e.target.value)}
            />
            <span className='csp-range-val'>
              {Math.round(style.vignetteStrength * 100)}%
            </span>
          </Field>
        )}
        <Field label='Glitch / VHS'>
          <Toggle
            checked={style.glitchEnabled}
            onChange={(v) => set('glitchEnabled', v)}
          />
        </Field>
        {style.glitchEnabled && (
          <Field label='Intensity'>
            <select
              className='csp-select'
              value={style.glitchIntensity}
              onChange={(e) => set('glitchIntensity', e.target.value)}
            >
              <option value='subtle'>Subtle</option>
              <option value='medium'>Medium</option>
              <option value='heavy'>Heavy</option>
            </select>
          </Field>
        )}
      </Section>
    </div>
  );
};

CompositionStylePanel.propTypes = {
  style: PropTypes.object.isRequired,
  onChange: PropTypes.func.isRequired,
  autoTransitionIntervalSeconds: PropTypes.number,
  autoTransitionReason: PropTypes.string,
};

export default CompositionStylePanel;
