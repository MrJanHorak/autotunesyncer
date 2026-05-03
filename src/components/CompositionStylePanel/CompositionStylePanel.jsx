import { useState, useId, cloneElement, isValidElement } from 'react';
import PropTypes from 'prop-types';
import { COLOR_THEMES, DEFAULT_COMPOSITION_STYLE, FONT_OPTIONS } from '../../js/styleDefaults';
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
      {open && <div id={bodyId} className='csp-section__body'>{children}</div>}
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
      <label className='csp-field__label' htmlFor={id}>{label}</label>
      <div className='csp-field__control'>{control}</div>
    </div>
  );
};

Field.propTypes = { label: PropTypes.string.isRequired, children: PropTypes.node.isRequired };

const Toggle = ({ id, checked, onChange }) => (
  <label className='csp-toggle'>
    <input id={id} type='checkbox' checked={checked} onChange={(e) => onChange(e.target.checked)} />
    <span className='csp-toggle__slider' />
  </label>
);
Toggle.propTypes = { id: PropTypes.string, checked: PropTypes.bool.isRequired, onChange: PropTypes.func.isRequired };

const FontSelect = ({ id, value, onChange }) => (
  <select id={id} className='csp-select' value={value || 'default'} onChange={(e) => onChange(e.target.value)}>
    {FONT_OPTIONS.map((f) => (
      <option key={f.value} value={f.value}>{f.label}</option>
    ))}
  </select>
);
FontSelect.propTypes = { id: PropTypes.string, value: PropTypes.string, onChange: PropTypes.func.isRequired };

const CompositionStylePanel = ({ style, onChange }) => {
  const set = (key, val) => onChange({ ...style, [key]: val });
  const { presets, savePreset, applyPreset, deletePreset } = useStylePresets();
  const [selectedPreset, setSelectedPreset] = useState('');

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

  return (
    <div className='csp'>
      <div className='csp__header'>
        <h3 className='csp__title'>🎨 Composition Style</h3>
        <button className='csp__reset' onClick={resetToDefaults} title='Reset all to defaults'>
          ↺ Reset
        </button>
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
              <option key={p.name} value={p.name}>{p.name}</option>
            ))}
          </select>
          <div className='csp-presets__actions'>
            <button
              className='csp-btn csp-btn--sm'
              onClick={handleApplyPreset}
              disabled={!selectedPreset}
              title='Apply selected preset'
            >Apply</button>
            <button
              className='csp-btn csp-btn--sm'
              onClick={handleSavePreset}
              title='Save current style as a new preset'
            >Save as…</button>
            <button
              className='csp-btn csp-btn--sm csp-btn--danger'
              onClick={handleDeletePreset}
              disabled={!selectedPreset}
              title='Delete selected preset'
            >Delete</button>
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
              style={{ background: COLOR_THEMES[t].backgroundColor, borderColor: COLOR_THEMES[t].clipDefaults.borderColor }}
              onClick={() => applyTheme(t)}
            >
              <span className='csp-theme-btn__dot' style={{ background: COLOR_THEMES[t].clipDefaults.borderColor }} />
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>
        <Field label='Canvas Background'>
          <input type='color' value={style.backgroundColor} onChange={(e) => set('backgroundColor', e.target.value)} />
          <span className='csp-color-hex'>{style.backgroundColor}</span>
        </Field>
      </Section>

      {/* Title */}
      <Section title='Title / Intro Card' icon='🎬'>
        <Field label='Enable'>
          <Toggle checked={style.titleEnabled} onChange={(v) => set('titleEnabled', v)} />
        </Field>
        {style.titleEnabled && (
          <>
            <Field label='Text'>
              <input className='csp-input' type='text' value={style.titleText} onChange={(e) => set('titleText', e.target.value)} placeholder='Your Song Title' maxLength={80} />
            </Field>
            <Field label='Position'>
              <select className='csp-select' value={style.titlePosition} onChange={(e) => set('titlePosition', e.target.value)}>
                <option value='top-center'>Top Center</option>
                <option value='bottom-center'>Bottom Center</option>
                <option value='center'>Center</option>
              </select>
            </Field>
            <Field label='Font Size'>
              <input className='csp-range' type='range' min={24} max={96} value={style.titleFontSize} onChange={(e) => set('titleFontSize', +e.target.value)} />
              <span className='csp-range-val'>{style.titleFontSize}px</span>
            </Field>
            <Field label='Font'>
              <FontSelect value={style.titleFont} onChange={(v) => set('titleFont', v)} />
            </Field>
            <Field label='Color'>
              <input type='color' value={style.titleColor} onChange={(e) => set('titleColor', e.target.value)} />
              <span className='csp-color-hex'>{style.titleColor}</span>
            </Field>
            <Field label='Subtitle'>
              <input className='csp-input' type='text' value={style.titleSubtitleText ?? ''} onChange={(e) => set('titleSubtitleText', e.target.value)} placeholder='Optional line below the title' maxLength={120} />
            </Field>
            {Boolean(style.titleSubtitleText?.trim()) && (
              <>
                <Field label='Subtitle Size'>
                  <input className='csp-range' type='range' min={14} max={40} value={style.titleSubtitleFontSize ?? 24} onChange={(e) => set('titleSubtitleFontSize', +e.target.value)} />
                  <span className='csp-range-val'>{style.titleSubtitleFontSize ?? 24}px</span>
                </Field>
                <Field label='Subtitle Color'>
                  <input type='color' value={style.titleSubtitleColor || '#d8d8e6'} onChange={(e) => set('titleSubtitleColor', e.target.value)} />
                  <span className='csp-color-hex'>{style.titleSubtitleColor || '#d8d8e6'}</span>
                </Field>
              </>
            )}
            <Field label='Background'>
              <Toggle checked={style.titleBackgroundEnabled ?? false} onChange={(v) => set('titleBackgroundEnabled', v)} />
            </Field>
            {style.titleBackgroundEnabled && (
              <>
                <Field label='Card Color'>
                  <input type='color' value={style.titleBackgroundColor || '#120b24'} onChange={(e) => set('titleBackgroundColor', e.target.value)} />
                  <span className='csp-color-hex'>{style.titleBackgroundColor || '#120b24'}</span>
                </Field>
                <Field label='Card Opacity'>
                  <input className='csp-range' type='range' min={0.1} max={1} step={0.05} value={style.titleBackgroundOpacity ?? 0.82} onChange={(e) => set('titleBackgroundOpacity', +e.target.value)} />
                  <span className='csp-range-val'>{Math.round((style.titleBackgroundOpacity ?? 0.82) * 100)}%</span>
                </Field>
              </>
            )}
            <Field label='Show at Start'>
              <Toggle checked={style.introCardEnabled} onChange={(v) => set('introCardEnabled', v)} />
            </Field>
            {style.introCardEnabled && (
              <Field label='Card Duration'>
                <input className='csp-range' type='range' min={1} max={8} step={0.5} value={style.introCardDuration} onChange={(e) => set('introCardDuration', +e.target.value)} />
                <span className='csp-range-val'>{style.introCardDuration}s</span>
              </Field>
            )}
            <Field label='Animate'>
              <Toggle checked={style.titleAnimated} onChange={(v) => set('titleAnimated', v)} />
            </Field>
            <Field label='Overlay Duration'>
              <input className='csp-range' type='range' min={0} max={15} step={0.5} value={style.titleDuration ?? 0} onChange={(e) => set('titleDuration', +e.target.value)} />
              <span className='csp-range-val'>{(style.titleDuration ?? 0) === 0 ? 'Permanent' : `${style.titleDuration}s`}</span>
            </Field>
          </>
        )}
      </Section>

      {/* Tagline */}
      <Section title='Tagline / Info Bar' icon='💬'>
        <Field label='Enable'>
          <Toggle checked={style.taglineEnabled} onChange={(v) => set('taglineEnabled', v)} />
        </Field>
        {style.taglineEnabled && (
          <>
            <Field label='Tagline'>
              <input className='csp-input' type='text' value={style.taglineText} onChange={(e) => set('taglineText', e.target.value)} placeholder='Persistent lower-third info bar text…' maxLength={120} />
            </Field>
            <Field label='Font Size'>
              <input className='csp-range' type='range' min={14} max={48} value={style.taglineFontSize} onChange={(e) => set('taglineFontSize', +e.target.value)} />
              <span className='csp-range-val'>{style.taglineFontSize}px</span>
            </Field>
            <Field label='Font'>
              <FontSelect value={style.taglineFont} onChange={(v) => set('taglineFont', v)} />
            </Field>
            <Field label='Color'>
              <input type='color' value={style.taglineColor} onChange={(e) => set('taglineColor', e.target.value)} />
            </Field>
            <Field label='Background'>
              <Toggle checked={style.taglineBackgroundEnabled ?? false} onChange={(v) => set('taglineBackgroundEnabled', v)} />
            </Field>
            {style.taglineBackgroundEnabled && (
              <>
                <Field label='Bar Color'>
                  <input type='color' value={style.taglineBackgroundColor || '#0c1220'} onChange={(e) => set('taglineBackgroundColor', e.target.value)} />
                  <span className='csp-color-hex'>{style.taglineBackgroundColor || '#0c1220'}</span>
                </Field>
                <Field label='Bar Opacity'>
                  <input className='csp-range' type='range' min={0.1} max={1} step={0.05} value={style.taglineBackgroundOpacity ?? 0.72} onChange={(e) => set('taglineBackgroundOpacity', +e.target.value)} />
                  <span className='csp-range-val'>{Math.round((style.taglineBackgroundOpacity ?? 0.72) * 100)}%</span>
                </Field>
                <Field label='Accent Line'>
                  <input type='color' value={style.taglineAccentColor || '#ff4db8'} onChange={(e) => set('taglineAccentColor', e.target.value)} />
                  <span className='csp-color-hex'>{style.taglineAccentColor || '#ff4db8'}</span>
                </Field>
              </>
            )}
          </>
        )}
      </Section>

      {/* Watermark */}
      <Section title='Watermark' icon='🔖'>
        <Field label='Enable'>
          <Toggle checked={style.watermarkEnabled} onChange={(v) => set('watermarkEnabled', v)} />
        </Field>
        {style.watermarkEnabled && (
          <>
            <Field label='Text'>
              <input className='csp-input' type='text' value={style.watermarkText} onChange={(e) => set('watermarkText', e.target.value)} placeholder='@yourhandle' maxLength={60} />
            </Field>
            <Field label='Position'>
              <select className='csp-select' value={style.watermarkPosition} onChange={(e) => set('watermarkPosition', e.target.value)}>
                <option value='bottom-right'>Bottom Right</option>
                <option value='bottom-left'>Bottom Left</option>
                <option value='top-right'>Top Right</option>
                <option value='top-left'>Top Left</option>
              </select>
            </Field>
            <Field label='Opacity'>
              <input className='csp-range' type='range' min={0.1} max={1} step={0.05} value={style.watermarkOpacity} onChange={(e) => set('watermarkOpacity', +e.target.value)} />
              <span className='csp-range-val'>{Math.round(style.watermarkOpacity * 100)}%</span>
            </Field>
            <Field label='Font Size'>
              <input className='csp-range' type='range' min={10} max={36} value={style.watermarkFontSize} onChange={(e) => set('watermarkFontSize', +e.target.value)} />
              <span className='csp-range-val'>{style.watermarkFontSize}px</span>
            </Field>
            <Field label='Font'>
              <FontSelect value={style.watermarkFont} onChange={(v) => set('watermarkFont', v)} />
            </Field>
            <Field label='Color'>
              <input type='color' value={style.watermarkColor} onChange={(e) => set('watermarkColor', e.target.value)} />
            </Field>
          </>
        )}
      </Section>

      {/* Waveform */}
      <Section title='Waveform Bar' icon='🌊'>
        <Field label='Enable'>
          <Toggle checked={style.waveformEnabled} onChange={(v) => set('waveformEnabled', v)} />
        </Field>
        {style.waveformEnabled && (
          <>
            <Field label='Color'>
              <input type='color' value={style.waveformColor} onChange={(e) => set('waveformColor', e.target.value)} />
            </Field>
            <Field label='Height'>
              <input className='csp-range' type='range' min={30} max={120} value={style.waveformHeight} onChange={(e) => set('waveformHeight', +e.target.value)} />
              <span className='csp-range-val'>{style.waveformHeight}px</span>
            </Field>
          </>
        )}
      </Section>

      {/* Visual Effects */}
      <Section title='Visual Effects' icon='✨'>
        <Field label='Vignette'>
          <Toggle checked={style.vignetteEnabled} onChange={(v) => set('vignetteEnabled', v)} />
        </Field>
        {style.vignetteEnabled && (
          <Field label='Strength'>
            <input className='csp-range' type='range' min={0.1} max={1} step={0.05} value={style.vignetteStrength} onChange={(e) => set('vignetteStrength', +e.target.value)} />
            <span className='csp-range-val'>{Math.round(style.vignetteStrength * 100)}%</span>
          </Field>
        )}
        <Field label='Glitch / VHS'>
          <Toggle checked={style.glitchEnabled} onChange={(v) => set('glitchEnabled', v)} />
        </Field>
        {style.glitchEnabled && (
          <Field label='Intensity'>
            <select className='csp-select' value={style.glitchIntensity} onChange={(e) => set('glitchIntensity', e.target.value)}>
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
};

export default CompositionStylePanel;
