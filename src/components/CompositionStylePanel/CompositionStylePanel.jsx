import { useState } from 'react';
import PropTypes from 'prop-types';
import { COLOR_THEMES, COLOR_GRADE_LABELS, DEFAULT_COMPOSITION_STYLE, FONT_OPTIONS } from '../../js/styleDefaults';
import './CompositionStylePanel.css';

const Section = ({ title, icon, children, defaultOpen = false }) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className='csp-section'>
      <button className='csp-section__header' onClick={() => setOpen((o) => !o)}>
        <span className='csp-section__icon'>{icon}</span>
        <span className='csp-section__title'>{title}</span>
        <span className='csp-section__chevron'>{open ? '▾' : '▸'}</span>
      </button>
      {open && <div className='csp-section__body'>{children}</div>}
    </div>
  );
};

Section.propTypes = {
  title: PropTypes.string.isRequired,
  icon: PropTypes.string.isRequired,
  children: PropTypes.node.isRequired,
  defaultOpen: PropTypes.bool,
};

const Field = ({ label, children }) => (
  <div className='csp-field'>
    <label className='csp-field__label'>{label}</label>
    <div className='csp-field__control'>{children}</div>
  </div>
);

Field.propTypes = { label: PropTypes.string.isRequired, children: PropTypes.node.isRequired };

const Toggle = ({ checked, onChange }) => (
  <label className='csp-toggle'>
    <input type='checkbox' checked={checked} onChange={(e) => onChange(e.target.checked)} />
    <span className='csp-toggle__slider' />
  </label>
);
Toggle.propTypes = { checked: PropTypes.bool.isRequired, onChange: PropTypes.func.isRequired };

const FontSelect = ({ value, onChange }) => (
  <select className='csp-select' value={value || 'default'} onChange={(e) => onChange(e.target.value)}>
    {FONT_OPTIONS.map((f) => (
      <option key={f.value} value={f.value}>{f.label}</option>
    ))}
  </select>
);
FontSelect.propTypes = { value: PropTypes.string, onChange: PropTypes.func.isRequired };


  const set = (key, val) => onChange({ ...style, [key]: val });

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

  return (
    <div className='csp'>
      <div className='csp__header'>
        <h3 className='csp__title'>🎨 Composition Style</h3>
        <button className='csp__reset' onClick={resetToDefaults} title='Reset all to defaults'>
          ↺ Reset
        </button>
      </div>

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
      <Section title='Song Title Overlay' icon='🎬'>
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
            <Field label='Fade In'>
              <Toggle checked={style.titleAnimated} onChange={(v) => set('titleAnimated', v)} />
            </Field>
          </>
        )}
      </Section>

      {/* Tagline */}
      <Section title='Tagline / Lower Third' icon='💬'>
        <Field label='Enable'>
          <Toggle checked={style.taglineEnabled} onChange={(v) => set('taglineEnabled', v)} />
        </Field>
        {style.taglineEnabled && (
          <>
            <Field label='Text'>
              <input className='csp-input' type='text' value={style.taglineText} onChange={(e) => set('taglineText', e.target.value)} placeholder='A short description…' maxLength={120} />
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

      {/* Intro Card */}
      <Section title='Intro Title Card' icon='🎞'>
        <Field label='Enable'>
          <Toggle checked={style.introCardEnabled} onChange={(v) => set('introCardEnabled', v)} />
        </Field>
        {style.introCardEnabled && (
          <>
            <Field label='Duration'>
              <input className='csp-range' type='range' min={1} max={8} step={0.5} value={style.introCardDuration} onChange={(e) => set('introCardDuration', +e.target.value)} />
              <span className='csp-range-val'>{style.introCardDuration}s</span>
            </Field>
            <Field label='Background'>
              <input type='color' value={style.introCardBg} onChange={(e) => set('introCardBg', e.target.value)} />
            </Field>
            <Field label='Title Text'>
              <input className='csp-input' type='text' value={style.introCardText} onChange={(e) => set('introCardText', e.target.value)} placeholder='Defaults to song title' maxLength={80} />
            </Field>
            <Field label='Subtitle'>
              <input className='csp-input' type='text' value={style.introCardSubtext} onChange={(e) => set('introCardSubtext', e.target.value)} placeholder='Artist / description…' maxLength={120} />
            </Field>
            <Field label='Text Color'>
              <input type='color' value={style.introCardTextColor} onChange={(e) => set('introCardTextColor', e.target.value)} />
            </Field>
            <Field label='Font'>
              <FontSelect value={style.introCardFont} onChange={(v) => set('introCardFont', v)} />
            </Field>
            <Field label='Animated'>
              <Toggle checked={style.introCardAnimated} onChange={(v) => set('introCardAnimated', v)} />
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
