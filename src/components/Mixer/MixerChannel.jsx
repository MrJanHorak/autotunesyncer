import { useEffect, useMemo, useState } from 'react';
import PropTypes from 'prop-types';
import './MixerChannel.css';

const getDisplayValue = (value) => {
  const rounded = Math.round((value || 0) * 10) / 10;
  if (rounded === 0) return '0 dB';
  return `${rounded > 0 ? '+' : ''}${rounded} dB`;
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const toDbFs = (value) => {
  if (!Number.isFinite(value)) return -Infinity;
  // Backward compatibility in case any meter still emits linear 0..1.
  if (value >= 0 && value <= 1) {
    if (value <= 0.00001) return -Infinity;
    return 20 * Math.log10(value);
  }
  return value;
};

const meterPercentFromDb = (db) => {
  if (!Number.isFinite(db)) return 0;
  // Console-like scaling: treat -54..-6 dBFS as the main travel band.
  const floor = -54;
  const ceiling = -6;
  const normalized = clamp((db - floor) / (ceiling - floor), 0, 1);
  // Expand lower dynamics so normal program level feels alive.
  return Math.pow(normalized, 0.55) * 100;
};

const MixerChannel = ({
  name,
  volume,
  levelDb,
  isMuted,
  isSolo,
  onVolumeChange,
  onMute,
  onSolo,
  variant = 'compact',
}) => {
  const liveDb = useMemo(() => toDbFs(levelDb), [levelDb]);
  const hasSignal = !isMuted && liveDb > -72;
  const meterPct = hasSignal ? Math.max(3, meterPercentFromDb(liveDb)) : 0;
  const volumePct = Math.max(0, Math.min(100, ((volume + 60) / 70) * 100));
  const [peakPct, setPeakPct] = useState(0);

  useEffect(() => {
    if (meterPct > peakPct) {
      setPeakPct(meterPct);
      return;
    }

    const decay = window.setTimeout(() => {
      setPeakPct((prev) => Math.max(meterPct, prev - 2.2));
    }, 50);

    return () => window.clearTimeout(decay);
  }, [meterPct, peakPct]);

  const liveDbLabel = Number.isFinite(liveDb)
    ? `${liveDb.toFixed(1)} dBFS`
    : '-inf dBFS';

  return (
    <div
      className={[
        'mixer-channel',
        `mixer-channel--${variant}`,
        isMuted ? 'is-muted' : '',
        isSolo ? 'is-solo' : '',
      ]
        .filter(Boolean)
        .join(' ')}
    >
      <div className='mixer-channel__nameWrap'>
        <span className='mixer-channel__name' title={name}>
          {name}
        </span>
      </div>

      <div
        className={['mixer-channel__signal', hasSignal ? 'is-active' : '']
          .filter(Boolean)
          .join(' ')}
        title={hasSignal ? `Live level: ${liveDbLabel}` : 'No signal'}
      />

      <div className='mixer-channel__body'>
        <div className='mixer-channel__meter'>
          <div
            className='mixer-channel__peakMarker'
            style={{ bottom: `calc(${peakPct}% - 1px)` }}
          />
          <div
            className={['mixer-channel__meterFill', hasSignal ? 'is-live' : '']
              .filter(Boolean)
              .join(' ')}
            style={{
              height: `${meterPct}%`,
              opacity: hasSignal ? 1 : 0.2,
            }}
          />
        </div>

        <div className='mixer-channel__sliderWrap'>
          <div className='mixer-channel__sliderRail' />
          <div
            className='mixer-channel__sliderFill'
            style={{ height: `${volumePct}%` }}
          />
          <div
            className='mixer-channel__sliderThumb'
            style={{ bottom: `calc(${volumePct}% - 8px)` }}
          />
          <input
            type='range'
            min='-60'
            max='10'
            step='0.1'
            value={volume}
            onChange={(event) => onVolumeChange(parseFloat(event.target.value))}
            className='mixer-channel__slider'
            title={`Volume: ${getDisplayValue(volume)}`}
            aria-label={`${name} volume`}
          />
        </div>
      </div>

      <div
        className='mixer-channel__liveReadout'
        title={`Live: ${liveDbLabel}`}
      >
        {liveDbLabel}
      </div>

      <div className='mixer-channel__db'>{getDisplayValue(volume)}</div>

      <div className='mixer-channel__controls'>
        <button
          className={[
            'mixer-channel__button',
            'mixer-channel__button--mute',
            isMuted ? 'is-active' : '',
          ]
            .filter(Boolean)
            .join(' ')}
          onClick={onMute}
          title='Mute'
        >
          M
        </button>
        <button
          className={[
            'mixer-channel__button',
            'mixer-channel__button--solo',
            isSolo ? 'is-active' : '',
          ]
            .filter(Boolean)
            .join(' ')}
          onClick={onSolo}
          title='Solo'
        >
          S
        </button>
      </div>
    </div>
  );
};

MixerChannel.propTypes = {
  name: PropTypes.string.isRequired,
  volume: PropTypes.number.isRequired,
  levelDb: PropTypes.number,
  isMuted: PropTypes.bool,
  isSolo: PropTypes.bool,
  onVolumeChange: PropTypes.func.isRequired,
  onMute: PropTypes.func.isRequired,
  onSolo: PropTypes.func.isRequired,
  variant: PropTypes.oneOf(['compact', 'expanded']),
};

export default MixerChannel;
