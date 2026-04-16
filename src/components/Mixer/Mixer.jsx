import PropTypes from 'prop-types';
import './Mixer.css';

const SIGNAL_THRESHOLD_DB = -60; // below this = silence

const Mixer = ({ instruments, volumes, onVolumeChange, muteStates = {}, soloTrack = null, onMuteChange, onSoloChange, activeLevels = {} }) => {
  const handleMute = (key) => {
    if (onMuteChange) onMuteChange(key, !muteStates[key]);
  };

  const handleSolo = (key) => {
    if (onSoloChange) onSoloChange(key);
  };

  const getDisplayValue = (value) => {
    const db = value || 0;
    return db === 0 ? '0 dB' : `${db > 0 ? '+' : ''}${db} dB`;
  };

  return (
    <div className='mixer-container'>
      <div className='mixer-header'>
        <h3>Master Mixer</h3>
        <div className='mixer-info'>
          <span className='channel-count'>{instruments.length} channels</span>
          {instruments.length > 8 && (
            <span className='scroll-hint'>← Scroll →</span>
          )}
        </div>
      </div>

      <div className='mixer-wrapper'>
        <div className='mixer-channels'>
          {instruments.map((inst, index) => {
            const key = inst.isDrum
              ? `drum_${inst.group.toLowerCase().replace(/\s+/g, '_')}`
              : inst.name.toLowerCase().replace(/\s+/g, '_');

            const volume = volumes[key] || 0;
            const isMuted = muteStates[key];
            const isSolo = soloTrack === key;

            const rawLevel = activeLevels[key];
            const levelDb = typeof rawLevel === 'number' && isFinite(rawLevel) ? rawLevel : -Infinity;
            const isActive = !isMuted && levelDb > SIGNAL_THRESHOLD_DB;
            // Map -60..0 dBFS → 0..100 %
            const meterPct = isActive
              ? Math.max(0, Math.min(100, ((levelDb + 60) / 60) * 100))
              : 0;

            return (
              <div
                key={`${key}-${index}`}
                className={`channel-strip ${isMuted ? 'muted' : ''} ${isSolo ? 'solo' : ''}`}
              >
                {/* Signal LED */}
                <div className={`signal-led ${isActive ? 'active' : ''}`} title={isActive ? `${levelDb.toFixed(1)} dBFS` : 'No signal'} />

                {/* Channel Label */}
                <div className='channel-label'>
                  <span className='track-name'>
                    {inst.isDrum ? inst.group : inst.name}
                  </span>
                </div>

                {/* Fader Section */}
                <div className='fader-section'>
                  <div className='level-meter'>
                    <div
                      className={`meter-fill ${isActive ? 'live' : ''}`}
                      style={{
                        height: isActive
                          ? `${meterPct}%`
                          : `${((volume + 60) / 70) * 100}%`,
                        opacity: isActive ? 1 : 0.35,
                      }}
                    />
                  </div>

                  <input
                    type='range'
                    min='-60'
                    max='10'
                    step='0.1'
                    value={volume}
                    onChange={(e) =>
                      onVolumeChange(key, parseFloat(e.target.value))
                    }
                    className='fader'
                  />
                </div>

                {/* Value Display */}
                <div className='value-display'>{getDisplayValue(volume)}</div>

                {/* Control Buttons */}
                <div className='control-buttons'>
                  <button
                    className={`btn-mute ${isMuted ? 'active' : ''}`}
                    onClick={() => handleMute(key)}
                    title='Mute'
                  >
                    M
                  </button>
                  <button
                    className={`btn-solo ${isSolo ? 'active' : ''}`}
                    onClick={() => handleSolo(key)}
                    title='Solo'
                  >
                    S
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

Mixer.propTypes = {
  instruments: PropTypes.array.isRequired,
  volumes: PropTypes.object.isRequired,
  onVolumeChange: PropTypes.func.isRequired,
  activeLevels: PropTypes.object,
};

export default Mixer;
