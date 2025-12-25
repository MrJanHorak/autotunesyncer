import PropTypes from 'prop-types';
import './Mixer.css';

const Mixer = ({ instruments, volumes, onVolumeChange }) => {
  return (
    <div className="mixer-container">
      <h3>Track Mixer</h3>
      <div className="mixer-tracks">
        {instruments.map((inst, index) => {
          const key = inst.isDrum 
            ? `drum_${inst.group.toLowerCase().replace(/\s+/g, '_')}`
            : inst.name.toLowerCase().replace(/\s+/g, '_');
            
          return (
            <div key={`${key}-${index}`} className="mixer-track">
              <label>{inst.isDrum ? inst.group : inst.name}</label>
              <input
                type="range"
                min="-60"
                max="6"
                step="1"
                value={volumes[key] || 0}
                onChange={(e) => onVolumeChange(key, parseFloat(e.target.value))}
              />
              <span className="volume-label">{volumes[key] || 0} dB</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

Mixer.propTypes = {
  instruments: PropTypes.array.isRequired,
  volumes: PropTypes.object.isRequired,
  onVolumeChange: PropTypes.func.isRequired
};

export default Mixer;
