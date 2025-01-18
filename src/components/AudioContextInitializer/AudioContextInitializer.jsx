/* eslint-disable react/prop-types */
import * as Tone from 'tone';
// import PropTypes from 'prop-types'; // Add this import

const AudioContextInitializer = ({ audioContextStarted, onInitialize }) => {
  const handleClick = async () => {
    await Tone.start();
    await onInitialize();
  };

  return !audioContextStarted ? (
    <button 
      onClick={handleClick}
      className="bg-yellow-100 p-4 rounded mb-4"
    >
      Click to initialize audio system
    </button>
  ) : null;
};

// // Add PropTypes validation
// AudioContextInitializer.propTypes = {
//   audioContextStarted: PropTypes.bool.isRequired,
//   onInitialize: PropTypes.func.isRequired
// };

export default AudioContextInitializer;