import './styles.css';

const LoadingSpinner = () => (
  <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', zIndex: 20 }}>
    <div className="spinner"></div>
  </div>
);

export default LoadingSpinner;
