/* eslint-disable react/prop-types */
import './ProgressBar.css';

const ProgressBar = ({ current, total }) => {
  const percentage = (current / total) * 100;

  return (
    <div className='progress-container'>
      <div className='progress-info'>
        <p>Recording Progress: {current} / {total}</p>
        <p>{Math.round(percentage)}%</p>
      </div>
      <div className='progress-bar-container'>
        <div 
          className='progress-bar'
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};

export default ProgressBar;