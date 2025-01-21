/* eslint-disable react/prop-types */
import './ToggleSwitch.css';
const ToggleSwitch = ({ checked, onChange, onText, offText }) => {
  console.log('ToggleSwitch render:', { checked, onText, offText });
  
  const handleChange = (e) => {
    console.log('Toggle clicked');
    onChange?.(e);
  };

  return (
    <div className='checkbox-wrapper'>
      <input
        className='toggle-switch skewed'
        type='checkbox'
        checked={checked}
        onChange={handleChange}
      />
      <label 
        className='toggle-btn'
        data-tg-on={onText}
        data-tg-off={offText}
      />
    </div>
  );
};

export default ToggleSwitch;