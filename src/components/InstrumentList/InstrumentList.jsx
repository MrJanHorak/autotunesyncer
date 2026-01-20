/* eslint-disable react/prop-types */
import { useState } from 'react';
import './InstrumentList.css';

const InstrumentList = ({ instruments, onInstrumentSelect }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className='instrument-list-container'>
      <div 
        className='instrument-list-header'
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <h3>Instruments found in Midi-File</h3>
        <span className={`arrow ${isExpanded ? 'expanded' : ''}`}>▼</span>
      </div>
      <div className={`instrument-list-content ${isExpanded ? 'expanded' : ''}`}>
        <ul className="instrument-list">
          {instruments.map((instrument, index) => (
            <li 
              className='instrument-list-item' 
              key={index} 
              onClick={() => onInstrumentSelect(instrument)}
            >
              <div className='instrument-name-line'>
                {instrument.isDrum ? (
                  <>🥁 {instrument.group}</>
                ) : (
                  <>{instrument.family} - {instrument.name}</>
                )}
              </div>
              <div className='instrument-channel-line'>
                Midi Channel: {instrument.number}
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default InstrumentList;