/* eslint-disable react/prop-types */
import './InstrumentList.css';

const InstrumentList = ({ instruments, onInstrumentSelect }) => {
  return (
    <div className={'instrument-list-container'}>
      <h3>Instruments found in Midi-File</h3>
      <ul className="instrument-list">
        {instruments.map((instrument, index) => (
          <li className={'instrument-list-item'}key={index} onClick={() => onInstrumentSelect(instrument)}>
            {instrument.family} - {instrument.name} - Midi Channel: {instrument.number}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default InstrumentList;
