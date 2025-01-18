/* eslint-disable react/prop-types */
const InstrumentList = ({ instruments, onInstrumentSelect }) => {
  return (
    <div>
      <h2>Instruments</h2>
      <ul>
        {instruments.map((instrument, index) => (
          <li key={index} onClick={() => onInstrumentSelect(instrument)}>
            {instrument.family} - {instrument.name} (Number: {instrument.number})
          </li>
        ))}
      </ul>
    </div>
  );
};

export default InstrumentList;
