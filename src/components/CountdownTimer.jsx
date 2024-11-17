/* eslint-disable react/prop-types */
import { useEffect, useState } from 'react';

const CountdownTimer = ({ duration, onComplete }) => {
  const [count, setCount] = useState(duration);

  useEffect(() => {
    if (count > 0) {
      const timer = setTimeout(() => setCount(count - 1), 1000);
      return () => clearTimeout(timer);
    } else {
      onComplete();
    }
  }, [count, onComplete]);

  return (
    <div className="countdown-overlay">
      <div className="countdown-number">{count}</div>
    </div>
  );
};

export default CountdownTimer;
