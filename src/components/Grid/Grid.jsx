/* eslint-disable no-unused-vars */
// src/components/Grid/Grid.jsx
import { useMemo } from 'react';
import PropTypes from 'prop-types';
import { isDrumTrack, getDrumName } from '../../js/drumUtils';
import './Grid.css';

const Grid = ({ midiData }) => {
  const gridData = useMemo(() => {
    const trackData = [];
    const drumData = new Map();
    
    // Process MIDI tracks
    midiData.tracks.forEach((track, index) => {
      if (!track.notes?.length) return;

      if (isDrumTrack(track)) {
        // Group drum notes by type
        track.notes.forEach(note => {
          const drumName = getDrumName(note.midi);
          const key = `drum_${drumName.toLowerCase().replace(/\s+/g, '_')}`;
          if (!drumData.has(key)) {
            drumData.set(key, {
              name: drumName,
              count: 0,
            });
          }
          drumData.get(key).count++;
        });
      } else {
        // Add melodic track
        trackData.push({
          name: track.instrument.name,
          count: track.notes.length,
        });
      }
    });

    return [...trackData, ...Array.from(drumData.values())];
  }, [midiData]);

  // Calculate heat intensity based on note count
  const getHeatIntensity = (count) => {
    const maxCount = Math.max(...gridData.map(item => item.count));
    return count / maxCount;
  };

  const getHeatColor = (intensity) => {
    // Create a gradient from blue (cold) to red (hot)
    const hue = (1 - intensity) * 240; // 240 is blue, 0 is red
    return `hsl(${hue}, 70%, 50%)`;
  };

  return (
    <div className="grid-container">
      <h2>Track Distribution</h2>
      <div className="grid">
        {gridData.map((item, index) => (
          <div 
            key={index}
            className="grid-cell"
            style={{
              backgroundColor: getHeatColor(getHeatIntensity(item.count))
            }}
          >
            <div className="cell-content">
              <span className="cell-name">{item.name}</span>
              <span className="cell-count">{item.count} notes</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

Grid.propTypes = {
  midiData: PropTypes.shape({
    tracks: PropTypes.arrayOf(PropTypes.shape({
      notes: PropTypes.array,
      instrument: PropTypes.shape({
        name: PropTypes.string
      })
    }))
  }).isRequired
};

export default Grid;