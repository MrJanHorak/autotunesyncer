/* eslint-disable no-unused-vars */
import { useState, useMemo } from 'react';
import PropTypes from 'prop-types';
import GridLayout from 'react-grid-layout';
import { isDrumTrack, getDrumName } from '../../js/drumUtils';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

const Grid = ({ midiData }) => {
  const initialGridData = useMemo(() => {
    const trackData = [];
    const drumData = new Map();
    
    midiData.tracks.forEach((track, index) => {
      if (!track.notes?.length) return;

      if (isDrumTrack(track)) {
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
        trackData.push({
          name: track.instrument.name,
          count: track.notes.length,
        });
      }
    });

    return [...trackData, ...Array.from(drumData.values())];
  }, [midiData]);

  const [items, setItems] = useState(initialGridData);
  const [layout, setLayout] = useState(
    items.map((_, i) => ({
      i: i.toString(),
      x: i % 4,
      y: Math.floor(i / 4),
      w: 1,
      h: 1
    }))
  );

  const getHeatColor = (intensity) => {
    const hue = (1 - intensity) * 240;
    return `hsl(${hue}, 70%, 50%)`;
  };

  const getHeatIntensity = (count) => {
    const maxCount = Math.max(...items.map(item => item.count));
    return count / maxCount;
  };

  const handleLayoutChange = (newLayout) => {
    setLayout(newLayout);
  };

  return (
    <div className="grid-container">
      <GridLayout
        className="grid"
        layout={layout}
        cols={4}
        rowHeight={Math.floor(window.innerHeight / 6)}
        width={window.innerWidth * 0.9}
        onLayoutChange={handleLayoutChange}
        isDraggable={true}
        isResizable={false}
        margin={[8, 8]}
      >
        {items.map((item, index) => (
          <div
            key={index.toString()}
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
      </GridLayout>
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