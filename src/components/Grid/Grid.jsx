import { useState, useMemo, useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
} from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  rectSortingStrategy,
} from '@dnd-kit/sortable';
import { SortableItem } from './SortableItems';
import { isDrumTrack, getDrumName } from '../../js/drumUtils';
import './Grid.css';

const Grid = ({ midiData, onArrangementChange, initialArrangement }) => {
  // 1. Process MIDI data first
  const processedData = useMemo(() => {
    const trackData = [];
    const drumData = new Map();

    midiData.tracks.forEach((track, trackIndex) => {
      if (!track.notes?.length) return;

      if (isDrumTrack(track)) {
        track.notes.forEach((note) => {
          const drumName = getDrumName(note.midi);
          const key = `drum_${drumName.toLowerCase().replace(/\s+/g, '_')}`;
          if (!drumData.has(key)) {
            drumData.set(key, {
              id: `drum-${key}`,
              name: drumName,
              count: 0,
            });
          }
          drumData.get(key).count++;
        });
      } else {
        trackData.push({
          id: `track-${trackIndex}`,
          name: track.instrument.name,
          count: track.notes.length,
        });
      }
    });

    return [...trackData, ...Array.from(drumData.values())];
  }, [midiData]);

  // 2. Calculate optimal columns based on processed data
  const calculateOptimalColumns = useMemo(() => {
    const itemCount = processedData.length;
    const optimalColumns = [];

    for (let cols = 1; cols <= Math.min(5, itemCount); cols++) {
      const rows = Math.ceil(itemCount / cols);
      const gridWidth = cols * 16;
      const gridHeight = rows * 9;
      const gridAspectRatio = gridWidth / gridHeight;

      const isViable =
        gridAspectRatio >= 1 &&
        gridAspectRatio <= 2 &&
        rows <= 3 &&
        rows * cols >= itemCount;

      if (isViable) {
        optimalColumns.push({
          cols,
          ratio: Math.abs(1.7777 - gridAspectRatio),
        });
      }
    }

    optimalColumns.sort((a, b) => a.ratio - b.ratio);
    return optimalColumns.length > 0
      ? optimalColumns.map((col) => col.cols)
      : [Math.ceil(Math.sqrt(itemCount))];
  }, [processedData.length]);

  // 3. Create initial grid data with empty spaces
  const initialGridData = useMemo(() => {
    const totalColumns = calculateOptimalColumns[0] || 4;
    const totalRows = Math.ceil(processedData.length / totalColumns);
    const totalSpaces = totalRows * totalColumns;

    const emptySpaces = Array.from(
      { length: totalSpaces - processedData.length },
      (_, index) => ({
        id: `empty-${index}`,
        name: '',
        count: 0,
        isEmpty: true,
      })
    );

    return [...processedData, ...emptySpaces];
  }, [processedData, calculateOptimalColumns]);

  // 4. Initialize state
  const [items, setItems] = useState(initialGridData);
  const [columnCount, setColumnCount] = useState(
    calculateOptimalColumns[0] || 4
  );
  const arrangementRestoredRef = useRef(false);

  // Restore saved drag order once (on first non-empty initialArrangement)
  useEffect(() => {
    if (arrangementRestoredRef.current) return;
    if (!initialArrangement || Object.keys(initialArrangement).length === 0) return;
    arrangementRestoredRef.current = true;
    setItems((current) =>
      [...current].sort((a, b) => {
        const idA = a.id.replace(/^(track-|drum-)/, '');
        const idB = b.id.replace(/^(track-|drum-)/, '');
        const posA = initialArrangement[idA]?.position ?? Infinity;
        const posB = initialArrangement[idB]?.position ?? Infinity;
        return posA - posB;
      })
    );
  }, [initialArrangement]);

  // DND setup
  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  // Handlers
  // const handleDragEnd = (event) => {
  //   const { active, over } = event;

  //   if (active.id !== over.id) {
  //     setItems((items) => {
  //       const oldIndex = items.findIndex((item) => item.id === active.id);
  //       const newIndex = items.findIndex((item) => item.id === over.id);

  //       if (oldIndex !== -1 && newIndex !== -1) {
  //         return arrayMove(items, oldIndex, newIndex);
  //       }
  //       return items;
  //     });
  //   }
  // };

  const handleDragEnd = (event) => {
    const { active, over } = event;
    if (active.id !== over.id) {
      setItems((items) => {
        const oldIndex = items.findIndex((item) => item.id === active.id);
        const newIndex = items.findIndex((item) => item.id === over.id);
        const newItems = arrayMove(items, oldIndex, newIndex);

        // Modify arrangement to match backend expectations
        const arrangement = newItems.reduce((acc, item, index) => {
          if (!item.isEmpty) {
            // Extract the actual identifier without the prefix
            const id = item.id.replace(/^(track-|drum-)/, '');
            acc[id] = {
              position: index,
              row: Math.floor(index / columnCount),
              column: index % columnCount,
              type: item.id.startsWith('drum-') ? 'drum' : 'track',
            };
          }
          return acc;
        }, {});
        onArrangementChange(arrangement);
        return newItems;
      });
    }
  };

  const handleColumnChange = (event) => {
    const newColumnCount = parseInt(event.target.value);
    setColumnCount(newColumnCount);
    document.documentElement.style.setProperty(
      '--column-count',
      newColumnCount
    );
  };

  // Heat map calculations with modern spectrum gradient - 10 tier system for maximum distinction
  const getHeatColor = (intensity) => {
    // Creates smooth gradient backgrounds through the full thermal spectrum
    // Blue (cold) → Cyan → Green → Yellow → Orange → Red (hot)
    // Maximum visual distinction across entire activity range
    if (intensity < 0.1) {
      // Extreme low (0-10%) - pale blue
      return 'linear-gradient(135deg, #bfdbfe 0%, #93c5fd 100%)';
    } else if (intensity < 0.2) {
      // Very minimal (10-20%) - sky blue
      return 'linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%)';
    } else if (intensity < 0.3) {
      // Minimal (20-30%) - bright blue
      return 'linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%)';
    } else if (intensity < 0.4) {
      // Very low (30-40%) - cyan
      return 'linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%)';
    } else if (intensity < 0.5) {
      // Low (40-50%) - cyan-green
      return 'linear-gradient(135deg, #14b8a6 0%, #10b981 100%)';
    } else if (intensity < 0.6) {
      // Medium-low (50-60%) - green
      return 'linear-gradient(135deg, #22c55e 0%, #84cc16 100%)';
    } else if (intensity < 0.7) {
      // Medium (60-70%) - yellow-green
      return 'linear-gradient(135deg, #84cc16 0%, #eab308 100%)';
    } else if (intensity < 0.8) {
      // Medium-high (70-80%) - yellow-orange
      return 'linear-gradient(135deg, #eab308 0%, #f59e0b 100%)';
    } else if (intensity < 0.9) {
      // High (80-90%) - orange
      return 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)';
    } else {
      // Extreme high (90-100%) - orange to red
      return 'linear-gradient(135deg, #ea580c 0%, #dc2626 100%)';
    }
  };

  const getHeatIntensity = (count) => {
    const maxCount = Math.max(...items.map((item) => item.count || 0));
    return maxCount > 0 ? count / maxCount : 0;
  };

  // Get accent color for text - always white for readability
  const getAccentColor = () => {
    return '#ffffff'; // Always white for best readability
  };
  useEffect(() => {
    if (calculateOptimalColumns.length > 0) {
      const optimalColumnCount = calculateOptimalColumns[0];
      setColumnCount(optimalColumnCount);
      document.documentElement.style.setProperty(
        '--column-count',
        optimalColumnCount
      );
    }
  }, [calculateOptimalColumns]);

  // Initialize arrangement on load
  useEffect(() => {
    if (items.length > 0) {
      const arrangement = items.reduce((acc, item, index) => {
        if (!item.isEmpty) {
          const id = item.id.replace(/^(track-|drum-)/, '');
          acc[id] = {
            position: index,
            row: Math.floor(index / columnCount),
            column: index % columnCount,
            type: item.id.startsWith('drum-') ? 'drum' : 'track',
          };
        }
        return acc;
      }, {});
      onArrangementChange(arrangement);
    }
  }, [items, columnCount, onArrangementChange]);

  return (
    <div className='grid-container'>
      <div className='grid-controls'>
        <label htmlFor='column-select'>Grid Columns: </label>
        <select
          id='column-select'
          value={columnCount}
          onChange={handleColumnChange}
          className='column-select'
        >
          {calculateOptimalColumns.map((cols) => (
            <option key={cols} value={cols}>
              {cols} {cols === 1 ? 'Column' : 'Columns'}
            </option>
          ))}
        </select>
      </div>

      <DndContext
        sensors={sensors}
        collisionDetection={closestCenter}
        onDragEnd={handleDragEnd}
      >
        <div
          className='grid'
          style={{
            display: 'grid',
            gridTemplateColumns: `repeat(${columnCount}, 1fr)`,
            gap: '8px',
            aspectRatio: '16/9',
            width: '100%',
            height: 'auto',
            maxHeight: '100%',
          }}
        >
          <SortableContext items={items} strategy={rectSortingStrategy}>
            {items.map((item) => {
              const intensity = getHeatIntensity(item.count);
              return (
                <SortableItem
                  key={item.id}
                  id={item.id}
                  item={item}
                  getHeatColor={
                    item.isEmpty
                      ? 'transparent'
                      : getHeatColor(intensity)
                  }
                  accentColor={
                    item.isEmpty
                      ? '#9ca3af'
                      : getAccentColor(intensity)
                  }
                  isEmpty={item.isEmpty}
                />
              );
            })}
          </SortableContext>
        </div>
      </DndContext>
    </div>
  );
};

Grid.propTypes = {
  midiData: PropTypes.shape({
    tracks: PropTypes.arrayOf(
      PropTypes.shape({
        notes: PropTypes.array,
        instrument: PropTypes.shape({
          name: PropTypes.string,
        }),
      })
    ),
  }).isRequired,
  onArrangementChange: PropTypes.func.isRequired,
  initialArrangement: PropTypes.object,
};

export default Grid;
