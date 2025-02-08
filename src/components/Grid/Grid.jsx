import { useState, useMemo, useEffect } from 'react';
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

const Grid = ({ midiData, onArrangementChange }) => {
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

  // Heat map calculations
  const getHeatColor = (intensity) => {
    const hue = (1 - intensity) * 240;
    return `hsl(${hue}, 70%, 50%)`;
  };

  const getHeatIntensity = (count) => {
    const maxCount = Math.max(...items.map((item) => item.count));
    return count / maxCount;
  };
  console.log('columnCount', columnCount);
  console.log('calculateOptimalColumns', calculateOptimalColumns);
  console.log('items', items);
  console.log('row', Math.ceil(items.length / columnCount));
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
            {items.map((item) => (
              <SortableItem
                key={item.id}
                id={item.id}
                item={item}
                getHeatColor={
                  item.isEmpty
                    ? 'transparent'
                    : getHeatColor(getHeatIntensity(item.count))
                }
                isEmpty={item.isEmpty}
              />
            ))}
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
};

export default Grid;
