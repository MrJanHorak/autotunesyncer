/* eslint-disable react/prop-types */
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

export const SortableItem = ({
  id,
  item,
  getHeatColor,
  accentColor,
  isEmpty,
}) => {
  const { attributes, listeners, setNodeRef, transform, transition } =
    useSortable({ id });

  const style = {
    transform: transform ? CSS.Transform.toString(transform) : '',
    transition,
    background: isEmpty ? '#f3f4f6' : getHeatColor,
    borderRadius: '12px',
    aspectRatio: '16/9',
  };

  const cellContentStyle = {
    '--accent-color': accentColor,
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={`grid-cell ${isEmpty ? 'empty' : ''}`}
      {...attributes}
      {...listeners}
    >
      {!isEmpty && (
        <div className='cell-content' style={cellContentStyle}>
          <span className='cell-name'>{item.name}</span>
          <span className='cell-count'>{item.count} notes</span>
        </div>
      )}
    </div>
  );
};
