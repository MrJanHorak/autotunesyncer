/* eslint-disable react/prop-types */
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

export const SortableItem = ({ id, item, getHeatColor, isEmpty }) => {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
  } = useSortable({ id });

  const style = {
    transform: transform ? CSS.Transform.toString(transform) : '',
    transition,
    backgroundColor: isEmpty ? 'transparent' : getHeatColor,
    border: '1px solid #fff',
    borderRadius: '0.375rem',
    aspectRatio: '16/9',
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
        <div className="cell-content">
          <span className="cell-name">{item.name}</span>
          <span className="cell-count">{item.count} notes</span>
        </div>
      )}
    </div>
  );
};