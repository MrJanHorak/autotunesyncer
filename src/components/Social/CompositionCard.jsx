import PropTypes from 'prop-types';

const API_BASE = 'http://localhost:3000';

function formatDate(dateStr) {
  const d = new Date(dateStr);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

const CompositionCard = ({ composition, onSelect, onProfileClick }) => {
  const thumbUrl = composition.thumbnail_path
    ? `${API_BASE}/published/${composition.thumbnail_path}`
    : null;

  return (
    <div className='comp-card' onClick={() => onSelect(composition.id)}>
      {thumbUrl ? (
        <img className='comp-card__thumb' src={thumbUrl} alt={composition.title} loading='lazy' />
      ) : (
        <div className='comp-card__thumb-placeholder'>🎬</div>
      )}
      <div className='comp-card__body'>
        <div className='comp-card__title' title={composition.title}>{composition.title}</div>
        <div className='comp-card__meta'>
          <button
            className='comp-card__author'
            onClick={(e) => { e.stopPropagation(); onProfileClick(composition.user_id); }}
          >
            @{composition.username}
          </button>
          <div className='comp-card__stats'>
            <span className='comp-card__stat' title='Likes'>
              {composition.liked_by_me ? '❤️' : '🤍'} {composition.like_count}
            </span>
            <span className='comp-card__stat' title='Comments'>
              💬 {composition.comment_count}
            </span>
          </div>
        </div>
        <div style={{ fontSize: '0.75rem', color: '#9ca3af', marginTop: '0.35rem' }}>
          {formatDate(composition.created_at)}
        </div>
      </div>
    </div>
  );
};

CompositionCard.propTypes = {
  composition: PropTypes.object.isRequired,
  onSelect: PropTypes.func.isRequired,
  onProfileClick: PropTypes.func.isRequired,
};

export default CompositionCard;
