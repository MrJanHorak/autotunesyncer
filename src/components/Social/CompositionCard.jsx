import PropTypes from 'prop-types';
import { Heart, MessageCircle, Share2, Play } from 'lucide-react';

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
      <div className='comp-card__thumb-wrap'>
        {thumbUrl ? (
          <img className='comp-card__thumb' src={thumbUrl} alt={composition.title} loading='lazy' />
        ) : (
          <div className='comp-card__thumb-placeholder'>🎬</div>
        )}
        <div className='comp-card__play-overlay'>
          <Play className='comp-card__play-icon' fill='currentColor' />
        </div>
      </div>

      <div className='comp-card__body'>
        <div className='comp-card__title' title={composition.title}>
          {composition.title}
        </div>
        <div className='comp-card__meta'>
          by&nbsp;
          <button
            className='comp-card__author'
            onClick={(e) => { e.stopPropagation(); onProfileClick(composition.user_id); }}
          >
            @{composition.username}
          </button>
          &nbsp;• {formatDate(composition.created_at)}
        </div>

        <div className='comp-card__stats'>
          <span
            className='comp-card__stat'
            style={composition.liked_by_me ? { color: 'var(--social-like)' } : {}}
            title='Likes'
          >
            <Heart
              className='comp-card__stat-icon'
              fill={composition.liked_by_me ? 'currentColor' : 'none'}
              style={composition.liked_by_me ? { color: 'var(--social-like)' } : {}}
            />
            {composition.like_count}
          </span>
          <span className='comp-card__stat' title='Comments'>
            <MessageCircle className='comp-card__stat-icon' />
            {composition.comment_count}
          </span>
          <button
            className='comp-card__share'
            title='Share'
            onClick={(e) => e.stopPropagation()}
          >
            <Share2 className='comp-card__share-icon' />
          </button>
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
