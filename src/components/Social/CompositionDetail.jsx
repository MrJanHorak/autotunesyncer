import { useState, useEffect, useCallback } from 'react';
import PropTypes from 'prop-types';
import './Social.css';

const API_BASE = 'http://localhost:3000/api';
const MEDIA_BASE = 'http://localhost:3000';

function getToken() {
  return localStorage.getItem('auth_token');
}

function getCurrentUserId() {
  try {
    const token = getToken();
    if (!token) return null;
    const payload = JSON.parse(atob(token.split('.')[1]));
    return payload.id;
  } catch { return null; }
}

async function apiFetch(path, options = {}) {
  const token = getToken();
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });
  if (!res.ok) {
    let msg = `API error ${res.status}`;
    try { const d = await res.json(); msg = d.error || msg; } catch { /* ignore */ }
    throw new Error(msg);
  }
  return res.json();
}

function formatDate(dateStr) {
  const d = new Date(dateStr);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function SidebarThumb({ item, onClick }) {
  const thumbUrl = item.thumbnail_path ? `${MEDIA_BASE}/published/${item.thumbnail_path}` : null;
  return (
    <div className='sidebar-thumb' onClick={onClick}>
      {thumbUrl
        ? <img className='sidebar-thumb__img' src={thumbUrl} alt={item.title} loading='lazy' />
        : <div className='sidebar-thumb__placeholder'>🎬</div>
      }
      <div>
        <div className='sidebar-thumb__title'>{item.title}</div>
        {item.username && <div className='sidebar-thumb__meta'>@{item.username}</div>}
        <div className='sidebar-thumb__meta'>{formatDate(item.created_at)}</div>
      </div>
    </div>
  );
}

SidebarThumb.propTypes = {
  item: PropTypes.shape({
    id: PropTypes.string,
    title: PropTypes.string,
    thumbnail_path: PropTypes.string,
    created_at: PropTypes.string,
    username: PropTypes.string,
  }).isRequired,
  onClick: PropTypes.func.isRequired,
};

const CompositionDetail = ({ compositionId, onBack, onSelectUser, onSelectComposition }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Comments state
  const [comments, setComments] = useState([]);
  const [commentPage, setCommentPage] = useState(1);
  const [commentTotal, setCommentTotal] = useState(0);
  const [commentPageSize, setCommentPageSize] = useState(12);
  const [commentText, setCommentText] = useState('');
  const [submittingComment, setSubmittingComment] = useState(false);

  // Like state
  const [likeCount, setLikeCount] = useState(0);
  const [likedByMe, setLikedByMe] = useState(false);
  const [likePending, setLikePending] = useState(false);

  const currentUserId = getCurrentUserId();

  const fetchDetail = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiFetch(`/social/compositions/${compositionId}`);
      setData(result);
      setLikeCount(result.composition.like_count);
      setLikedByMe(result.composition.liked_by_me);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [compositionId]);

  const fetchComments = useCallback(async (page) => {
    try {
      const result = await apiFetch(`/social/compositions/${compositionId}/comments?page=${page}`);
      setComments(result.comments);
      setCommentTotal(result.total);
      setCommentPageSize(result.page_size);
    } catch { /* non-critical */ }
  }, [compositionId]);

  useEffect(() => { fetchDetail(); }, [fetchDetail]);
  useEffect(() => { fetchComments(commentPage); }, [fetchComments, commentPage]);

  const handleLike = async () => {
    if (likePending) return;
    setLikePending(true);
    try {
      const method = likedByMe ? 'DELETE' : 'PUT';
      const result = await apiFetch(`/social/compositions/${compositionId}/like`, { method, headers: {} });
      setLikeCount(result.like_count);
      setLikedByMe(result.liked_by_me);
    } catch { /* ignore */ } finally {
      setLikePending(false);
    }
  };

  const handleShare = () => {
    const url = `${MEDIA_BASE}/published/${data.composition.video_path}`;
    if (navigator.clipboard) {
      navigator.clipboard.writeText(url).then(() => alert('Video URL copied to clipboard!')).catch(() => {});
    }
  };

  const handleAddComment = async (e) => {
    e.preventDefault();
    if (!commentText.trim() || submittingComment) return;
    setSubmittingComment(true);
    try {
      const result = await apiFetch(`/social/compositions/${compositionId}/comments`, {
        method: 'POST',
        body: JSON.stringify({ body: commentText.trim() }),
      });
      setComments((prev) => [...prev, result.comment]);
      setCommentTotal((n) => n + 1);
      setCommentText('');
    } catch { /* ignore */ } finally {
      setSubmittingComment(false);
    }
  };

  const handleDeleteComment = async (commentId) => {
    try {
      await apiFetch(`/social/comments/${commentId}`, { method: 'DELETE', headers: {} });
      setComments((prev) => prev.filter((c) => c.id !== commentId));
      setCommentTotal((n) => n - 1);
    } catch { /* ignore */ }
  };

  if (loading) return <div className='social-loading'>Loading…</div>;
  if (error) return <div className='social-error'>⚠️ {error} <button onClick={fetchDetail}>Retry</button></div>;
  if (!data) return null;

  const { composition, more_from_user, recent } = data;
  const videoUrl = `${MEDIA_BASE}/published/${composition.video_path}`;
  const commentPages = Math.ceil(commentTotal / commentPageSize);

  return (
    <div className='comp-detail'>
      <button className='comp-detail__back' onClick={onBack}>← Back to Feed</button>

      <div className='comp-detail__layout'>
        {/* ── Main column ── */}
        <div className='comp-detail__main'>
          <video
            className='comp-detail__video'
            src={videoUrl}
            controls
            autoPlay={false}
          />

          <div className='comp-detail__info'>
            <h1 className='comp-detail__title'>{composition.title}</h1>
            <div className='comp-detail__author-row'>
              <button className='comp-detail__author-link' onClick={() => onSelectUser(composition.user_id)}>
                @{composition.username}
              </button>
              <span className='comp-detail__date'>{formatDate(composition.created_at)}</span>
            </div>
            {composition.description && (
              <p className='comp-detail__description'>{composition.description}</p>
            )}

            <div className='comp-detail__actions'>
              <button
                className={`action-btn action-btn--like${likedByMe ? ' active' : ''}`}
                onClick={handleLike}
                disabled={likePending}
              >
                {likedByMe ? '❤️' : '🤍'} {likeCount} {likeCount === 1 ? 'Like' : 'Likes'}
              </button>
              <button className='action-btn action-btn--share' onClick={handleShare}>
                🔗 Copy Link
              </button>
            </div>
          </div>

          {/* Comments */}
          <div className='comp-detail__comments'>
            <h3>💬 Comments ({commentTotal})</h3>
            <form className='comment-form' onSubmit={handleAddComment}>
              <textarea
                value={commentText}
                onChange={(e) => setCommentText(e.target.value)}
                placeholder='Add a comment…'
                maxLength={500}
              />
              <button type='submit' disabled={submittingComment || !commentText.trim()}>
                {submittingComment ? '…' : 'Post'}
              </button>
            </form>

            {comments.map((c) => (
              <div key={c.id} className='comment'>
                <div className='comment__avatar'>{c.username[0].toUpperCase()}</div>
                <div className='comment__content' style={{ flex: 1 }}>
                  <div className='comment__header'>
                    <button className='comment__username' onClick={() => onSelectUser(c.user_id)}>
                      @{c.username}
                    </button>
                    <span className='comment__date'>{formatDate(c.created_at)}</span>
                    {c.user_id === currentUserId && (
                      <button className='comment__delete' onClick={() => handleDeleteComment(c.id)} title='Delete'>✕</button>
                    )}
                  </div>
                  <p className='comment__body'>{c.body}</p>
                </div>
              </div>
            ))}

            {commentPages > 1 && (
              <div className='social-feed__pagination' style={{ justifyContent: 'flex-start', marginTop: '1rem' }}>
                <button onClick={() => setCommentPage((p) => p - 1)} disabled={commentPage <= 1}>← Prev</button>
                <span style={{ fontSize: '0.8rem', color: '#6b7280' }}>{commentPage}/{commentPages}</span>
                <button onClick={() => setCommentPage((p) => p + 1)} disabled={commentPage >= commentPages}>Next →</button>
              </div>
            )}
          </div>
        </div>

        {/* ── Sidebar ── */}
        <div className='comp-detail__sidebar'>
          {more_from_user.length > 0 && (
            <div className='sidebar-section'>
              <h4>More from @{composition.username}</h4>
              {more_from_user.map((item) => (
                <SidebarThumb key={item.id} item={item} onClick={() => onSelectComposition(item.id)} />
              ))}
            </div>
          )}
          {recent.length > 0 && (
            <div className='sidebar-section'>
              <h4>Recently Shared</h4>
              {recent.map((item) => (
                <SidebarThumb key={item.id} item={{ ...item }} onClick={() => onSelectComposition(item.id)} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

CompositionDetail.propTypes = {
  compositionId: PropTypes.string.isRequired,
  onBack: PropTypes.func.isRequired,
  onSelectUser: PropTypes.func.isRequired,
  onSelectComposition: PropTypes.func.isRequired,
};

export default CompositionDetail;
