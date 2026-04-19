import { useState, useEffect, useCallback } from 'react';
import PropTypes from 'prop-types';
import { ArrowLeft, Heart, Share2, Send, Copy, Twitter, Facebook, MessageCircle } from 'lucide-react';
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
  return new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
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
  const [comments, setComments] = useState([]);
  const [commentPage, setCommentPage] = useState(1);
  const [commentTotal, setCommentTotal] = useState(0);
  const [commentPageSize, setCommentPageSize] = useState(12);
  const [commentText, setCommentText] = useState('');
  const [submittingComment, setSubmittingComment] = useState(false);
  const [likeCount, setLikeCount] = useState(0);
  const [likedByMe, setLikedByMe] = useState(false);
  const [likePending, setLikePending] = useState(false);
  const [shareOpen, setShareOpen] = useState(false);

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

  const handleCopyLink = () => {
    const url = `${window.location.origin}?composition=${compositionId}`;
    if (navigator.clipboard) {
      navigator.clipboard.writeText(url).then(() => {
        setShareOpen(false);
        alert('Link copied!');
      }).catch(() => {});
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
  const authorInitial = composition.username[0].toUpperCase();
  // Use a site page URL for sharing — keeps users on the site and prevents direct video downloads
  const siteShareUrl = `${window.location.origin}?composition=${compositionId}`;
  const shareText = `Check out "${composition.title}" by @${composition.username} on Symphovie!`;

  return (
    <div className='comp-detail'>
      <button className='comp-detail__back' onClick={onBack}>
        <ArrowLeft size={16} /> Back to Feed
      </button>

      <div className='comp-detail__layout'>
        {/* ── Main column ── */}
        <div className='comp-detail__main'>
          <video className='comp-detail__video' src={videoUrl} controls autoPlay={false} />

          <div className='comp-detail__info-card'>
            <h1 className='comp-detail__title'>{composition.title}</h1>

            <div className='comp-detail__author-left'>
              <div className='comp-detail__author-avatar'>{authorInitial}</div>
              <div>
                <button className='comp-detail__author-link' onClick={() => onSelectUser(composition.user_id)}>
                  @{composition.username}
                </button>
                <div className='comp-detail__date'>{formatDate(composition.created_at)}</div>
              </div>
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
                <Heart
                  size={16}
                  className='action-btn-icon'
                  fill={likedByMe ? 'currentColor' : 'none'}
                />
                {likeCount} {likeCount === 1 ? 'Like' : 'Likes'}
              </button>

              <div className='share-dropdown-wrap'>
                <button className='action-btn action-btn--share' onClick={() => setShareOpen((v) => !v)}>
                  <Share2 size={16} className='action-btn-icon' /> Share
                </button>
                {shareOpen && (
                  <div className='share-dropdown'>
                    <a
                      href={`https://twitter.com/intent/tweet?url=${encodeURIComponent(siteShareUrl)}&text=${encodeURIComponent(shareText)}`}
                      target='_blank' rel='noopener noreferrer'
                      className='share-dropdown__item'
                      onClick={() => setShareOpen(false)}
                    >
                      <Twitter size={14} /> Twitter
                    </a>
                    <a
                      href={`https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(siteShareUrl)}`}
                      target='_blank' rel='noopener noreferrer'
                      className='share-dropdown__item'
                      onClick={() => setShareOpen(false)}
                    >
                      <Facebook size={14} /> Facebook
                    </a>
                    <button className='share-dropdown__item' onClick={handleCopyLink}>
                      <Copy size={14} /> Copy Link
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Comments */}
          <div className='comp-detail__comments-card'>
            <h3 className='comp-detail__comments-heading'>
              <MessageCircle size={18} style={{ color: 'var(--social-accent)' }} />
              Comments ({commentTotal})
            </h3>

            {currentUserId && (
              <form className='comment-form' onSubmit={handleAddComment}>
                <div className='comment-form__avatar'>{currentUserId[0]?.toUpperCase() ?? '?'}</div>
                <input
                  className='comment-form__input'
                  value={commentText}
                  onChange={(e) => setCommentText(e.target.value)}
                  placeholder='Add a comment…'
                  maxLength={500}
                />
                <button className='comment-form__submit' type='submit' disabled={submittingComment || !commentText.trim()}>
                  <Send size={16} />
                </button>
              </form>
            )}

            {comments.map((c) => (
              <div key={c.id} className='comment'>
                <div className='comment__avatar'>{c.username[0].toUpperCase()}</div>
                <div className='comment__body-wrap'>
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
                <span>{commentPage}/{commentPages}</span>
                <button onClick={() => setCommentPage((p) => p + 1)} disabled={commentPage >= commentPages}>Next →</button>
              </div>
            )}
          </div>
        </div>

        {/* ── Sidebar ── */}
        <div className='comp-detail__sidebar'>
          {more_from_user.length > 0 && (
            <div className='sidebar-section'>
              <div className='creator-info'>
                <div className='creator-info__avatar'>{authorInitial}</div>
                <div>
                  <button className='comp-detail__author-link' onClick={() => onSelectUser(composition.user_id)}>
                    @{composition.username}
                  </button>
                  <div className='sidebar-section__follow-btn'>
                    <button className='follow-btn follow-btn--follow' onClick={() => onSelectUser(composition.user_id)}>
                      View Profile
                    </button>
                  </div>
                </div>
              </div>
              <h4 className='sidebar-section__title'>More from @{composition.username}</h4>
              {more_from_user.map((item) => (
                <SidebarThumb key={item.id} item={item} onClick={() => onSelectComposition(item.id)} />
              ))}
            </div>
          )}
          {recent.length > 0 && (
            <div className='sidebar-section'>
              <h4 className='sidebar-section__title'>Recently Shared</h4>
              {recent.map((item) => (
                <SidebarThumb key={item.id} item={item} onClick={() => onSelectComposition(item.id)} />
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
