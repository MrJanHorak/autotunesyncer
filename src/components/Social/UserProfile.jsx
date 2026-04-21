import { useState, useEffect, useCallback } from 'react';
import PropTypes from 'prop-types';
import { ArrowLeft, UserPlus, UserCheck, Calendar, Music, Globe, Users, Lock } from 'lucide-react';
import CompositionCard from './CompositionCard.jsx';
import { useAuth } from '../../context/AuthContext.jsx';
import './Social.css';

const API_BASE = 'http://localhost:3000/api';

function getToken() {
  return localStorage.getItem('auth_token');
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

const UserProfile = ({ userId, onBack, onSelectComposition, onSelectUser }) => {
  const { user: currentUser } = useAuth();
  const [profile, setProfile] = useState(null);
  const [compositions, setCompositions] = useState([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [pageSize, setPageSize] = useState(12);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [followPending, setFollowPending] = useState(false);
  // Tracks in-flight visibility updates per composition
  const [visUpdating, setVisUpdating] = useState({});

  const currentUserId = currentUser?.id;
  const isOwnProfile = currentUserId === userId;

  const handleVisibilityChange = async (compositionId, newVis) => {
    setVisUpdating((prev) => ({ ...prev, [compositionId]: true }));
    // Optimistic update
    setCompositions((prev) => prev.map((c) => c.id === compositionId ? { ...c, visibility: newVis } : c));
    try {
      await apiFetch(`/social/compositions/${compositionId}/visibility`, {
        method: 'PATCH',
        body: JSON.stringify({ visibility: newVis }),
      });
    } catch {
      // Revert on failure
      setCompositions((prev) => prev.map((c) => c.id === compositionId ? { ...c, visibility: c.visibility } : c));
    } finally {
      setVisUpdating((prev) => ({ ...prev, [compositionId]: false }));
    }
  };

  const fetchProfile = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [profileData, compData] = await Promise.all([
        apiFetch(`/social/users/${userId}`),
        apiFetch(`/social/users/${userId}/compositions?page=1`),
      ]);
      setProfile(profileData.user);
      setCompositions(compData.compositions);
      setTotal(compData.total);
      setPageSize(compData.page_size);
      setPage(1);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [userId]);

  const fetchCompositions = useCallback(async (p) => {
    try {
      const data = await apiFetch(`/social/users/${userId}/compositions?page=${p}`);
      setCompositions(data.compositions);
      setTotal(data.total);
    } catch { /* non-critical */ }
  }, [userId]);

  useEffect(() => { fetchProfile(); }, [fetchProfile]);
  useEffect(() => { if (page > 1) fetchCompositions(page); }, [page, fetchCompositions]);

  const handleFollow = async () => {
    if (followPending || !profile) return;
    setFollowPending(true);
    try {
      const method = profile.is_following ? 'DELETE' : 'PUT';
      const result = await apiFetch(`/social/users/${userId}/follow`, { method, headers: {} });
      setProfile((prev) => ({ ...prev, is_following: result.is_following, followers: result.followers }));
    } catch { /* ignore */ } finally {
      setFollowPending(false);
    }
  };

  if (loading) return <div className='social-loading'>Loading…</div>;
  if (error) return <div className='social-error'>⚠️ {error}</div>;
  if (!profile) return null;

  const totalPages = Math.ceil(total / pageSize);
  const initial = profile.username[0].toUpperCase();

  return (
    <div className='user-profile'>
      <button className='user-profile__back' onClick={onBack}>
        <ArrowLeft size={16} />
        Back to Feed
      </button>

      <div className='user-profile__card'>
        <div className='user-profile__banner' />
        <div className='user-profile__body'>
          <div className='user-profile__header-row'>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <div className='user-profile__avatar'>{initial}</div>
              <div>
                <div className='user-profile__username'>@{profile.username}</div>
                {profile.bio && <p className='user-profile__bio'>{profile.bio}</p>}
                <div className='user-profile__info-row'>
                  {profile.joined_at && (
                    <span className='user-profile__info-item'>
                      <Calendar className='user-profile__info-icon' />
                      Joined {new Date(profile.joined_at).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}
                    </span>
                  )}
                  <span className='user-profile__info-item'>
                    <Music className='user-profile__info-icon' />
                    {profile.compositions} compositions
                  </span>
                </div>
              </div>
            </div>

            {!isOwnProfile && (
              <button
                className={`follow-btn ${profile.is_following ? 'follow-btn--unfollow' : 'follow-btn--follow'}`}
                onClick={handleFollow}
                disabled={followPending}
              >
                {profile.is_following
                  ? <><UserCheck className='follow-btn-icon' /> Following</>
                  : <><UserPlus className='follow-btn-icon' /> Follow</>
                }
              </button>
            )}
          </div>

          <div className='user-profile__stats'>
            <div className='user-profile__stat'>
              <span className='user-profile__stat-value'>{profile.followers?.toLocaleString()}</span>
              <span className='user-profile__stat-label'>Followers</span>
            </div>
            <div className='user-profile__stat'>
              <span className='user-profile__stat-value'>{profile.following?.toLocaleString()}</span>
              <span className='user-profile__stat-label'>Following</span>
            </div>
            <div className='user-profile__stat'>
              <span className='user-profile__stat-value'>{profile.compositions?.toLocaleString()}</span>
              <span className='user-profile__stat-label'>Compositions</span>
            </div>
          </div>
        </div>
      </div>

      <h2 className='user-profile__compositions-title'>Compositions ({total})</h2>

      {isOwnProfile && (
        <p className='user-profile__visibility-note'>
          <Globe size={13} /> Set who can see each composition. Changes take effect immediately.
        </p>
      )}

      <div className='social-feed__grid'>
        {compositions.length === 0 ? (
          <div className='social-feed__empty'>
            <h3>No compositions yet</h3>
            <p>{isOwnProfile ? 'Share your first composition from the Editor tab!' : "This user hasn't shared anything yet."}</p>
          </div>
        ) : (
          compositions.map((c) => {
            const vis = c.visibility || 'public';
            return (
              <div key={c.id} className='user-profile__comp-wrap'>
                <CompositionCard
                  composition={c}
                  onSelect={onSelectComposition}
                  onProfileClick={onSelectUser}
                />
                {isOwnProfile && (
                  <div className='vis-badge-row'>
                    {[
                      { value: 'public', label: 'Public', Icon: Globe },
                      { value: 'followers', label: 'Followers', Icon: Users },
                      { value: 'private', label: 'Private', Icon: Lock },
                    ].map(({ value, label, Icon }) => (
                      <button
                        key={value}
                        className={`vis-badge ${vis === value ? 'vis-badge--active' : ''}`}
                        onClick={() => handleVisibilityChange(c.id, value)}
                        disabled={visUpdating[c.id]}
                        title={label}
                      >
                        <Icon size={11} /> {label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {totalPages > 1 && (
        <div className='social-feed__pagination'>
          <button onClick={() => setPage((p) => p - 1)} disabled={page <= 1}>← Prev</button>
          <span>Page {page} of {totalPages}</span>
          <button onClick={() => setPage((p) => p + 1)} disabled={page >= totalPages}>Next →</button>
        </div>
      )}
    </div>
  );
};

UserProfile.propTypes = {
  userId: PropTypes.string.isRequired,
  onBack: PropTypes.func.isRequired,
  onSelectComposition: PropTypes.func.isRequired,
  onSelectUser: PropTypes.func.isRequired,
};

export default UserProfile;
