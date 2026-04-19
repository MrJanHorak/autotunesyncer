import { useState, useEffect, useCallback } from 'react';
import PropTypes from 'prop-types';
import CompositionCard from './CompositionCard.jsx';
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

const SocialFeed = ({ onSelectComposition, onSelectUser }) => {
  const [tab, setTab] = useState('all'); // 'all' | 'following'
  const [compositions, setCompositions] = useState([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [pageSize, setPageSize] = useState(12);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchFeed = useCallback(async (currentTab, currentPage) => {
    setLoading(true);
    setError(null);
    try {
      const endpoint = currentTab === 'following'
        ? `/social/feed/following?page=${currentPage}`
        : `/social/feed?page=${currentPage}`;
      const data = await apiFetch(endpoint);
      setCompositions(data.compositions);
      setTotal(data.total);
      setPageSize(data.page_size);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchFeed(tab, page);
  }, [tab, page, fetchFeed]);

  const handleTabChange = (newTab) => {
    setTab(newTab);
    setPage(1);
  };

  const totalPages = Math.ceil(total / pageSize);

  return (
    <div className='social-feed'>
      <div className='social-feed__tabs'>
        <button
          className={`social-feed__tab${tab === 'all' ? ' social-feed__tab--active' : ''}`}
          onClick={() => handleTabChange('all')}
        >
          🌍 All Compositions
        </button>
        <button
          className={`social-feed__tab${tab === 'following' ? ' social-feed__tab--active' : ''}`}
          onClick={() => handleTabChange('following')}
        >
          👥 Following
        </button>
      </div>

      {loading && <div className='social-loading'>Loading…</div>}
      {error && <div className='social-error'>⚠️ {error}</div>}

      {!loading && !error && (
        <>
          <div className='social-feed__grid'>
            {compositions.length === 0 ? (
              <div className='social-feed__empty'>
                <h3>
                  {tab === 'following'
                    ? 'No compositions from people you follow yet'
                    : 'No compositions shared yet'}
                </h3>
                <p>
                  {tab === 'following'
                    ? 'Follow some creators to see their work here.'
                    : 'Be the first to share a composition!'}
                </p>
              </div>
            ) : (
              compositions.map((c) => (
                <CompositionCard
                  key={c.id}
                  composition={c}
                  onSelect={onSelectComposition}
                  onProfileClick={onSelectUser}
                />
              ))
            )}
          </div>

          {totalPages > 1 && (
            <div className='social-feed__pagination'>
              <button onClick={() => setPage((p) => p - 1)} disabled={page <= 1}>
                ← Prev
              </button>
              <span style={{ fontSize: '0.875rem', color: '#6b7280' }}>
                Page {page} of {totalPages}
              </span>
              <button onClick={() => setPage((p) => p + 1)} disabled={page >= totalPages}>
                Next →
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
};

SocialFeed.propTypes = {
  onSelectComposition: PropTypes.func.isRequired,
  onSelectUser: PropTypes.func.isRequired,
};

export default SocialFeed;
