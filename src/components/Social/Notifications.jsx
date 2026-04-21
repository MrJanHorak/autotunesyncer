import { useState, useEffect, useCallback } from 'react';
import PropTypes from 'prop-types';
import { X, Heart, MessageCircle, UserPlus, Bell } from 'lucide-react';
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

function timeAgo(dateStr) {
  const diff = Math.floor((Date.now() - new Date(dateStr).getTime()) / 1000);
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

const TYPE_ICON = {
  like: <Heart size={14} style={{ color: '#f472b6' }} />,
  comment: <MessageCircle size={14} style={{ color: '#c084fc' }} />,
  follow: <UserPlus size={14} style={{ color: '#34d399' }} />,
};

function notifMessage(n) {
  if (n.type === 'like') return <>liked your <strong>{n.composition_title || 'composition'}</strong></>;
  if (n.type === 'comment') return <>commented on <strong>{n.composition_title || 'your composition'}</strong></>;
  if (n.type === 'follow') return <>started following you</>;
  return n.type;
}

const Notifications = ({ onClose, onSelectComposition }) => {
  const [notifications, setNotifications] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchNotifications = useCallback(async () => {
    setLoading(true);
    try {
      const data = await apiFetch('/social/notifications');
      setNotifications(data.notifications);
    } catch { /* ignore */ } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchNotifications();
  }, [fetchNotifications]);

  const handleMarkAllRead = async () => {
    await apiFetch('/social/notifications/read-all', { method: 'PATCH', headers: {} });
    setNotifications((prev) => prev.map((n) => ({ ...n, read: 1 })));
  };

  const handleClickNotif = async (notif) => {
    if (!notif.read) {
      apiFetch(`/social/notifications/${notif.id}/read`, { method: 'PATCH', headers: {} });
      setNotifications((prev) => prev.map((n) => n.id === notif.id ? { ...n, read: 1 } : n));
    }
    if (notif.composition_id && onSelectComposition) {
      onClose();
      onSelectComposition(notif.composition_id);
    }
  };

  const unread = notifications.filter((n) => !n.read).length;

  return (
    <div className='notif-panel'>
      <div className='notif-panel__header'>
        <span className='notif-panel__title'>
          <Bell size={16} /> Notifications {unread > 0 && <span className='notif-unread-badge'>{unread}</span>}
        </span>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          {unread > 0 && (
            <button className='notif-panel__mark-all' onClick={handleMarkAllRead}>Mark all read</button>
          )}
          <button className='notif-panel__close' onClick={onClose} aria-label='Close notifications'>
            <X size={16} />
          </button>
        </div>
      </div>

      <div className='notif-panel__list'>
        {loading && <div className='notif-panel__empty'>Loading…</div>}
        {!loading && notifications.length === 0 && (
          <div className='notif-panel__empty'>
            <Bell size={32} style={{ opacity: 0.3 }} />
            <p>No notifications yet</p>
          </div>
        )}
        {!loading && notifications.map((n) => (
          <button
            key={n.id}
            className={`notif-item ${!n.read ? 'notif-item--unread' : ''}`}
            onClick={() => handleClickNotif(n)}
          >
            <span className='notif-item__icon'>{TYPE_ICON[n.type]}</span>
            <span className='notif-item__body'>
              <strong>@{n.actor_username}</strong> {notifMessage(n)}
            </span>
            <span className='notif-item__time'>{timeAgo(n.created_at)}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

Notifications.propTypes = {
  onClose: PropTypes.func.isRequired,
  onSelectComposition: PropTypes.func.isRequired,
};

export default Notifications;
