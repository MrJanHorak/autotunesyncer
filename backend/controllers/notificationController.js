import db from '../db/database.js';

const PAGE_SIZE = 30;

export const getNotifications = (req, res) => {
  const userId = req.user.id;
  const rows = db.prepare(`
    SELECT
      n.id, n.type, n.composition_id, n.comment_id, n.read, n.created_at,
      a.id   AS actor_id,
      a.username AS actor_username,
      c.title    AS composition_title
    FROM notifications n
    JOIN users u ON n.user_id = u.id
    JOIN users a ON n.actor_id = a.id
    LEFT JOIN compositions c ON n.composition_id = c.id
    WHERE n.user_id = ?
    ORDER BY n.created_at DESC
    LIMIT ?
  `).all(userId, PAGE_SIZE);
  res.json({ notifications: rows });
};

export const getUnreadCount = (req, res) => {
  const { cnt } = db.prepare(
    'SELECT COUNT(*) as cnt FROM notifications WHERE user_id = ? AND read = 0'
  ).get(req.user.id);
  res.json({ count: cnt });
};

export const markAllRead = (req, res) => {
  db.prepare('UPDATE notifications SET read = 1 WHERE user_id = ? AND read = 0').run(req.user.id);
  res.json({ ok: true });
};

export const markOneRead = (req, res) => {
  const { notifId } = req.params;
  const row = db.prepare('SELECT user_id FROM notifications WHERE id = ?').get(notifId);
  if (!row) return res.status(404).json({ error: 'Notification not found' });
  if (row.user_id !== req.user.id) return res.status(403).json({ error: 'Forbidden' });
  db.prepare('UPDATE notifications SET read = 1 WHERE id = ?').run(notifId);
  res.json({ ok: true });
};
