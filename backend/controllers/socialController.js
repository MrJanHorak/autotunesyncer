import { v4 as uuidv4 } from 'uuid';
import { join, dirname, extname } from 'path';
import { fileURLToPath } from 'url';
import { mkdirSync, existsSync, unlinkSync, copyFileSync } from 'fs';
import { execFile } from 'child_process';
import { promisify } from 'util';
import db from '../db/database.js';

const execFileAsync = promisify(execFile);
const __dirname = dirname(fileURLToPath(import.meta.url));
const PUBLISHED_DIR = join(__dirname, '../published');

const PAGE_SIZE = 12;

function ensurePublishedDir(userId) {
  const dir = join(PUBLISHED_DIR, userId);
  mkdirSync(dir, { recursive: true });
  return dir;
}

async function generateThumbnail(videoPath, outputPath) {
  try {
    await execFileAsync('ffmpeg', [
      '-y', '-ss', '1', '-i', videoPath,
      '-vframes', '1', '-q:v', '2',
      outputPath,
    ]);
    return outputPath;
  } catch {
    return null;
  }
}

// ── Compositions ─────────────────────────────────────────────────────────────

export const shareComposition = async (req, res) => {
  const { title, description = '' } = req.body;
  if (!title?.trim()) return res.status(400).json({ error: 'title is required' });

  const videoFile = req.files?.video?.[0];
  const thumbnailFile = req.files?.thumbnail?.[0];

  if (!videoFile) return res.status(400).json({ error: 'video file is required' });

  const id = uuidv4();
  const userId = req.user.id;
  const dir = ensurePublishedDir(userId);

  const videoFilename = `${id}.mp4`;
  const videoPath = join(dir, videoFilename);
  const thumbFilename = `${id}_thumb.jpg`;
  const thumbPath = join(dir, thumbFilename);

  // Move the uploaded video to published dir
  try {
    copyFileSync(videoFile.path, videoPath);
    try { unlinkSync(videoFile.path); } catch { /* ignore */ }
  } catch (err) {
    return res.status(500).json({ error: 'Failed to save video file' });
  }

  // Use user-provided thumbnail or auto-generate from video
  let thumbnailPath;
  if (thumbnailFile) {
    try {
      copyFileSync(thumbnailFile.path, thumbPath);
      try { unlinkSync(thumbnailFile.path); } catch { /* ignore */ }
      thumbnailPath = thumbPath;
    } catch { /* fall back to auto-generate */ }
  }
  if (!thumbnailPath) {
    thumbnailPath = await generateThumbnail(videoPath, thumbPath);
  }

  const relVideoPath = `${userId}/${videoFilename}`;
  const relThumbPath = thumbnailPath ? `${userId}/${thumbFilename}` : null;

  try {
    db.prepare(`
      INSERT INTO compositions (id, user_id, title, description, video_path, thumbnail_path)
      VALUES (?, ?, ?, ?, ?, ?)
    `).run(id, userId, title.trim(), description.trim(), relVideoPath, relThumbPath);

    const composition = db.prepare(`
      SELECT c.*, u.username FROM compositions c
      JOIN users u ON c.user_id = u.id
      WHERE c.id = ?
    `).get(id);

    return res.status(201).json({ composition });
  } catch (err) {
    // DB failed — clean up files
    try { unlinkSync(videoPath); } catch { /* ignore */ }
    try { if (thumbnailPath) unlinkSync(thumbPath); } catch { /* ignore */ }
    console.error('shareComposition DB error:', err);
    return res.status(500).json({ error: 'Failed to save composition' });
  }
};

const FEED_QUERY = (whereClause) => `
  WITH like_counts AS (
    SELECT composition_id, COUNT(*) as cnt FROM likes GROUP BY composition_id
  ),
  comment_counts AS (
    SELECT composition_id, COUNT(*) as cnt FROM comments GROUP BY composition_id
  )
  SELECT
    c.id, c.title, c.description, c.video_path, c.thumbnail_path,
    c.duration, c.created_at, c.visibility,
    u.id as user_id, u.username,
    COALESCE(lc.cnt, 0) as like_count,
    COALESCE(cc.cnt, 0) as comment_count
  FROM compositions c
  JOIN users u ON c.user_id = u.id
  LEFT JOIN like_counts lc ON lc.composition_id = c.id
  LEFT JOIN comment_counts cc ON cc.composition_id = c.id
  ${whereClause}
  ORDER BY c.created_at DESC
  LIMIT ? OFFSET ?
`;

function attachUserLiked(rows, userId) {
  if (!rows.length) return rows;
  const ids = rows.map((r) => r.id);
  const liked = new Set(
    db.prepare(
      `SELECT composition_id FROM likes WHERE user_id = ? AND composition_id IN (${ids.map(() => '?').join(',')})`
    ).all(userId, ...ids).map((r) => r.composition_id)
  );
  return rows.map((r) => ({ ...r, liked_by_me: liked.has(r.id) }));
}

export const getPublicFeed = (req, res) => {
  const page = Math.max(1, parseInt(req.query.page) || 1);
  const offset = (page - 1) * PAGE_SIZE;
  // Public feed: only visibility='public'
  const rows = db.prepare(FEED_QUERY("WHERE c.visibility = 'public'")).all(PAGE_SIZE, offset);
  const total = db.prepare("SELECT COUNT(*) as n FROM compositions WHERE visibility = 'public'").get().n;
  res.json({ compositions: attachUserLiked(rows, req.user.id), page, total, page_size: PAGE_SIZE });
};

export const getFollowingFeed = (req, res) => {
  const page = Math.max(1, parseInt(req.query.page) || 1);
  const offset = (page - 1) * PAGE_SIZE;
  // Following feed: public + followers-only from people you follow
  const rows = db.prepare(
    FEED_QUERY(`WHERE c.user_id IN (SELECT following_id FROM follows WHERE follower_id = ?)
                AND c.visibility IN ('public', 'followers')`)
  ).all(req.user.id, PAGE_SIZE, offset);
  const total = db.prepare(
    `SELECT COUNT(*) as n FROM compositions
     WHERE user_id IN (SELECT following_id FROM follows WHERE follower_id = ?)
     AND visibility IN ('public', 'followers')`
  ).get(req.user.id).n;
  res.json({ compositions: attachUserLiked(rows, req.user.id), page, total, page_size: PAGE_SIZE });
};

export const getComposition = (req, res) => {
  const { id } = req.params;
  const viewerId = req.user.id;
  const row = db.prepare(`
    WITH like_counts AS (SELECT composition_id, COUNT(*) as cnt FROM likes GROUP BY composition_id),
         comment_counts AS (SELECT composition_id, COUNT(*) as cnt FROM comments GROUP BY composition_id)
    SELECT c.id, c.title, c.description, c.video_path, c.thumbnail_path,
           c.duration, c.created_at, c.visibility,
           u.id as user_id, u.username,
           COALESCE(lc.cnt, 0) as like_count,
           COALESCE(cc.cnt, 0) as comment_count
    FROM compositions c
    JOIN users u ON c.user_id = u.id
    LEFT JOIN like_counts lc ON lc.composition_id = c.id
    LEFT JOIN comment_counts cc ON cc.composition_id = c.id
    WHERE c.id = ?
  `).get(id);

  if (!row) return res.status(404).json({ error: 'Composition not found' });

  // Enforce visibility
  if (row.visibility === 'private' && row.user_id !== viewerId) {
    return res.status(403).json({ error: 'This composition is private' });
  }
  if (row.visibility === 'followers') {
    const isOwner = row.user_id === viewerId;
    const isFollower = !!db.prepare('SELECT 1 FROM follows WHERE follower_id = ? AND following_id = ?').get(viewerId, row.user_id);
    if (!isOwner && !isFollower) {
      return res.status(403).json({ error: 'This composition is for followers only' });
    }
  }

  const liked = db.prepare('SELECT 1 FROM likes WHERE user_id = ? AND composition_id = ?')
    .get(viewerId, id);

  // More from same user — only show what the viewer can see
  const isFollowingAuthor = !!db.prepare('SELECT 1 FROM follows WHERE follower_id = ? AND following_id = ?').get(viewerId, row.user_id);
  const allowedVis = viewerId === row.user_id ? ["'public'", "'followers'", "'private'"]
    : isFollowingAuthor ? ["'public'", "'followers'"]
    : ["'public'"];
  const moreFromUser = db.prepare(`
    SELECT id, title, thumbnail_path, created_at FROM compositions
    WHERE user_id = ? AND id != ? AND visibility IN (${allowedVis.join(',')})
    ORDER BY created_at DESC LIMIT 4
  `).all(row.user_id, id);

  // Recent from anyone — only public
  const excludeIds = [id, ...moreFromUser.map((c) => c.id)];
  const recent = db.prepare(`
    SELECT c.id, c.title, c.thumbnail_path, c.created_at, u.username FROM compositions c
    JOIN users u ON c.user_id = u.id
    WHERE c.id NOT IN (${excludeIds.map(() => '?').join(',')}) AND c.visibility = 'public'
    ORDER BY c.created_at DESC LIMIT 4
  `).all(...excludeIds);

  res.json({ composition: { ...row, liked_by_me: !!liked }, more_from_user: moreFromUser, recent });
};

export const deleteComposition = (req, res) => {
  const { id } = req.params;
  const row = db.prepare('SELECT * FROM compositions WHERE id = ?').get(id);
  if (!row) return res.status(404).json({ error: 'Composition not found' });
  if (row.user_id !== req.user.id) return res.status(403).json({ error: 'Forbidden' });

  db.prepare('DELETE FROM compositions WHERE id = ?').run(id);

  // Delete files
  try { unlinkSync(join(PUBLISHED_DIR, row.video_path)); } catch { /* ignore */ }
  if (row.thumbnail_path) {
    try { unlinkSync(join(PUBLISHED_DIR, row.thumbnail_path)); } catch { /* ignore */ }
  }

  res.json({ ok: true });
};

// ── Likes ─────────────────────────────────────────────────────────────────────

export const likeComposition = (req, res) => {
  const { id } = req.params;
  const actorId = req.user.id;
  const comp = db.prepare('SELECT user_id FROM compositions WHERE id = ?').get(id);
  if (!comp) return res.status(404).json({ error: 'Composition not found' });

  db.transaction(() => {
    db.prepare('INSERT OR IGNORE INTO likes (user_id, composition_id) VALUES (?, ?)').run(actorId, id);
    // Notify composition owner (skip self-like)
    if (comp.user_id !== actorId) {
      db.prepare(`
        INSERT OR IGNORE INTO notifications (id, user_id, actor_id, type, composition_id)
        VALUES (?, ?, ?, 'like', ?)
      `).run(uuidv4(), comp.user_id, actorId, id);
    }
  })();

  const { cnt } = db.prepare('SELECT COUNT(*) as cnt FROM likes WHERE composition_id = ?').get(id);
  res.json({ like_count: cnt, liked_by_me: true });
};

export const unlikeComposition = (req, res) => {
  const { id } = req.params;
  db.prepare('DELETE FROM likes WHERE user_id = ? AND composition_id = ?').run(req.user.id, id);
  const { cnt } = db.prepare('SELECT COUNT(*) as cnt FROM likes WHERE composition_id = ?').get(id);
  res.json({ like_count: cnt, liked_by_me: false });
};

// ── Comments ──────────────────────────────────────────────────────────────────

export const getComments = (req, res) => {
  const { id } = req.params;
  const page = Math.max(1, parseInt(req.query.page) || 1);
  const offset = (page - 1) * PAGE_SIZE;
  const rows = db.prepare(`
    SELECT c.id, c.body, c.created_at, u.id as user_id, u.username
    FROM comments c JOIN users u ON c.user_id = u.id
    WHERE c.composition_id = ?
    ORDER BY c.created_at ASC
    LIMIT ? OFFSET ?
  `).all(id, PAGE_SIZE, offset);
  const total = db.prepare('SELECT COUNT(*) as n FROM comments WHERE composition_id = ?').get(id).n;
  res.json({ comments: rows, page, total, page_size: PAGE_SIZE });
};

export const addComment = (req, res) => {
  const { id } = req.params;
  const { body } = req.body;
  if (!body?.trim()) return res.status(400).json({ error: 'comment body is required' });
  const comp = db.prepare('SELECT user_id FROM compositions WHERE id = ?').get(id);
  if (!comp) return res.status(404).json({ error: 'Composition not found' });

  const commentId = uuidv4();
  const actorId = req.user.id;

  db.transaction(() => {
    db.prepare('INSERT INTO comments (id, user_id, composition_id, body) VALUES (?, ?, ?, ?)').run(
      commentId, actorId, id, body.trim()
    );
    // Notify composition owner (skip self-comment)
    if (comp.user_id !== actorId) {
      db.prepare(`
        INSERT INTO notifications (id, user_id, actor_id, type, composition_id, comment_id)
        VALUES (?, ?, ?, 'comment', ?, ?)
      `).run(uuidv4(), comp.user_id, actorId, id, commentId);
    }
  })();

  const comment = db.prepare(`
    SELECT c.id, c.body, c.created_at, u.id as user_id, u.username
    FROM comments c JOIN users u ON c.user_id = u.id WHERE c.id = ?
  `).get(commentId);
  res.status(201).json({ comment });
};

export const deleteComment = (req, res) => {
  const { commentId } = req.params;
  const row = db.prepare('SELECT * FROM comments WHERE id = ?').get(commentId);
  if (!row) return res.status(404).json({ error: 'Comment not found' });
  if (row.user_id !== req.user.id) return res.status(403).json({ error: 'Forbidden' });
  db.prepare('DELETE FROM comments WHERE id = ?').run(commentId);
  res.json({ ok: true });
};

// ── Users / Follows ───────────────────────────────────────────────────────────

export const getUserProfile = (req, res) => {
  const { userId } = req.params;
  const viewerId = req.user.id;
  const user = db.prepare('SELECT id, username, bio, created_at FROM users WHERE id = ?').get(userId);
  if (!user) return res.status(404).json({ error: 'User not found' });

  const { followers } = db.prepare('SELECT COUNT(*) as followers FROM follows WHERE following_id = ?').get(userId);
  const { following } = db.prepare('SELECT COUNT(*) as following FROM follows WHERE follower_id = ?').get(userId);
  const isFollowing = !!db.prepare('SELECT 1 FROM follows WHERE follower_id = ? AND following_id = ?').get(viewerId, userId);

  // Composition count reflects what the viewer can actually see
  let compCount;
  if (viewerId === userId) {
    compCount = db.prepare('SELECT COUNT(*) as n FROM compositions WHERE user_id = ?').get(userId).n;
  } else if (isFollowing) {
    compCount = db.prepare("SELECT COUNT(*) as n FROM compositions WHERE user_id = ? AND visibility IN ('public','followers')").get(userId).n;
  } else {
    compCount = db.prepare("SELECT COUNT(*) as n FROM compositions WHERE user_id = ? AND visibility = 'public'").get(userId).n;
  }

  res.json({ user: { ...user, followers, following, compositions: compCount, is_following: isFollowing } });
};

export const getUserCompositions = (req, res) => {
  const { userId } = req.params;
  const viewerId = req.user.id;
  if (!db.prepare('SELECT 1 FROM users WHERE id = ?').get(userId)) {
    return res.status(404).json({ error: 'User not found' });
  }
  const page = Math.max(1, parseInt(req.query.page) || 1);
  const offset = (page - 1) * PAGE_SIZE;

  // Determine what visibility levels this viewer can see
  let visFilter;
  if (viewerId === userId) {
    // Own profile — see everything
    visFilter = `c.user_id = ?`;
  } else {
    const isFollowing = !!db.prepare('SELECT 1 FROM follows WHERE follower_id = ? AND following_id = ?').get(viewerId, userId);
    if (isFollowing) {
      visFilter = `c.user_id = ? AND c.visibility IN ('public', 'followers')`;
    } else {
      visFilter = `c.user_id = ? AND c.visibility = 'public'`;
    }
  }

  const rows = db.prepare(FEED_QUERY(`WHERE ${visFilter}`)).all(userId, PAGE_SIZE, offset);
  const total = db.prepare(`SELECT COUNT(*) as n FROM compositions c WHERE ${visFilter}`).get(userId).n;
  res.json({ compositions: attachUserLiked(rows, viewerId), page, total, page_size: PAGE_SIZE });
};

export const followUser = (req, res) => {
  const { userId } = req.params;
  const actorId = req.user.id;
  if (userId === actorId) return res.status(400).json({ error: 'Cannot follow yourself' });
  if (!db.prepare('SELECT 1 FROM users WHERE id = ?').get(userId)) {
    return res.status(404).json({ error: 'User not found' });
  }

  db.transaction(() => {
    db.prepare('INSERT OR IGNORE INTO follows (follower_id, following_id) VALUES (?, ?)').run(actorId, userId);
    db.prepare(`
      INSERT OR IGNORE INTO notifications (id, user_id, actor_id, type)
      VALUES (?, ?, ?, 'follow')
    `).run(uuidv4(), userId, actorId);
  })();

  const { cnt } = db.prepare('SELECT COUNT(*) as cnt FROM follows WHERE following_id = ?').get(userId);
  res.json({ is_following: true, followers: cnt });
};

export const unfollowUser = (req, res) => {
  const { userId } = req.params;
  db.prepare('DELETE FROM follows WHERE follower_id = ? AND following_id = ?').run(req.user.id, userId);
  const { cnt } = db.prepare('SELECT COUNT(*) as cnt FROM follows WHERE following_id = ?').get(userId);
  res.json({ is_following: false, followers: cnt });
};

// ── Visibility ────────────────────────────────────────────────────────────────

export const updateVisibility = (req, res) => {
  const { id } = req.params;
  const { visibility } = req.body;
  if (!['public', 'followers', 'private'].includes(visibility)) {
    return res.status(400).json({ error: 'visibility must be public, followers, or private' });
  }
  const row = db.prepare('SELECT user_id FROM compositions WHERE id = ?').get(id);
  if (!row) return res.status(404).json({ error: 'Composition not found' });
  if (row.user_id !== req.user.id) return res.status(403).json({ error: 'Forbidden' });
  db.prepare('UPDATE compositions SET visibility = ? WHERE id = ?').run(visibility, id);
  res.json({ id, visibility });
};
