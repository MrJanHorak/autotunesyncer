import Database from 'better-sqlite3';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { mkdirSync, existsSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));

if (!existsSync(__dirname)) mkdirSync(__dirname, { recursive: true });

const db = new Database(join(__dirname, 'autotunesyncer.db'));

db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');
db.pragma('busy_timeout = 5000');

db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    username    TEXT NOT NULL UNIQUE,
    email       TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    bio         TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    state       TEXT DEFAULT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS compositions (
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title          TEXT NOT NULL,
    description    TEXT NOT NULL DEFAULT '',
    video_path     TEXT NOT NULL,
    thumbnail_path TEXT,
    duration       REAL NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS follows (
    follower_id  TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    following_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (follower_id, following_id)
  );

  CREATE TABLE IF NOT EXISTS likes (
    user_id        TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    composition_id TEXT NOT NULL REFERENCES compositions(id) ON DELETE CASCADE,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, composition_id)
  );

  CREATE TABLE IF NOT EXISTS comments (
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    composition_id TEXT NOT NULL REFERENCES compositions(id) ON DELETE CASCADE,
    body           TEXT NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE INDEX IF NOT EXISTS idx_projects_user_id        ON projects(user_id);
  CREATE INDEX IF NOT EXISTS idx_users_email             ON users(email);
  CREATE INDEX IF NOT EXISTS idx_users_username          ON users(username);
  CREATE INDEX IF NOT EXISTS idx_compositions_user_id    ON compositions(user_id, created_at);
  CREATE INDEX IF NOT EXISTS idx_compositions_created_at ON compositions(created_at);
  CREATE INDEX IF NOT EXISTS idx_likes_composition_id    ON likes(composition_id);
  CREATE INDEX IF NOT EXISTS idx_comments_composition_id ON comments(composition_id, created_at);
  CREATE INDEX IF NOT EXISTS idx_follows_follower_id     ON follows(follower_id);
  CREATE INDEX IF NOT EXISTS idx_follows_following_id    ON follows(following_id);

  CREATE TABLE IF NOT EXISTS project_clips (
    project_id     TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    instrument_key TEXT NOT NULL,
    file_path      TEXT NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (project_id, instrument_key)
  );

  CREATE INDEX IF NOT EXISTS idx_project_clips_project_id ON project_clips(project_id);
`);

// Additive migrations for columns added after initial schema
const userCols = db.pragma('table_info(users)').map((c) => c.name);
if (!userCols.includes('bio')) {
  db.exec(`ALTER TABLE users ADD COLUMN bio TEXT NOT NULL DEFAULT ''`);
}

const compCols = db.pragma('table_info(compositions)').map((c) => c.name);
if (!compCols.includes('visibility')) {
  db.exec(`ALTER TABLE compositions ADD COLUMN visibility TEXT NOT NULL DEFAULT 'public'`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_compositions_visibility ON compositions(visibility)`);
}

// Notifications table
db.exec(`
  CREATE TABLE IF NOT EXISTS notifications (
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    actor_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type           TEXT NOT NULL,
    composition_id TEXT REFERENCES compositions(id) ON DELETE CASCADE,
    comment_id     TEXT REFERENCES comments(id) ON DELETE CASCADE,
    read           INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE INDEX IF NOT EXISTS idx_notif_user_id ON notifications(user_id, created_at DESC);
  CREATE INDEX IF NOT EXISTS idx_notif_unread  ON notifications(user_id, read, created_at DESC);

  -- Prevent duplicate like notifications from same actor on same composition
  CREATE UNIQUE INDEX IF NOT EXISTS idx_notif_like_dedup
    ON notifications(user_id, actor_id, composition_id) WHERE type = 'like';

  -- Prevent duplicate follow notifications from same actor
  CREATE UNIQUE INDEX IF NOT EXISTS idx_notif_follow_dedup
    ON notifications(user_id, actor_id) WHERE type = 'follow';
`);

export default db;
