import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { v4 as uuidv4 } from 'uuid';
import db from '../db/database.js';
import { JWT_SECRET } from '../middleware/auth.js';

export const register = async (req, res) => {
  try {
    const { username, email, password } = req.body;

    if (!username?.trim() || !email?.trim() || !password) {
      return res.status(400).json({ error: 'username, email, and password are required' });
    }
    if (password.length < 6) {
      return res.status(400).json({ error: 'Password must be at least 6 characters' });
    }

    const existing = db
      .prepare('SELECT id FROM users WHERE email = ? OR username = ?')
      .get(email.toLowerCase(), username.toLowerCase());
    if (existing) {
      return res.status(409).json({ error: 'Email or username already taken' });
    }

    const passwordHash = await bcrypt.hash(password, 12);
    const id = uuidv4();

    db.prepare('INSERT INTO users (id, username, email, password_hash) VALUES (?, ?, ?, ?)')
      .run(id, username.trim(), email.toLowerCase().trim(), passwordHash);

    const token = jwt.sign({ id, username: username.trim(), email: email.toLowerCase().trim() }, JWT_SECRET, { expiresIn: '7d' });
    res.status(201).json({ token, user: { id, username: username.trim(), email: email.toLowerCase().trim(), bio: '' } });
  } catch (err) {
    console.error('Register error:', err);
    res.status(500).json({ error: 'Registration failed' });
  }
};

export const login = async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email?.trim() || !password) {
      return res.status(400).json({ error: 'email and password are required' });
    }

    const user = db.prepare('SELECT * FROM users WHERE email = ?').get(email.toLowerCase().trim());
    if (!user) {
      return res.status(401).json({ error: 'Invalid email or password' });
    }

    const valid = await bcrypt.compare(password, user.password_hash);
    if (!valid) {
      return res.status(401).json({ error: 'Invalid email or password' });
    }

    const token = jwt.sign(
      { id: user.id, username: user.username, email: user.email },
      JWT_SECRET,
      { expiresIn: '7d' }
    );
    res.json({ token, user: { id: user.id, username: user.username, email: user.email, bio: user.bio || '' } });
  } catch (err) {
    console.error('Login error:', err);
    res.status(500).json({ error: 'Login failed' });
  }
};

export const getMe = (req, res) => {
  // Return full profile including bio
  const user = db.prepare('SELECT id, username, email, bio FROM users WHERE id = ?').get(req.user.id);
  res.json({ user: user || req.user });
};

export const updateProfile = async (req, res) => {
  const { username, bio } = req.body;
  const userId = req.user.id;
  if (!username?.trim()) return res.status(400).json({ error: 'username is required' });

  const conflict = db.prepare('SELECT id FROM users WHERE username = ? AND id != ?').get(username.trim(), userId);
  if (conflict) return res.status(409).json({ error: 'Username already taken' });

  db.prepare('UPDATE users SET username = ?, bio = ? WHERE id = ?').run(
    username.trim(), (bio || '').trim(), userId
  );
  const updated = db.prepare('SELECT id, username, email, bio FROM users WHERE id = ?').get(userId);
  // Re-issue token with updated username
  const token = jwt.sign({ id: updated.id, username: updated.username, email: updated.email }, JWT_SECRET, { expiresIn: '7d' });
  res.json({ user: updated, token });
};

export const updateEmail = async (req, res) => {
  const { email, currentPassword } = req.body;
  if (!email?.trim()) return res.status(400).json({ error: 'email is required' });
  if (!currentPassword) return res.status(400).json({ error: 'currentPassword is required' });

  const userId = req.user.id;
  const row = db.prepare('SELECT * FROM users WHERE id = ?').get(userId);
  const valid = await bcrypt.compare(currentPassword, row.password_hash);
  if (!valid) return res.status(401).json({ error: 'Current password is incorrect' });

  const conflict = db.prepare('SELECT id FROM users WHERE email = ? AND id != ?').get(email.toLowerCase().trim(), userId);
  if (conflict) return res.status(409).json({ error: 'Email already in use' });

  db.prepare('UPDATE users SET email = ? WHERE id = ?').run(email.toLowerCase().trim(), userId);
  const updated = db.prepare('SELECT id, username, email, bio FROM users WHERE id = ?').get(userId);
  const token = jwt.sign({ id: updated.id, username: updated.username, email: updated.email }, JWT_SECRET, { expiresIn: '7d' });
  res.json({ user: updated, token });
};

export const updatePassword = async (req, res) => {
  const { currentPassword, newPassword } = req.body;
  if (!currentPassword || !newPassword) return res.status(400).json({ error: 'currentPassword and newPassword are required' });
  if (newPassword.length < 6) return res.status(400).json({ error: 'New password must be at least 6 characters' });

  const row = db.prepare('SELECT * FROM users WHERE id = ?').get(req.user.id);
  const valid = await bcrypt.compare(currentPassword, row.password_hash);
  if (!valid) return res.status(401).json({ error: 'Current password is incorrect' });

  const hash = await bcrypt.hash(newPassword, 12);
  db.prepare('UPDATE users SET password_hash = ? WHERE id = ?').run(hash, req.user.id);
  res.json({ ok: true });
};
