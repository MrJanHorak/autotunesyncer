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
    res.status(201).json({ token, user: { id, username: username.trim(), email: email.toLowerCase().trim() } });
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
    res.json({ token, user: { id: user.id, username: user.username, email: user.email } });
  } catch (err) {
    console.error('Login error:', err);
    res.status(500).json({ error: 'Login failed' });
  }
};

export const getMe = (req, res) => {
  res.json({ user: req.user });
};
