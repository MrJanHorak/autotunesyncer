import express from 'express';
import { register, login, getMe, updateProfile, updateEmail, updatePassword } from '../controllers/authController.js';
import { authenticateToken } from '../middleware/auth.js';

const router = express.Router();

router.post('/register', register);
router.post('/login', login);
router.get('/me', authenticateToken, getMe);
router.patch('/profile', authenticateToken, updateProfile);
router.patch('/email', authenticateToken, updateEmail);
router.patch('/password', authenticateToken, updatePassword);

export default router;
