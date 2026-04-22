import express from 'express';
import multer from 'multer';
import os from 'os';
import { authenticateToken } from '../middleware/auth.js';
import {
  shareComposition,
  getPublicFeed,
  getFollowingFeed,
  getComposition,
  deleteComposition,
  updateVisibility,
  likeComposition,
  unlikeComposition,
  getComments,
  addComment,
  deleteComment,
  getUserProfile,
  getUserCompositions,
  followUser,
  unfollowUser,
} from '../controllers/socialController.js';
import {
  getNotifications,
  getUnreadCount,
  markAllRead,
  markOneRead,
} from '../controllers/notificationController.js';

const router = express.Router();

// Multer for video + optional thumbnail uploads (store in temp, controller moves to published/)
const upload = multer({
  dest: os.tmpdir(),
  limits: { fileSize: 500 * 1024 * 1024 }, // 500 MB
  fileFilter: (req, file, cb) => {
    const isVideo = file.mimetype.startsWith('video/') || /\.(mp4|webm|mov)$/i.test(file.originalname);
    const isImage = file.mimetype.startsWith('image/') || /\.(jpe?g|png|gif|webp)$/i.test(file.originalname);
    if (isVideo || isImage) {
      cb(null, true);
    } else {
      cb(new Error('Only video or image files are accepted'));
    }
  },
});

// All social routes require authentication
router.use(authenticateToken);

// Feed
router.get('/feed', getPublicFeed);
router.get('/feed/following', getFollowingFeed);

// Compositions
router.post('/compositions', upload.fields([{ name: 'video', maxCount: 1 }, { name: 'thumbnail', maxCount: 1 }]), shareComposition);
router.get('/compositions/:id', getComposition);
router.delete('/compositions/:id', deleteComposition);
router.patch('/compositions/:id/visibility', updateVisibility);

// Likes (explicit put/delete for idempotency)
router.put('/compositions/:id/like', likeComposition);
router.delete('/compositions/:id/like', unlikeComposition);

// Comments
router.get('/compositions/:id/comments', getComments);
router.post('/compositions/:id/comments', addComment);
router.delete('/comments/:commentId', deleteComment);

// Users
router.get('/users/:userId', getUserProfile);
router.get('/users/:userId/compositions', getUserCompositions);
router.put('/users/:userId/follow', followUser);
router.delete('/users/:userId/follow', unfollowUser);

// Notifications
router.get('/notifications', getNotifications);
router.get('/notifications/unread-count', getUnreadCount);
router.patch('/notifications/read-all', markAllRead);
router.patch('/notifications/:notifId/read', markOneRead);

export default router;
