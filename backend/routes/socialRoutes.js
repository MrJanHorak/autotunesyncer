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

const router = express.Router();

// Multer for video uploads (store in temp, controller moves to published/)
const upload = multer({
  dest: os.tmpdir(),
  limits: { fileSize: 500 * 1024 * 1024 }, // 500 MB
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('video/') || /\.(mp4|webm|mov)$/i.test(file.originalname)) {
      cb(null, true);
    } else {
      cb(new Error('Only video files are accepted'));
    }
  },
});

// All social routes require authentication
router.use(authenticateToken);

// Feed
router.get('/feed', getPublicFeed);
router.get('/feed/following', getFollowingFeed);

// Compositions
router.post('/compositions', upload.single('video'), shareComposition);
router.get('/compositions/:id', getComposition);
router.delete('/compositions/:id', deleteComposition);

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

export default router;
