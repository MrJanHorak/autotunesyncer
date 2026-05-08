import express from 'express';
import { authenticateToken } from '../middleware/auth.js';
import {
  createBillingPortalSession,
  createCheckoutSession,
  getBillingStatus,
} from '../controllers/billingController.js';

const router = express.Router();

router.use(authenticateToken);

router.get('/status', getBillingStatus);
router.post('/checkout-session', createCheckoutSession);
router.post('/portal-session', createBillingPortalSession);

export default router;