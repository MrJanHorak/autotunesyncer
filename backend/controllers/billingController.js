import Stripe from 'stripe';
import db from '../db/database.js';
import {
  ACTIVE_BILLING_SUBSCRIPTION_STATUSES,
  getBillingAccess,
} from '../services/billingAccessService.js';

const PLAN_DEFINITIONS = [
  {
    key: 'creator',
    name: 'Creator',
    description: 'Solo workspace with exports, private projects, and billing-ready growth.',
    priceEnv: 'STRIPE_PRICE_CREATOR_MONTHLY',
    priceLabel: '$12 / month',
  },
  {
    key: 'studio',
    name: 'Studio',
    description: 'Team billing for shared projects, invite-based collaboration, and admin controls.',
    priceEnv: 'STRIPE_PRICE_STUDIO_MONTHLY',
    priceLabel: '$29 / month',
  },
];

let stripeClient = null;

function getStripeClient() {
  const secretKey = String(process.env.STRIPE_SECRET_KEY || '').trim();
  if (!secretKey) {
    return null;
  }

  if (!stripeClient) {
    stripeClient = new Stripe(secretKey);
  }

  return stripeClient;
}

function getWebhookSecret() {
  return String(process.env.STRIPE_WEBHOOK_SECRET || '').trim();
}

function normalizeBaseUrl(url) {
  return String(url || '')
    .trim()
    .replace(/\/+$/, '');
}

function getAppBaseUrl(req) {
  return (
    normalizeBaseUrl(process.env.APP_URL) ||
    normalizeBaseUrl(process.env.FRONTEND_URL) ||
    normalizeBaseUrl(req.get('origin')) ||
    'http://localhost:5173'
  );
}

function getPlanCatalog() {
  return PLAN_DEFINITIONS.map((plan) => ({
    key: plan.key,
    name: plan.name,
    description: plan.description,
    priceLabel: plan.priceLabel,
    available: Boolean(process.env[plan.priceEnv]),
  }));
}

function findPlanByKey(planKey) {
  const definition = PLAN_DEFINITIONS.find((plan) => plan.key === planKey);
  if (!definition) {
    return null;
  }

  return {
    ...definition,
    priceId: String(process.env[definition.priceEnv] || '').trim() || null,
  };
}

function findPlanByPriceId(priceId) {
  if (!priceId) {
    return null;
  }

  return PLAN_DEFINITIONS.find((plan) => {
    const configuredPriceId = String(process.env[plan.priceEnv] || '').trim();
    return configuredPriceId && configuredPriceId === priceId;
  });
}

function isBillingEnabled() {
  return Boolean(getStripeClient()) && getPlanCatalog().some((plan) => plan.available);
}

function readBillingCustomer(userId) {
  return db
    .prepare(
      `
        SELECT
          user_id,
          stripe_customer_id,
          email,
          created_at,
          updated_at
        FROM billing_customers
        WHERE user_id = ?
      `,
    )
    .get(userId);
}

function readBillingSubscription(userId) {
  return db
    .prepare(
      `
        SELECT
          user_id,
          stripe_subscription_id,
          stripe_customer_id,
          stripe_price_id,
          plan_key,
          status,
          cancel_at_period_end,
          current_period_start,
          current_period_end,
          created_at,
          updated_at
        FROM billing_subscriptions
        WHERE user_id = ?
      `,
    )
    .get(userId);
}

function upsertBillingCustomer({ userId, stripeCustomerId, email }) {
  db.prepare(
    `
      INSERT INTO billing_customers (
        user_id,
        stripe_customer_id,
        email,
        created_at,
        updated_at
      )
      VALUES (?, ?, ?, datetime('now'), datetime('now'))
      ON CONFLICT(user_id) DO UPDATE SET
        stripe_customer_id = excluded.stripe_customer_id,
        email = excluded.email,
        updated_at = datetime('now')
    `,
  ).run(userId, stripeCustomerId, email || '');
}

function upsertBillingSubscription({ userId, stripeCustomerId, subscription }) {
  const priceId = subscription.items?.data?.[0]?.price?.id || null;
  const mappedPlan = findPlanByPriceId(priceId);
  const currentPeriodStart = subscription.current_period_start
    ? new Date(subscription.current_period_start * 1000).toISOString()
    : null;
  const currentPeriodEnd = subscription.current_period_end
    ? new Date(subscription.current_period_end * 1000).toISOString()
    : null;

  db.prepare(
    `
      INSERT INTO billing_subscriptions (
        user_id,
        stripe_subscription_id,
        stripe_customer_id,
        stripe_price_id,
        plan_key,
        status,
        cancel_at_period_end,
        current_period_start,
        current_period_end,
        created_at,
        updated_at
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
      ON CONFLICT(user_id) DO UPDATE SET
        stripe_subscription_id = excluded.stripe_subscription_id,
        stripe_customer_id = excluded.stripe_customer_id,
        stripe_price_id = excluded.stripe_price_id,
        plan_key = excluded.plan_key,
        status = excluded.status,
        cancel_at_period_end = excluded.cancel_at_period_end,
        current_period_start = excluded.current_period_start,
        current_period_end = excluded.current_period_end,
        updated_at = datetime('now')
    `,
  ).run(
    userId,
    subscription.id,
    stripeCustomerId,
    priceId,
    mappedPlan?.key || subscription.metadata?.planKey || null,
    subscription.status,
    subscription.cancel_at_period_end ? 1 : 0,
    currentPeriodStart,
    currentPeriodEnd,
  );
}

function serializeBillingCustomer(customer) {
  if (!customer) {
    return null;
  }

  return {
    email: customer.email,
    createdAt: customer.created_at,
    updatedAt: customer.updated_at,
  };
}

function serializeBillingSubscription(subscription) {
  if (!subscription) {
    return null;
  }

  const mappedPlan = PLAN_DEFINITIONS.find((plan) => plan.key === subscription.plan_key);

  return {
    status: subscription.status,
    planKey: subscription.plan_key,
    planName: mappedPlan?.name || 'Custom',
    priceLabel: mappedPlan?.priceLabel || null,
    currentPeriodStart: subscription.current_period_start,
    currentPeriodEnd: subscription.current_period_end,
    cancelAtPeriodEnd: Boolean(subscription.cancel_at_period_end),
    isActive: ACTIVE_BILLING_SUBSCRIPTION_STATUSES.has(subscription.status),
  };
}

async function ensureStripeCustomer(user) {
  const stripe = getStripeClient();
  if (!stripe) {
    throw new Error('Stripe billing is not configured');
  }

  const existingCustomer = readBillingCustomer(user.id);
  const metadata = { userId: user.id, username: user.username };

  if (existingCustomer?.stripe_customer_id) {
    try {
      await stripe.customers.update(existingCustomer.stripe_customer_id, {
        email: user.email,
        name: user.username,
        metadata,
      });
      upsertBillingCustomer({
        userId: user.id,
        stripeCustomerId: existingCustomer.stripe_customer_id,
        email: user.email,
      });
      return existingCustomer.stripe_customer_id;
    } catch (error) {
      if (error?.type !== 'StripeInvalidRequestError') {
        throw error;
      }
    }
  }

  const customer = await stripe.customers.create({
    email: user.email,
    name: user.username,
    metadata,
  });

  upsertBillingCustomer({
    userId: user.id,
    stripeCustomerId: customer.id,
    email: user.email,
  });

  return customer.id;
}

async function findPromotionCodeId(stripe, rawCode) {
  const code = String(rawCode || '').trim();
  if (!code) {
    return null;
  }

  const result = await stripe.promotionCodes.list({
    code,
    active: true,
    limit: 1,
  });

  return result.data[0]?.id || null;
}

async function readStripeCustomer(stripe, stripeCustomerId) {
  const customer = await stripe.customers.retrieve(stripeCustomerId);
  if (customer.deleted) {
    return null;
  }
  return customer;
}

async function resolveUserIdForCustomer(stripe, stripeCustomerId) {
  const mapped = db
    .prepare(
      `
        SELECT user_id
        FROM billing_customers
        WHERE stripe_customer_id = ?
      `,
    )
    .get(stripeCustomerId);

  if (mapped?.user_id) {
    return mapped.user_id;
  }

  const customer = await readStripeCustomer(stripe, stripeCustomerId);
  if (!customer?.metadata?.userId) {
    return null;
  }

  const user = db
    .prepare(
      `
        SELECT id, email
        FROM users
        WHERE id = ?
      `,
    )
    .get(customer.metadata.userId);

  if (!user) {
    return null;
  }

  upsertBillingCustomer({
    userId: user.id,
    stripeCustomerId,
    email: customer.email || user.email,
  });

  return user.id;
}

async function syncSubscriptionRecord(stripe, subscription, userIdHint = null) {
  const stripeCustomerId =
    typeof subscription.customer === 'string'
      ? subscription.customer
      : subscription.customer?.id;

  if (!stripeCustomerId) {
    return null;
  }

  const userId =
    userIdHint || (await resolveUserIdForCustomer(stripe, stripeCustomerId));

  if (!userId) {
    return null;
  }

  const customer = await readStripeCustomer(stripe, stripeCustomerId);
  const user = db
    .prepare(
      `
        SELECT email
        FROM users
        WHERE id = ?
      `,
    )
    .get(userId);

  upsertBillingCustomer({
    userId,
    stripeCustomerId,
    email: customer?.email || user?.email || '',
  });
  upsertBillingSubscription({ userId, stripeCustomerId, subscription });

  return readBillingSubscription(userId);
}

async function syncLatestSubscriptionForUser(userId) {
  const stripe = getStripeClient();
  if (!stripe) {
    return readBillingSubscription(userId);
  }

  const customer = readBillingCustomer(userId);
  if (!customer?.stripe_customer_id) {
    return readBillingSubscription(userId);
  }

  const subscriptions = await stripe.subscriptions.list({
    customer: customer.stripe_customer_id,
    status: 'all',
    limit: 10,
  });

  const latestSubscription =
    subscriptions.data.find((item) => ACTIVE_BILLING_SUBSCRIPTION_STATUSES.has(item.status)) ||
    subscriptions.data.sort((left, right) => (right.created || 0) - (left.created || 0))[0];

  if (!latestSubscription) {
    return readBillingSubscription(userId);
  }

  await syncSubscriptionRecord(stripe, latestSubscription, userId);
  return readBillingSubscription(userId);
}

export const getBillingStatus = async (req, res) => {
  try {
    const subscription = await syncLatestSubscriptionForUser(req.user.id);
    const billingAccess = getBillingAccess(req.user.id);

    res.json({
      billing: {
        enabled: isBillingEnabled(),
        webhookReady: Boolean(getWebhookSecret()),
        customer: serializeBillingCustomer(readBillingCustomer(req.user.id)),
        subscription: serializeBillingSubscription(subscription),
        capabilities: {
          canExportProject: billingAccess.canExportProject,
          canManageCollaboration: billingAccess.canManageCollaboration,
        },
        plans: getPlanCatalog(),
      },
    });
  } catch (error) {
    console.error('[billing] Failed to load billing status:', error);
    res.status(500).json({ error: 'Failed to load billing status' });
  }
};

export const createCheckoutSession = async (req, res) => {
  if (!isBillingEnabled()) {
    return res.status(503).json({ error: 'Stripe billing is not configured' });
  }

  const plan = findPlanByKey(String(req.body?.planKey || '').trim());
  if (!plan) {
    return res.status(400).json({ error: 'A valid billing plan is required' });
  }

  if (!plan.priceId) {
    return res.status(400).json({ error: 'That billing plan is not configured yet' });
  }

  try {
    const stripe = getStripeClient();
    const activeSubscription = await syncLatestSubscriptionForUser(req.user.id);
    if (
      activeSubscription &&
      ACTIVE_BILLING_SUBSCRIPTION_STATUSES.has(activeSubscription.status)
    ) {
      return res.status(409).json({
        error: 'An active subscription already exists. Use Manage Billing instead.',
      });
    }

    const promotionCodeId = await findPromotionCodeId(
      stripe,
      req.body?.promotionCode,
    );

    if (String(req.body?.promotionCode || '').trim() && !promotionCodeId) {
      return res.status(400).json({ error: 'Promo code was not found or is inactive' });
    }

    const customerId = await ensureStripeCustomer(req.user);
    const baseUrl = getAppBaseUrl(req);
    const session = await stripe.checkout.sessions.create({
      mode: 'subscription',
      customer: customerId,
      client_reference_id: req.user.id,
      success_url: `${baseUrl}/?settings=billing&checkout=success`,
      cancel_url: `${baseUrl}/?settings=billing&checkout=cancelled`,
      line_items: [{ price: plan.priceId, quantity: 1 }],
      allow_promotion_codes: !promotionCodeId,
      discounts: promotionCodeId ? [{ promotion_code: promotionCodeId }] : undefined,
      metadata: {
        userId: req.user.id,
        planKey: plan.key,
      },
      subscription_data: {
        metadata: {
          userId: req.user.id,
          planKey: plan.key,
        },
      },
    });

    if (!session.url) {
      throw new Error('Stripe did not return a checkout URL');
    }

    res.status(201).json({ url: session.url });
  } catch (error) {
    console.error('[billing] Failed to create checkout session:', error);
    res.status(500).json({ error: 'Failed to create checkout session' });
  }
};

export const createBillingPortalSession = async (req, res) => {
  const stripe = getStripeClient();
  if (!stripe) {
    return res.status(503).json({ error: 'Stripe billing is not configured' });
  }

  const customer = readBillingCustomer(req.user.id);
  if (!customer?.stripe_customer_id) {
    return res.status(404).json({ error: 'No billing customer exists for this account yet' });
  }

  try {
    const session = await stripe.billingPortal.sessions.create({
      customer: customer.stripe_customer_id,
      return_url: `${getAppBaseUrl(req)}/?settings=billing`,
    });

    res.json({ url: session.url });
  } catch (error) {
    console.error('[billing] Failed to create portal session:', error);
    res.status(500).json({ error: 'Failed to create billing portal session' });
  }
};

async function handleCheckoutCompleted(stripe, session) {
  const userId = session.client_reference_id || session.metadata?.userId || null;
  const stripeCustomerId =
    typeof session.customer === 'string' ? session.customer : session.customer?.id;

  if (!userId || !stripeCustomerId) {
    return;
  }

  const user = db
    .prepare(
      `
        SELECT email
        FROM users
        WHERE id = ?
      `,
    )
    .get(userId);

  upsertBillingCustomer({
    userId,
    stripeCustomerId,
    email: session.customer_details?.email || user?.email || '',
  });

  if (session.subscription) {
    const subscription = await stripe.subscriptions.retrieve(session.subscription);
    await syncSubscriptionRecord(stripe, subscription, userId);
  }
}

async function handleSubscriptionUpdated(stripe, subscription) {
  await syncSubscriptionRecord(stripe, subscription);
}

export const handleStripeWebhook = async (req, res) => {
  const stripe = getStripeClient();
  const webhookSecret = getWebhookSecret();

  if (!stripe || !webhookSecret) {
    return res.status(503).json({ error: 'Stripe webhook is not configured' });
  }

  const signature = req.headers['stripe-signature'];
  if (!signature) {
    return res.status(400).json({ error: 'Missing Stripe signature header' });
  }

  let event;
  try {
    event = stripe.webhooks.constructEvent(req.body, signature, webhookSecret);
  } catch (error) {
    console.error('[billing] Invalid Stripe webhook signature:', error.message);
    return res.status(400).json({ error: 'Invalid Stripe signature' });
  }

  const alreadyProcessed = db
    .prepare(
      `
        SELECT 1
        FROM billing_webhook_events
        WHERE stripe_event_id = ?
      `,
    )
    .get(event.id);

  if (alreadyProcessed) {
    return res.json({ received: true, duplicate: true });
  }

  try {
    if (event.type === 'checkout.session.completed') {
      await handleCheckoutCompleted(stripe, event.data.object);
    }

    if (
      event.type === 'customer.subscription.created' ||
      event.type === 'customer.subscription.updated' ||
      event.type === 'customer.subscription.deleted'
    ) {
      await handleSubscriptionUpdated(stripe, event.data.object);
    }

    db.prepare(
      `
        INSERT INTO billing_webhook_events (
          stripe_event_id,
          event_type,
          processed_at
        )
        VALUES (?, ?, datetime('now'))
      `,
    ).run(event.id, event.type);

    res.json({ received: true });
  } catch (error) {
    console.error('[billing] Failed to process Stripe webhook:', error);
    res.status(500).json({ error: 'Failed to process Stripe webhook' });
  }
};