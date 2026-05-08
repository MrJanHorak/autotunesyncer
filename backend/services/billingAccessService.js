import db from '../db/database.js';

export const ACTIVE_BILLING_SUBSCRIPTION_STATUSES = new Set([
  'active',
  'trialing',
  'past_due',
  'unpaid',
]);

const PLAN_RANK = {
  creator: 1,
  studio: 2,
};

const BILLING_PRICE_ENV_KEYS = [
  'STRIPE_PRICE_CREATOR_MONTHLY',
  'STRIPE_PRICE_STUDIO_MONTHLY',
];

function normalizePlanKey(planKey) {
  return Object.prototype.hasOwnProperty.call(PLAN_RANK, planKey)
    ? planKey
    : null;
}

function parseCsvEnv(name) {
  return new Set(
    String(process.env[name] || '')
      .split(',')
      .map((value) => value.trim())
      .filter(Boolean),
  );
}

function isBillingConfigured() {
  const secretKey = String(process.env.STRIPE_SECRET_KEY || '').trim();
  if (!secretKey) {
    return false;
  }

  return BILLING_PRICE_ENV_KEYS.some((key) =>
    Boolean(String(process.env[key] || '').trim()),
  );
}

function buildBypassAccess(reason, subscription = null) {
  return {
    subscription,
    isActive: false,
    planKey: null,
    planRank: Number.MAX_SAFE_INTEGER,
    hasBypass: true,
    bypassReason: reason,
    canExportProject: true,
    canManageCollaboration: true,
  };
}

function getBypassUser(userId) {
  const bypassUserIds = parseCsvEnv('BILLING_BYPASS_USER_IDS');
  const bypassEmails = new Set(
    [...parseCsvEnv('BILLING_BYPASS_EMAILS')].map((value) =>
      value.toLowerCase(),
    ),
  );

  if (bypassUserIds.size === 0 && bypassEmails.size === 0) {
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
    .get(userId);

  if (!user) {
    return null;
  }

  const normalizedEmail = String(user.email || '').trim().toLowerCase();
  if (bypassUserIds.has(user.id) || bypassEmails.has(normalizedEmail)) {
    return user;
  }

  return null;
}

export function getBillingAccess(userId) {
  if (!isBillingConfigured()) {
    return buildBypassAccess('billing_disabled');
  }

  const subscription = db
    .prepare(
      `
        SELECT
          plan_key,
          status,
          cancel_at_period_end,
          current_period_start,
          current_period_end,
          updated_at
        FROM billing_subscriptions
        WHERE user_id = ?
      `,
    )
    .get(userId);

  if (getBypassUser(userId)) {
    return buildBypassAccess('configured_user', subscription || null);
  }

  const normalizedPlanKey = normalizePlanKey(subscription?.plan_key);
  const isActive = Boolean(
    normalizedPlanKey &&
    ACTIVE_BILLING_SUBSCRIPTION_STATUSES.has(subscription?.status),
  );
  const effectivePlanKey = isActive ? normalizedPlanKey : null;
  const planRank = effectivePlanKey ? PLAN_RANK[effectivePlanKey] : 0;

  return {
    subscription,
    isActive,
    planKey: effectivePlanKey,
    planRank,
    hasBypass: false,
    bypassReason: null,
    canExportProject: planRank >= PLAN_RANK.creator,
    canManageCollaboration: planRank >= PLAN_RANK.studio,
  };
}

export function hasBillingPlan(userId, requiredPlanKey) {
  const requiredPlanRank =
    PLAN_RANK[requiredPlanKey] || Number.MAX_SAFE_INTEGER;
  return getBillingAccess(userId).planRank >= requiredPlanRank;
}
