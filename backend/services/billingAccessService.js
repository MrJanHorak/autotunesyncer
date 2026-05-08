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

function normalizePlanKey(planKey) {
  return Object.prototype.hasOwnProperty.call(PLAN_RANK, planKey)
    ? planKey
    : null;
}

export function getBillingAccess(userId) {
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
    canExportProject: planRank >= PLAN_RANK.creator,
    canManageCollaboration: planRank >= PLAN_RANK.studio,
  };
}

export function hasBillingPlan(userId, requiredPlanKey) {
  const requiredPlanRank =
    PLAN_RANK[requiredPlanKey] || Number.MAX_SAFE_INTEGER;
  return getBillingAccess(userId).planRank >= requiredPlanRank;
}
