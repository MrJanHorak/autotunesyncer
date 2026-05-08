import { useEffect, useState } from 'react';
import PropTypes from 'prop-types';
import {
  X,
  User,
  Lock,
  CreditCard,
  Sparkles,
  Check,
  AlertCircle,
} from 'lucide-react';
import { useAuth } from '../../context/AuthContext.jsx';

const API_BASE = 'http://localhost:3000/api';

function getToken() {
  return localStorage.getItem('auth_token');
}

async function apiFetch(path, options = {}) {
  const token = getToken();
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `Error ${res.status}`);
  return data;
}

function FieldMsg({ ok, msg }) {
  if (!msg) return null;
  return (
    <div
      className={`settings-msg ${ok ? 'settings-msg--ok' : 'settings-msg--err'}`}
    >
      {ok ? <Check size={13} /> : <AlertCircle size={13} />} {msg}
    </div>
  );
}

FieldMsg.propTypes = { ok: PropTypes.bool, msg: PropTypes.string };

function getInitialBillingMessage() {
  if (typeof window === 'undefined') {
    return { ok: false, msg: '' };
  }

  const checkoutState = new URLSearchParams(window.location.search).get(
    'checkout',
  );
  if (checkoutState === 'success') {
    return {
      ok: true,
      msg: 'Stripe checkout completed. Billing status will refresh once the subscription is confirmed.',
    };
  }

  if (checkoutState === 'cancelled') {
    return {
      ok: false,
      msg: 'Stripe checkout was cancelled before a subscription was created.',
    };
  }

  return { ok: false, msg: '' };
}

function normalizeTab(tab) {
  return tab === 'billing' || tab === 'security' ? tab : 'profile';
}

const Settings = ({ initialTab, onClose }) => {
  const { user, updateUser } = useAuth();
  const [tab, setTab] = useState(normalizeTab(initialTab));

  // Profile tab state
  const [username, setUsername] = useState(user?.username || '');
  const [bio, setBio] = useState(user?.bio || '');
  const [profileSaving, setProfileSaving] = useState(false);
  const [profileMsg, setProfileMsg] = useState({ ok: false, msg: '' });

  // Email tab state
  const [email, setEmail] = useState(user?.email || '');
  const [emailPassword, setEmailPassword] = useState('');
  const [emailSaving, setEmailSaving] = useState(false);
  const [emailMsg, setEmailMsg] = useState({ ok: false, msg: '' });

  // Password tab state
  const [currentPw, setCurrentPw] = useState('');
  const [newPw, setNewPw] = useState('');
  const [confirmPw, setConfirmPw] = useState('');
  const [pwSaving, setPwSaving] = useState(false);
  const [pwMsg, setPwMsg] = useState({ ok: false, msg: '' });

  // Billing tab state
  const [billingLoading, setBillingLoading] = useState(false);
  const [billingAction, setBillingAction] = useState('');
  const [billingData, setBillingData] = useState({
    enabled: false,
    webhookReady: false,
    customer: null,
    subscription: null,
    plans: [],
  });
  const [promotionCode, setPromotionCode] = useState('');
  const [billingMsg, setBillingMsg] = useState(getInitialBillingMessage);

  useEffect(() => {
    setTab(normalizeTab(initialTab));
  }, [initialTab]);

  useEffect(() => {
    if (tab !== 'billing') {
      return undefined;
    }

    let cancelled = false;

    const loadBillingStatus = async () => {
      setBillingLoading(true);
      try {
        const data = await apiFetch('/billing/status');
        if (cancelled) return;
        setBillingData(
          data.billing || {
            enabled: false,
            webhookReady: false,
            customer: null,
            subscription: null,
            plans: [],
          },
        );
      } catch (err) {
        if (cancelled) return;
        setBillingMsg((current) =>
          current.msg ? current : { ok: false, msg: err.message },
        );
      } finally {
        if (!cancelled) {
          setBillingLoading(false);
        }
      }
    };

    void loadBillingStatus();

    return () => {
      cancelled = true;
    };
  }, [tab]);

  const handleSaveProfile = async (e) => {
    e.preventDefault();
    setProfileSaving(true);
    setProfileMsg({ ok: false, msg: '' });
    try {
      const data = await apiFetch('/auth/profile', {
        method: 'PATCH',
        body: JSON.stringify({ username: username.trim(), bio: bio.trim() }),
      });
      updateUser(data.user, data.token);
      setProfileMsg({ ok: true, msg: 'Profile updated!' });
    } catch (err) {
      setProfileMsg({ ok: false, msg: err.message });
    } finally {
      setProfileSaving(false);
    }
  };

  const handleSaveEmail = async (e) => {
    e.preventDefault();
    setEmailSaving(true);
    setEmailMsg({ ok: false, msg: '' });
    try {
      const data = await apiFetch('/auth/email', {
        method: 'PATCH',
        body: JSON.stringify({
          email: email.trim(),
          currentPassword: emailPassword,
        }),
      });
      updateUser(data.user, data.token);
      setEmailMsg({ ok: true, msg: 'Email updated!' });
      setEmailPassword('');
    } catch (err) {
      setEmailMsg({ ok: false, msg: err.message });
    } finally {
      setEmailSaving(false);
    }
  };

  const handleSavePassword = async (e) => {
    e.preventDefault();
    if (newPw !== confirmPw) {
      setPwMsg({ ok: false, msg: 'New passwords do not match' });
      return;
    }
    setPwSaving(true);
    setPwMsg({ ok: false, msg: '' });
    try {
      await apiFetch('/auth/password', {
        method: 'PATCH',
        body: JSON.stringify({
          currentPassword: currentPw,
          newPassword: newPw,
        }),
      });
      setPwMsg({ ok: true, msg: 'Password changed!' });
      setCurrentPw('');
      setNewPw('');
      setConfirmPw('');
    } catch (err) {
      setPwMsg({ ok: false, msg: err.message });
    } finally {
      setPwSaving(false);
    }
  };

  const handleStartCheckout = async (planKey) => {
    setBillingAction(`checkout:${planKey}`);
    setBillingMsg({ ok: false, msg: '' });
    try {
      const data = await apiFetch('/billing/checkout-session', {
        method: 'POST',
        body: JSON.stringify({
          planKey,
          promotionCode: promotionCode.trim() || undefined,
        }),
      });

      if (!data.url) {
        throw new Error('Checkout URL was not returned');
      }

      window.location.assign(data.url);
    } catch (err) {
      setBillingMsg({ ok: false, msg: err.message });
      setBillingAction('');
    }
  };

  const handleOpenBillingPortal = async () => {
    setBillingAction('portal');
    setBillingMsg({ ok: false, msg: '' });
    try {
      const data = await apiFetch('/billing/portal-session', {
        method: 'POST',
      });

      if (!data.url) {
        throw new Error('Billing portal URL was not returned');
      }

      window.location.assign(data.url);
    } catch (err) {
      setBillingMsg({ ok: false, msg: err.message });
      setBillingAction('');
    }
  };

  return (
    <div
      className='settings-overlay'
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className='settings-modal'>
        <div className='settings-modal__header'>
          <span className='settings-modal__title'>Account Settings</span>
          <button
            className='settings-modal__close'
            onClick={onClose}
            aria-label='Close settings'
          >
            <X size={18} />
          </button>
        </div>

        <div className='settings-tabs'>
          <button
            className={`settings-tab ${tab === 'profile' ? 'settings-tab--active' : ''}`}
            onClick={() => setTab('profile')}
          >
            <User size={14} /> Profile
          </button>
          <button
            className={`settings-tab ${tab === 'security' ? 'settings-tab--active' : ''}`}
            onClick={() => setTab('security')}
          >
            <Lock size={14} /> Security
          </button>
          <button
            className={`settings-tab ${tab === 'billing' ? 'settings-tab--active' : ''}`}
            onClick={() => setTab('billing')}
          >
            <CreditCard size={14} /> Billing
          </button>
        </div>

        {tab === 'profile' && (
          <form className='settings-form' onSubmit={handleSaveProfile}>
            <label className='settings-label'>
              Username
              <input
                className='settings-input'
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                minLength={2}
                maxLength={30}
              />
            </label>
            <label className='settings-label'>
              Bio
              <textarea
                className='settings-input settings-textarea'
                value={bio}
                onChange={(e) => setBio(e.target.value)}
                rows={3}
                maxLength={200}
                placeholder='Tell the world about yourself…'
              />
              <span className='settings-char-count'>{bio.length}/200</span>
            </label>
            <FieldMsg {...profileMsg} />
            <button
              className='settings-save-btn'
              type='submit'
              disabled={profileSaving}
            >
              {profileSaving ? 'Saving…' : 'Save Profile'}
            </button>
          </form>
        )}

        {tab === 'security' && (
          <div className='settings-security'>
            {/* Email section */}
            <form className='settings-form' onSubmit={handleSaveEmail}>
              <h4 className='settings-section-title'>Email Address</h4>
              <label className='settings-label'>
                New email
                <input
                  className='settings-input'
                  type='email'
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </label>
              <label className='settings-label'>
                Current password (required to change email)
                <input
                  className='settings-input'
                  type='password'
                  value={emailPassword}
                  onChange={(e) => setEmailPassword(e.target.value)}
                  required
                  placeholder='Your current password'
                />
              </label>
              <FieldMsg {...emailMsg} />
              <button
                className='settings-save-btn'
                type='submit'
                disabled={emailSaving}
              >
                {emailSaving ? 'Saving…' : 'Update Email'}
              </button>
            </form>

            <hr className='settings-divider' />

            {/* Password section */}
            <form className='settings-form' onSubmit={handleSavePassword}>
              <h4 className='settings-section-title'>Change Password</h4>
              <label className='settings-label'>
                Current password
                <input
                  className='settings-input'
                  type='password'
                  value={currentPw}
                  onChange={(e) => setCurrentPw(e.target.value)}
                  required
                />
              </label>
              <label className='settings-label'>
                New password
                <input
                  className='settings-input'
                  type='password'
                  value={newPw}
                  onChange={(e) => setNewPw(e.target.value)}
                  required
                  minLength={6}
                />
              </label>
              <label className='settings-label'>
                Confirm new password
                <input
                  className='settings-input'
                  type='password'
                  value={confirmPw}
                  onChange={(e) => setConfirmPw(e.target.value)}
                  required
                />
              </label>
              <FieldMsg {...pwMsg} />
              <button
                className='settings-save-btn'
                type='submit'
                disabled={pwSaving}
              >
                {pwSaving ? 'Saving…' : 'Change Password'}
              </button>
            </form>
          </div>
        )}

        {tab === 'billing' && (
          <div className='settings-billing'>
            <div className='settings-billing__hero'>
              <div>
                <h4 className='settings-billing__title'>
                  Stripe billing foundation
                </h4>
                <p className='settings-billing__copy'>
                  Keep subscriptions, promo codes, invoices, and plan changes
                  out of the editor workflow.
                </p>
              </div>
              <span className='settings-billing__badge'>
                <Sparkles size={14} /> Private tool stack
              </span>
            </div>

            <FieldMsg {...billingMsg} />

            {billingLoading ? (
              <div className='settings-billing__panel'>
                <p className='settings-billing__hint'>
                  Loading billing status…
                </p>
              </div>
            ) : !billingData.enabled ? (
              <div className='settings-billing__panel'>
                <h4 className='settings-billing__panel-title'>
                  Billing is not configured yet
                </h4>
                <p className='settings-billing__hint'>
                  Set `STRIPE_SECRET_KEY` and at least one Stripe price ID in
                  the backend environment to enable checkout.
                </p>
                {!billingData.webhookReady ? (
                  <p className='settings-billing__hint'>
                    `STRIPE_WEBHOOK_SECRET` is also missing, so subscription
                    status sync is not active yet.
                  </p>
                ) : null}
              </div>
            ) : (
              <>
                <div className='settings-billing__panel'>
                  <div className='settings-billing__panel-row'>
                    <div>
                      <h4 className='settings-billing__panel-title'>
                        Current plan
                      </h4>
                      <p className='settings-billing__hint'>
                        {billingData.subscription
                          ? `${billingData.subscription.planName} · ${billingData.subscription.status}`
                          : 'No paid plan is active yet.'}
                      </p>
                    </div>
                    <span className='settings-billing__pill'>
                      {billingData.subscription?.currentPeriodEnd
                        ? `Renews ${new Date(billingData.subscription.currentPeriodEnd).toLocaleDateString()}`
                        : 'Free'}
                    </span>
                  </div>

                  {billingData.subscription?.cancelAtPeriodEnd ? (
                    <p className='settings-billing__hint'>
                      This subscription is set to cancel at the end of the
                      current billing period.
                    </p>
                  ) : null}

                  <div className='settings-billing__actions'>
                    <button
                      type='button'
                      className='settings-save-btn'
                      onClick={handleOpenBillingPortal}
                      disabled={
                        !billingData.customer || billingAction === 'portal'
                      }
                    >
                      {billingAction === 'portal'
                        ? 'Opening…'
                        : 'Manage Billing'}
                    </button>
                  </div>
                </div>

                <div className='settings-billing__panel'>
                  <label className='settings-label'>
                    Promotion code
                    <input
                      className='settings-input'
                      value={promotionCode}
                      onChange={(e) => setPromotionCode(e.target.value)}
                      placeholder='Optional promo or launch code'
                      maxLength={64}
                    />
                  </label>
                  <p className='settings-billing__hint'>
                    Leave this blank to let Stripe handle promotion codes during
                    checkout.
                  </p>
                </div>

                <div className='settings-billing__plan-grid'>
                  {billingData.plans.map((plan) => (
                    <article
                      key={plan.key}
                      className='settings-billing__plan-card'
                    >
                      <div className='settings-billing__plan-copy'>
                        <div>
                          <h4 className='settings-billing__plan-name'>
                            {plan.name}
                          </h4>
                          <div className='settings-billing__plan-price'>
                            {plan.priceLabel}
                          </div>
                        </div>
                        <p className='settings-billing__plan-desc'>
                          {plan.description}
                        </p>
                      </div>
                      <button
                        type='button'
                        className='settings-save-btn'
                        onClick={() => handleStartCheckout(plan.key)}
                        disabled={!plan.available || Boolean(billingAction)}
                      >
                        {billingAction === `checkout:${plan.key}`
                          ? 'Redirecting…'
                          : plan.available
                            ? `Start ${plan.name}`
                            : 'Not configured'}
                      </button>
                    </article>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

Settings.propTypes = {
  initialTab: PropTypes.oneOf(['profile', 'security', 'billing']),
  onClose: PropTypes.func.isRequired,
};

export default Settings;
