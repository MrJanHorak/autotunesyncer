import { v4 as uuidv4 } from 'uuid';
import db from '../db/database.js';
import {
  canManageProject,
  getProjectAccess,
} from '../services/projectAccessService.js';
import { getBillingAccess } from '../services/billingAccessService.js';

const INVITE_ROLE = 'editor';
const INVITE_LINK_TTL_MS = 14 * 24 * 60 * 60 * 1000;

function requireStudioForCollaboration(access, res) {
  const billingAccess = getBillingAccess(access.ownerId);
  if (billingAccess.canManageCollaboration) {
    return true;
  }

  res.status(403).json({
    error:
      'The Studio plan is required before this project can invite new collaborators.',
  });
  return false;
}

function getInviteLinkRecord(token) {
  return db
    .prepare(
      `
        SELECT
          l.*,
          p.name AS project_name,
          p.user_id AS owner_id,
          inviter.username AS inviter_username
        FROM project_invite_links l
        JOIN projects p ON p.id = l.project_id
        JOIN users inviter ON inviter.id = l.inviter_id
        WHERE l.token = ?
      `,
    )
    .get(token);
}

function validateInviteLink(link) {
  if (!link) {
    return { ok: false, status: 404, error: 'Invite link not found' };
  }

  if (link.status !== 'pending') {
    return {
      ok: false,
      status: 400,
      error: 'Invite link is no longer active',
    };
  }

  const expiresAtMs = new Date(link.expires_at).getTime();
  if (Number.isFinite(expiresAtMs) && expiresAtMs <= Date.now()) {
    db.prepare(
      `
        UPDATE project_invite_links
        SET status = 'expired'
        WHERE id = ? AND status = 'pending'
      `,
    ).run(link.id);

    return { ok: false, status: 400, error: 'Invite link has expired' };
  }

  return { ok: true };
}

export const listPendingProjectInvites = (req, res) => {
  const invites = db
    .prepare(
      `
        SELECT
          i.id,
          i.project_id,
          i.role,
          i.created_at,
          p.name AS project_name,
          inviter.username AS inviter_username
        FROM project_invites i
        JOIN projects p ON p.id = i.project_id
        JOIN users inviter ON inviter.id = i.inviter_id
        WHERE i.invitee_id = ? AND i.status = 'pending'
        ORDER BY i.created_at DESC
      `,
    )
    .all(req.user.id);

  res.json({ invites });
};

export const listProjectCollaborators = (req, res) => {
  const access = getProjectAccess(req.params.id, req.user.id);
  if (!access) return res.status(404).json({ error: 'Project not found' });
  const billingAccess = canManageProject(access)
    ? getBillingAccess(access.ownerId)
    : null;

  const collaborators = db
    .prepare(
      `
        SELECT
          u.id,
          u.username,
          pc.role,
          pc.created_at
        FROM project_collaborators pc
        JOIN users u ON u.id = pc.user_id
        WHERE pc.project_id = ?
        ORDER BY pc.created_at ASC
      `,
    )
    .all(req.params.id);

  const pendingInvites = canManageProject(access)
    ? db
        .prepare(
          `
            SELECT
              i.id,
              i.role,
              i.created_at,
              u.id AS invitee_id,
              u.username AS invitee_username
            FROM project_invites i
            JOIN users u ON u.id = i.invitee_id
            WHERE i.project_id = ? AND i.status = 'pending'
            ORDER BY i.created_at DESC
          `,
        )
        .all(req.params.id)
    : [];

  const pendingInviteLinks = canManageProject(access)
    ? db
        .prepare(
          `
            SELECT
              id,
              role,
              token,
              created_at,
              expires_at
            FROM project_invite_links
            WHERE project_id = ? AND status = 'pending'
            ORDER BY created_at DESC
          `,
        )
        .all(req.params.id)
    : [];

  res.json({
    owner: {
      id: access.ownerId,
      username: access.ownerUsername,
      role: 'owner',
    },
    billing: billingAccess
      ? {
          requiredPlan: 'studio',
          canManageCollaboration: billingAccess.canManageCollaboration,
          planKey: billingAccess.planKey,
        }
      : null,
    collaborators,
    pendingInvites,
    pendingInviteLinks,
  });
};

export const createProjectInviteLink = (req, res) => {
  const access = getProjectAccess(req.params.id, req.user.id);
  if (!access) return res.status(404).json({ error: 'Project not found' });
  if (!canManageProject(access)) {
    return res
      .status(403)
      .json({ error: 'Only the project owner can create invite links' });
  }
  if (!requireStudioForCollaboration(access, res)) {
    return;
  }

  const inviteLinkId = uuidv4();
  const token = uuidv4();
  const expiresAt = new Date(Date.now() + INVITE_LINK_TTL_MS).toISOString();

  db.prepare(
    `
      INSERT INTO project_invite_links (
        id,
        project_id,
        inviter_id,
        role,
        token,
        expires_at
      )
      VALUES (?, ?, ?, ?, ?, ?)
    `,
  ).run(inviteLinkId, req.params.id, req.user.id, INVITE_ROLE, token, expiresAt);

  res.status(201).json({
    inviteLink: {
      id: inviteLinkId,
      token,
      role: INVITE_ROLE,
      expiresAt,
    },
  });
};

export const revokeProjectInviteLink = (req, res) => {
  const inviteLink = db
    .prepare(
      `
        SELECT l.id, l.project_id, l.status
        FROM project_invite_links l
        WHERE l.id = ?
      `,
    )
    .get(req.params.inviteLinkId);

  if (!inviteLink) {
    return res.status(404).json({ error: 'Invite link not found' });
  }

  const access = getProjectAccess(inviteLink.project_id, req.user.id);
  if (!access) return res.status(404).json({ error: 'Project not found' });
  if (!canManageProject(access)) {
    return res
      .status(403)
      .json({ error: 'Only the project owner can revoke invite links' });
  }

  if (inviteLink.status !== 'pending') {
    return res.status(400).json({ error: 'Invite link is no longer active' });
  }

  db.prepare(
    `
      UPDATE project_invite_links
      SET status = 'revoked', revoked_at = datetime('now')
      WHERE id = ?
    `,
  ).run(inviteLink.id);

  res.json({ ok: true });
};

export const getProjectInviteLink = (req, res) => {
  const inviteLink = getInviteLinkRecord(req.params.token);
  const validation = validateInviteLink(inviteLink);
  if (!validation.ok) {
    return res.status(validation.status).json({ error: validation.error });
  }

  const alreadyCollaborator = db
    .prepare(
      `
        SELECT 1
        FROM project_collaborators
        WHERE project_id = ? AND user_id = ?
      `,
    )
    .get(inviteLink.project_id, req.user.id);

  res.json({
    inviteLink: {
      projectId: inviteLink.project_id,
      projectName: inviteLink.project_name,
      inviterUsername: inviteLink.inviter_username,
      expiresAt: inviteLink.expires_at,
      canAccept:
        req.user.id !== inviteLink.owner_id && !Boolean(alreadyCollaborator),
    },
  });
};

export const inviteProjectCollaborator = (req, res) => {
  const access = getProjectAccess(req.params.id, req.user.id);
  if (!access) return res.status(404).json({ error: 'Project not found' });
  if (!canManageProject(access)) {
    return res
      .status(403)
      .json({ error: 'Only the project owner can invite collaborators' });
  }
  if (!requireStudioForCollaboration(access, res)) {
    return;
  }

  const username = String(req.body?.username || '').trim();
  if (!username) {
    return res.status(400).json({ error: 'username is required' });
  }

  const invitee = db
    .prepare('SELECT id, username FROM users WHERE lower(username) = lower(?)')
    .get(username);

  if (!invitee) {
    return res.status(404).json({ error: 'User not found' });
  }

  if (invitee.id === access.ownerId) {
    return res.status(400).json({ error: 'Project owner already has access' });
  }

  const existingCollaborator = db
    .prepare(
      'SELECT 1 FROM project_collaborators WHERE project_id = ? AND user_id = ?',
    )
    .get(req.params.id, invitee.id);
  if (existingCollaborator) {
    return res
      .status(409)
      .json({ error: 'That user is already a collaborator' });
  }

  const existingInvite = db
    .prepare(
      `
        SELECT id FROM project_invites
        WHERE project_id = ? AND invitee_id = ? AND status = 'pending'
      `,
    )
    .get(req.params.id, invitee.id);
  if (existingInvite) {
    return res
      .status(409)
      .json({ error: 'An invite is already pending for that user' });
  }

  const inviteId = uuidv4();

  db.transaction(() => {
    db.prepare(
      `
        INSERT INTO project_invites (
          id,
          project_id,
          inviter_id,
          invitee_id,
          role,
          status
        )
        VALUES (?, ?, ?, ?, ?, 'pending')
      `,
    ).run(inviteId, req.params.id, req.user.id, invitee.id, INVITE_ROLE);

    db.prepare(
      `
        INSERT INTO notifications (id, user_id, actor_id, type, project_id)
        VALUES (?, ?, ?, 'project_invite', ?)
      `,
    ).run(uuidv4(), invitee.id, req.user.id, req.params.id);
  })();

  res.status(201).json({
    invite: {
      id: inviteId,
      projectId: req.params.id,
      projectName: access.name,
      inviteeId: invitee.id,
      inviteeUsername: invitee.username,
      role: INVITE_ROLE,
    },
  });
};

export const acceptProjectInvite = (req, res) => {
  const invite = db
    .prepare(
      `
        SELECT i.*, p.name AS project_name, p.user_id AS owner_id
        FROM project_invites i
        JOIN projects p ON p.id = i.project_id
        WHERE i.id = ?
      `,
    )
    .get(req.params.inviteId);

  if (!invite) return res.status(404).json({ error: 'Invite not found' });
  if (invite.invitee_id !== req.user.id) {
    return res.status(403).json({ error: 'Forbidden' });
  }
  if (invite.status !== 'pending') {
    return res.status(400).json({ error: 'Invite is no longer pending' });
  }

  db.transaction(() => {
    db.prepare(
      `
        UPDATE project_invites
        SET status = 'accepted', responded_at = datetime('now')
        WHERE id = ?
      `,
    ).run(invite.id);

    db.prepare(
      `
        INSERT INTO project_collaborators (project_id, user_id, role, invited_by)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(project_id, user_id) DO UPDATE SET
          role = excluded.role,
          invited_by = excluded.invited_by,
          created_at = datetime('now')
      `,
    ).run(
      invite.project_id,
      req.user.id,
      invite.role || INVITE_ROLE,
      invite.inviter_id,
    );

    db.prepare(
      `
        INSERT INTO notifications (id, user_id, actor_id, type, project_id)
        VALUES (?, ?, ?, 'project_invite_accepted', ?)
      `,
    ).run(uuidv4(), invite.inviter_id, req.user.id, invite.project_id);
  })();

  res.json({
    ok: true,
    project: {
      id: invite.project_id,
      name: invite.project_name,
    },
  });
};

export const acceptProjectInviteLink = (req, res) => {
  const inviteLink = getInviteLinkRecord(req.params.token);
  const validation = validateInviteLink(inviteLink);
  if (!validation.ok) {
    return res.status(validation.status).json({ error: validation.error });
  }

  if (req.user.id === inviteLink.owner_id) {
    return res.status(400).json({ error: 'Project owner already has access' });
  }

  const existingCollaborator = db
    .prepare(
      'SELECT 1 FROM project_collaborators WHERE project_id = ? AND user_id = ?',
    )
    .get(inviteLink.project_id, req.user.id);
  if (existingCollaborator) {
    return res
      .status(409)
      .json({ error: 'You already have access to this project' });
  }

  db.transaction(() => {
    db.prepare(
      `
        UPDATE project_invite_links
        SET status = 'accepted', claimed_by = ?, claimed_at = datetime('now')
        WHERE id = ?
      `,
    ).run(req.user.id, inviteLink.id);

    db.prepare(
      `
        INSERT INTO project_collaborators (project_id, user_id, role, invited_by)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(project_id, user_id) DO UPDATE SET
          role = excluded.role,
          invited_by = excluded.invited_by,
          created_at = datetime('now')
      `,
    ).run(
      inviteLink.project_id,
      req.user.id,
      inviteLink.role || INVITE_ROLE,
      inviteLink.inviter_id,
    );

    db.prepare(
      `
        INSERT INTO notifications (id, user_id, actor_id, type, project_id)
        VALUES (?, ?, ?, 'project_invite_accepted', ?)
      `,
    ).run(uuidv4(), inviteLink.inviter_id, req.user.id, inviteLink.project_id);
  })();

  res.json({
    ok: true,
    project: {
      id: inviteLink.project_id,
      name: inviteLink.project_name,
    },
  });
};

export const declineProjectInvite = (req, res) => {
  const invite = db
    .prepare('SELECT * FROM project_invites WHERE id = ?')
    .get(req.params.inviteId);

  if (!invite) return res.status(404).json({ error: 'Invite not found' });
  if (invite.invitee_id !== req.user.id) {
    return res.status(403).json({ error: 'Forbidden' });
  }
  if (invite.status !== 'pending') {
    return res.status(400).json({ error: 'Invite is no longer pending' });
  }

  db.prepare(
    `
      UPDATE project_invites
      SET status = 'declined', responded_at = datetime('now')
      WHERE id = ?
    `,
  ).run(invite.id);

  res.json({ ok: true });
};
