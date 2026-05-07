import { v4 as uuidv4 } from 'uuid';
import db from '../db/database.js';
import {
  canManageProject,
  getProjectAccess,
} from '../services/projectAccessService.js';

const INVITE_ROLE = 'editor';

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

  res.json({
    owner: {
      id: access.ownerId,
      username: access.ownerUsername,
      role: 'owner',
    },
    collaborators,
    pendingInvites,
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
