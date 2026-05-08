import jwt from 'jsonwebtoken';
import { Server } from 'socket.io';
import { JWT_SECRET } from '../middleware/auth.js';
import { canReadProject, getProjectAccess } from './projectAccessService.js';

const PROJECT_ROOM_PREFIX = 'project:';

let io = null;
const projectPresence = new Map();

function getProjectRoom(projectId) {
  return `${PROJECT_ROOM_PREFIX}${projectId}`;
}

function getOrCreatePresenceMap(projectId) {
  if (!projectPresence.has(projectId)) {
    projectPresence.set(projectId, new Map());
  }
  return projectPresence.get(projectId);
}

function buildPresenceSnapshot(projectId) {
  const socketPresence = projectPresence.get(projectId);
  if (!socketPresence?.size) return [];

  const uniqueUsers = new Map();
  for (const presence of socketPresence.values()) {
    const existing = uniqueUsers.get(presence.id);
    if (existing) {
      existing.connectionCount += 1;
      continue;
    }

    uniqueUsers.set(presence.id, {
      id: presence.id,
      username: presence.username,
      accessRole: presence.accessRole,
      connectionCount: 1,
    });
  }

  return [...uniqueUsers.values()].sort((left, right) =>
    left.username.localeCompare(right.username),
  );
}

function emitProjectPresence(projectId) {
  if (!io || !projectId) return;
  io.to(getProjectRoom(projectId)).emit('project:presence', {
    projectId,
    users: buildPresenceSnapshot(projectId),
  });
}

function removeSocketFromProject(socket) {
  const activeProjectId = socket.data.projectId;
  if (!activeProjectId) return;

  const socketPresence = projectPresence.get(activeProjectId);
  if (socketPresence) {
    socketPresence.delete(socket.id);
    if (socketPresence.size === 0) {
      projectPresence.delete(activeProjectId);
    }
  }

  socket.leave(getProjectRoom(activeProjectId));
  socket.data.projectId = null;
  emitProjectPresence(activeProjectId);
}

function readSocketToken(socket) {
  const authToken = socket.handshake.auth?.token;
  if (authToken) return authToken;

  const authHeader = socket.handshake.headers?.authorization;
  return authHeader?.startsWith('Bearer ')
    ? authHeader.slice(7)
    : authHeader || null;
}

export function initRealtime(server, { allowedOrigins = [] } = {}) {
  if (io) return io;

  io = new Server(server, {
    cors: {
      origin: allowedOrigins.length > 0 ? allowedOrigins : true,
      methods: ['GET', 'POST'],
      allowedHeaders: ['Authorization'],
      credentials: true,
    },
  });

  io.use((socket, next) => {
    const token = readSocketToken(socket);
    if (!token) {
      next(new Error('Authentication required'));
      return;
    }

    try {
      const payload = jwt.verify(token, JWT_SECRET);
      socket.data.user = {
        id: payload.id,
        username: payload.username,
        email: payload.email,
      };
      next();
    } catch {
      next(new Error('Invalid or expired token'));
    }
  });

  io.on('connection', (socket) => {
    socket.data.projectId = null;

    socket.on('project:join', ({ projectId } = {}, ack = () => {}) => {
      if (!projectId) {
        ack({ ok: false, error: 'projectId is required' });
        return;
      }

      const projectAccess = getProjectAccess(projectId, socket.data.user.id);
      if (!canReadProject(projectAccess)) {
        ack({ ok: false, error: 'Project access required' });
        return;
      }

      if (socket.data.projectId && socket.data.projectId !== projectId) {
        removeSocketFromProject(socket);
      }

      socket.join(getProjectRoom(projectId));
      socket.data.projectId = projectId;
      getOrCreatePresenceMap(projectId).set(socket.id, {
        id: socket.data.user.id,
        username: socket.data.user.username,
        accessRole: projectAccess.accessRole,
      });

      const users = buildPresenceSnapshot(projectId);
      emitProjectPresence(projectId);
      ack({ ok: true, users });
    });

    socket.on('project:leave', ({ projectId } = {}, ack = () => {}) => {
      if (!socket.data.projectId) {
        ack({ ok: true });
        return;
      }

      if (projectId && projectId !== socket.data.projectId) {
        ack({ ok: true });
        return;
      }

      removeSocketFromProject(socket);
      ack({ ok: true });
    });

    socket.on('disconnect', () => {
      removeSocketFromProject(socket);
    });
  });

  return io;
}

export function emitProjectStateSaved(projectId, payload) {
  if (!io || !projectId) return;

  io.to(getProjectRoom(projectId)).emit('project:state-saved', {
    projectId,
    ...payload,
  });
}
