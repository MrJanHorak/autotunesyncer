import { useEffect, useRef, useState } from 'react';
import { io } from 'socket.io-client';

const SOCKET_URL = 'http://localhost:3000';

export function useProjectRealtime({
  token,
  currentProjectId,
  onRemoteStateSaved,
}) {
  const socketRef = useRef(null);
  const activeProjectRef = useRef(null);
  const [presenceUsers, setPresenceUsers] = useState([]);
  const [connectionState, setConnectionState] = useState('disconnected');

  useEffect(() => {
    if (!token) {
      socketRef.current?.disconnect();
      socketRef.current = null;
      activeProjectRef.current = null;
      setPresenceUsers([]);
      setConnectionState('disconnected');
      return undefined;
    }

    const socket = io(SOCKET_URL, {
      auth: { token },
    });

    socketRef.current = socket;
    setConnectionState('connecting');

    socket.on('connect', () => {
      setConnectionState('connected');
    });

    socket.on('connect_error', () => {
      activeProjectRef.current = null;
      setConnectionState('disconnected');
      setPresenceUsers([]);
    });

    socket.on('disconnect', () => {
      activeProjectRef.current = null;
      setConnectionState('disconnected');
      setPresenceUsers([]);
    });

    socket.on('project:presence', (payload) => {
      if (
        !payload?.projectId ||
        payload.projectId !== activeProjectRef.current
      ) {
        return;
      }

      setPresenceUsers(payload.users || []);
    });

    return () => {
      activeProjectRef.current = null;
      socket.disconnect();
      if (socketRef.current === socket) {
        socketRef.current = null;
      }
      setPresenceUsers([]);
      setConnectionState('disconnected');
    };
  }, [token]);

  useEffect(() => {
    const socket = socketRef.current;
    const projectId = currentProjectId || null;

    if (!socket) {
      setPresenceUsers([]);
      return undefined;
    }

    const handleRemoteStateSaved = (payload) => {
      if (!payload?.projectId || payload.projectId !== projectId) {
        return;
      }

      onRemoteStateSaved?.(payload);
    };

    socket.on('project:state-saved', handleRemoteStateSaved);

    if (!projectId) {
      if (activeProjectRef.current) {
        socket.emit('project:leave', {
          projectId: activeProjectRef.current,
        });
      }
      activeProjectRef.current = null;
      setPresenceUsers([]);
      return () => {
        socket.off('project:state-saved', handleRemoteStateSaved);
      };
    }

    const joinRoom = () => {
      socket.emit('project:join', { projectId }, (response) => {
        if (!response?.ok) {
          if (activeProjectRef.current === projectId) {
            activeProjectRef.current = null;
          }
          setPresenceUsers([]);
          return;
        }

        activeProjectRef.current = projectId;
        setPresenceUsers(response.users || []);
      });
    };

    if (!socket.connected) {
      setConnectionState('connecting');
      setPresenceUsers([]);
      socket.once('connect', joinRoom);
      return () => {
        socket.off('project:state-saved', handleRemoteStateSaved);
        socket.off('connect', joinRoom);
      };
    }

    if (activeProjectRef.current && activeProjectRef.current !== projectId) {
      socket.emit('project:leave', { projectId: activeProjectRef.current });
      activeProjectRef.current = null;
    }

    joinRoom();

    return () => {
      socket.off('project:state-saved', handleRemoteStateSaved);
      socket.off('connect', joinRoom);
      if (socketRef.current !== socket) return;

      if (activeProjectRef.current === projectId) {
        socket.emit('project:leave', { projectId });
        activeProjectRef.current = null;
      }
      setPresenceUsers([]);
    };
  }, [currentProjectId, onRemoteStateSaved]);

  return {
    connectionState,
    presenceUsers,
  };
}
