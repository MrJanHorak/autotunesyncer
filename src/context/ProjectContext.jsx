import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
} from 'react';
import { useAuth } from './AuthContext';
import {
  DEFAULT_RENDER_PRESET,
  normalizeRenderPreset,
} from '../../shared/renderPresets.js';

const ProjectContext = createContext(null);

const API_BASE = 'http://localhost:3000/api';

function readStoredProject() {
  try {
    const stored = localStorage.getItem('current_project');
    return stored ? JSON.parse(stored) : null;
  } catch {
    return null;
  }
}

export function ProjectProvider({ children }) {
  const { token } = useAuth();
  const [projects, setProjects] = useState([]);
  const [realtimeClientId] = useState(() => {
    if (globalThis.crypto?.randomUUID) {
      return globalThis.crypto.randomUUID();
    }

    return `rtc-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  });
  const [currentProject, setCurrentProject] = useState(() =>
    readStoredProject(),
  );
  const [currentProjectStateVersion, setCurrentProjectStateVersion] = useState(
    () => Number(readStoredProject()?.stateVersion) || 1,
  );
  const [loadingProjects, setLoadingProjects] = useState(false);
  const currentProjectRef = useRef(readStoredProject());
  const currentProjectStateVersionRef = useRef(
    Number(readStoredProject()?.stateVersion) || 1,
  );
  const saveQueueRef = useRef(Promise.resolve());

  useEffect(() => {
    currentProjectRef.current = currentProject;
  }, [currentProject]);

  useEffect(() => {
    currentProjectStateVersionRef.current = currentProjectStateVersion;
  }, [currentProjectStateVersion]);

  const persistCurrentProject = useCallback((project) => {
    if (project) {
      localStorage.setItem('current_project', JSON.stringify(project));
    } else {
      localStorage.removeItem('current_project');
    }
  }, []);

  const authFetch = useCallback(
    (path, options = {}) => {
      return fetch(`${API_BASE}${path}`, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
          ...options.headers,
        },
      });
    },
    [token],
  );

  const fetchProjects = useCallback(async () => {
    if (!token) return;
    setLoadingProjects(true);
    try {
      const res = await authFetch('/projects');
      const data = await res.json();
      if (res.ok) {
        const nextProjects = data.projects || [];
        setProjects(nextProjects);

        if (currentProject?.id) {
          const refreshedCurrentProject = nextProjects.find(
            (project) => project.id === currentProject.id,
          );

          if (refreshedCurrentProject) {
            currentProjectRef.current = refreshedCurrentProject;
            currentProjectStateVersionRef.current =
              Number(refreshedCurrentProject.stateVersion) || 1;
            setCurrentProject(refreshedCurrentProject);
            setCurrentProjectStateVersion(
              Number(refreshedCurrentProject.stateVersion) || 1,
            );
            persistCurrentProject(refreshedCurrentProject);
          }
        }
      }
    } catch (err) {
      console.error('Failed to fetch projects:', err);
    } finally {
      setLoadingProjects(false);
    }
  }, [token, authFetch, currentProject?.id, persistCurrentProject]);

  // Load projects whenever token changes
  useEffect(() => {
    if (token) fetchProjects();
    else {
      setProjects([]);
      setCurrentProject(null);
      setCurrentProjectStateVersion(1);
      currentProjectRef.current = null;
      currentProjectStateVersionRef.current = 1;
      localStorage.removeItem('current_project');
    }
  }, [token, fetchProjects]);

  const selectProject = useCallback(
    (project) => {
      const nextProject = project
        ? {
            ...project,
            stateVersion: Number(project.stateVersion) || 1,
          }
        : null;

      currentProjectRef.current = nextProject;
      currentProjectStateVersionRef.current =
        Number(nextProject?.stateVersion) || 1;
      setCurrentProject(nextProject);
      setCurrentProjectStateVersion(Number(nextProject?.stateVersion) || 1);
      persistCurrentProject(nextProject);
    },
    [persistCurrentProject],
  );

  const updateCurrentProjectSnapshot = useCallback(
    (patch, projectId = null) => {
      setCurrentProject((prev) => {
        if (!prev) return prev;
        if (projectId && prev.id !== projectId) return prev;
        const nextProject = {
          ...prev,
          ...patch,
          stateVersion: Number(patch?.stateVersion ?? prev.stateVersion) || 1,
        };
        currentProjectRef.current = nextProject;
        persistCurrentProject(nextProject);
        return nextProject;
      });

      if (patch?.stateVersion !== undefined) {
        const nextVersion = Number(patch.stateVersion) || 1;
        currentProjectStateVersionRef.current = nextVersion;
        setCurrentProjectStateVersion(nextVersion);
      }
    },
    [persistCurrentProject],
  );

  const createProject = useCallback(
    async (name, description = '', renderPreset = DEFAULT_RENDER_PRESET) => {
      const res = await authFetch('/projects', {
        method: 'POST',
        body: JSON.stringify({
          name,
          description,
          renderPreset: normalizeRenderPreset(renderPreset),
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to create project');
      setProjects((prev) => [data.project, ...prev]);
      selectProject(data.project);
      return data.project;
    },
    [authFetch, selectProject],
  );

  const deleteProject = useCallback(
    async (projectId) => {
      const res = await authFetch(`/projects/${projectId}`, {
        method: 'DELETE',
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || 'Failed to delete project');
      }
      setProjects((prev) => prev.filter((p) => p.id !== projectId));
      if (currentProject?.id === projectId) selectProject(null);
    },
    [authFetch, currentProject, selectProject],
  );

  const saveProjectState = useCallback(
    (state, options = {}) => {
      if (!currentProject) return Promise.resolve();

      const projectId = currentProject.id;
      const fallbackUpdatedAt = currentProject.updated_at;

      const queuedSave = saveQueueRef.current
        .catch(() => undefined)
        .then(async () => {
          const baseStateVersion =
            options.baseStateVersion ?? currentProjectStateVersionRef.current;
          const res = await authFetch(`/projects/${projectId}/state`, {
            method: 'POST',
            body: JSON.stringify({
              ...state,
              baseStateVersion,
              realtimeClientId,
            }),
          });
          const data = await res.json();
          if (!res.ok) {
            if (res.status === 409) {
              const conflictError = new Error(
                data.error || 'Project state has changed since you loaded it',
              );
              conflictError.code = 'PROJECT_CONFLICT';
              conflictError.currentState = data.currentState || null;
              conflictError.currentStateVersion =
                Number(data.currentStateVersion) ||
                currentProjectStateVersionRef.current;
              throw conflictError;
            }

            throw new Error(data.error || 'Failed to save state');
          }

          const nextVersion = Number(data.stateVersion) || baseStateVersion;
          if (currentProjectRef.current?.id === projectId) {
            updateCurrentProjectSnapshot(
              {
                stateVersion: nextVersion,
                updated_at: data.updatedAt || fallbackUpdatedAt,
              },
              projectId,
            );
          }
        });

      saveQueueRef.current = queuedSave;
      return queuedSave;
    },
    [authFetch, currentProject, realtimeClientId, updateCurrentProjectSnapshot],
  );

  const loadProjectState = useCallback(
    async (projectId) => {
      const id = projectId ?? currentProject?.id;
      if (!id) return null;
      const res = await authFetch(`/projects/${id}/state`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to load state');

      if (id === currentProject?.id) {
        updateCurrentProjectSnapshot(
          {
            stateVersion:
              Number(data.stateVersion) || currentProjectStateVersion,
            updated_at: data.updatedAt || currentProject?.updated_at,
          },
          id,
        );
      }

      return data.state;
    },
    [
      authFetch,
      currentProject?.id,
      currentProject?.updated_at,
      currentProjectStateVersion,
      updateCurrentProjectSnapshot,
    ],
  );

  return (
    <ProjectContext.Provider
      value={{
        projects,
        currentProject,
        currentProjectStateVersion,
        realtimeClientId,
        loadingProjects,
        fetchProjects,
        selectProject,
        createProject,
        deleteProject,
        saveProjectState,
        loadProjectState,
      }}
    >
      {children}
    </ProjectContext.Provider>
  );
}

export const useProject = () => {
  const ctx = useContext(ProjectContext);
  if (!ctx) throw new Error('useProject must be used inside ProjectProvider');
  return ctx;
};
