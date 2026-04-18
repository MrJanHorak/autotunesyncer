import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useAuth } from './AuthContext';

const ProjectContext = createContext(null);

const API_BASE = 'http://localhost:3000/api';

export function ProjectProvider({ children }) {
  const { token } = useAuth();
  const [projects, setProjects] = useState([]);
  const [currentProject, setCurrentProject] = useState(() => {
    try {
      const stored = localStorage.getItem('current_project');
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  });
  const [loadingProjects, setLoadingProjects] = useState(false);

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
    [token]
  );

  const fetchProjects = useCallback(async () => {
    if (!token) return;
    setLoadingProjects(true);
    try {
      const res = await authFetch('/projects');
      const data = await res.json();
      if (res.ok) setProjects(data.projects || []);
    } catch (err) {
      console.error('Failed to fetch projects:', err);
    } finally {
      setLoadingProjects(false);
    }
  }, [token, authFetch]);

  // Load projects whenever token changes
  useEffect(() => {
    if (token) fetchProjects();
    else {
      setProjects([]);
      setCurrentProject(null);
      localStorage.removeItem('current_project');
    }
  }, [token, fetchProjects]);

  const selectProject = useCallback((project) => {
    setCurrentProject(project);
    if (project) {
      localStorage.setItem('current_project', JSON.stringify(project));
    } else {
      localStorage.removeItem('current_project');
    }
  }, []);

  const createProject = useCallback(
    async (name, description = '') => {
      const res = await authFetch('/projects', {
        method: 'POST',
        body: JSON.stringify({ name, description }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to create project');
      setProjects((prev) => [data.project, ...prev]);
      selectProject(data.project);
      return data.project;
    },
    [authFetch, selectProject]
  );

  const deleteProject = useCallback(
    async (projectId) => {
      const res = await authFetch(`/projects/${projectId}`, { method: 'DELETE' });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || 'Failed to delete project');
      }
      setProjects((prev) => prev.filter((p) => p.id !== projectId));
      if (currentProject?.id === projectId) selectProject(null);
    },
    [authFetch, currentProject, selectProject]
  );

  const saveProjectState = useCallback(
    async (state) => {
      if (!currentProject) return;
      const res = await authFetch(`/projects/${currentProject.id}/state`, {
        method: 'POST',
        body: JSON.stringify(state),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || 'Failed to save state');
      }
    },
    [authFetch, currentProject]
  );

  const loadProjectState = useCallback(
    async (projectId) => {
      const id = projectId ?? currentProject?.id;
      if (!id) return null;
      const res = await authFetch(`/projects/${id}/state`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to load state');
      return data.state;
    },
    [authFetch, currentProject]
  );

  return (
    <ProjectContext.Provider
      value={{
        projects,
        currentProject,
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
