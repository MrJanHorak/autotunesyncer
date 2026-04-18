import { useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import { useProject } from '../../context/ProjectContext';
import './ProjectManager.css';

export default function ProjectManager() {
  const { user, logout } = useAuth();
  const { projects, currentProject, loadingProjects, selectProject, createProject, deleteProject } = useProject();

  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState('');
  const [deleting, setDeleting] = useState(null);

  const handleCreate = async (e) => {
    e.preventDefault();
    if (!newName.trim()) return;
    setCreating(true);
    setError('');
    try {
      await createProject(newName.trim(), newDesc.trim());
      setNewName('');
      setNewDesc('');
      setShowCreate(false);
    } catch (err) {
      setError(err.message);
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (projectId, e) => {
    e.stopPropagation();
    if (!window.confirm('Delete this project and all its uploaded clips? This cannot be undone.')) return;
    setDeleting(projectId);
    try {
      await deleteProject(projectId);
    } catch (err) {
      setError(err.message);
    } finally {
      setDeleting(null);
    }
  };

  return (
    <div className="pm-page">
      <div className="pm-container">
        <header className="pm-header">
          <div>
            <h1 className="pm-title">🎵 AutoTuneSyncer</h1>
            <p className="pm-welcome">Welcome, <strong>{user?.username}</strong></p>
          </div>
          <button className="pm-logout" onClick={logout}>Sign Out</button>
        </header>

        <section className="pm-section">
          <div className="pm-section-header">
            <h2>Your Projects</h2>
            <button className="pm-btn-primary" onClick={() => setShowCreate((v) => !v)}>
              {showCreate ? 'Cancel' : '+ New Project'}
            </button>
          </div>

          {showCreate && (
            <form className="pm-create-form" onSubmit={handleCreate}>
              <input
                type="text"
                placeholder="Project name (e.g. Summer EP, Song 1)"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                required
                autoFocus
              />
              <input
                type="text"
                placeholder="Description (optional)"
                value={newDesc}
                onChange={(e) => setNewDesc(e.target.value)}
              />
              {error && <p className="pm-error">{error}</p>}
              <button type="submit" className="pm-btn-primary" disabled={creating || !newName.trim()}>
                {creating ? 'Creating…' : 'Create Project'}
              </button>
            </form>
          )}

          {loadingProjects && <p className="pm-loading">Loading projects…</p>}

          {!loadingProjects && projects.length === 0 && !showCreate && (
            <div className="pm-empty">
              <p>You have no projects yet.</p>
              <p>Create your first project to start uploading clips!</p>
            </div>
          )}

          <div className="pm-grid">
            {projects.map((project) => (
              <div
                key={project.id}
                className={`pm-card ${currentProject?.id === project.id ? 'selected' : ''}`}
                onClick={() => selectProject(project)}
              >
                <div className="pm-card-body">
                  <h3 className="pm-card-title">{project.name}</h3>
                  {project.description && (
                    <p className="pm-card-desc">{project.description}</p>
                  )}
                  <p className="pm-card-date">
                    Updated {new Date(project.updated_at).toLocaleDateString()}
                  </p>
                </div>
                <div className="pm-card-actions">
                  <button
                    className="pm-btn-select"
                    onClick={() => selectProject(project)}
                  >
                    {currentProject?.id === project.id ? '✓ Selected' : 'Open'}
                  </button>
                  <button
                    className="pm-btn-delete"
                    onClick={(e) => handleDelete(project.id, e)}
                    disabled={deleting === project.id}
                    title="Delete project"
                  >
                    {deleting === project.id ? '…' : '🗑'}
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>

        {currentProject && (
          <div className="pm-continue-bar">
            <span>Working on: <strong>{currentProject.name}</strong></span>
            <button className="pm-btn-primary" onClick={() => selectProject(currentProject)}>
              Continue →
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
