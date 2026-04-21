import { useState, useRef, useEffect } from 'react';
import { Film, Music, Calendar, MoreVertical, Trash2, Edit, Plus, FolderOpen } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useProject } from '../../context/ProjectContext';
import './ProjectManager.css';

function formatRelative(dateStr) {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `${mins || 1}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function ProjectCard({ project, isSelected, onSelect, onDelete, deleting }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    if (!menuOpen) return;
    const close = (e) => { if (!menuRef.current?.contains(e.target)) setMenuOpen(false); };
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, [menuOpen]);

  return (
    <div className={`pm-card${isSelected ? ' pm-card--selected' : ''}`}>
      {/* Thumbnail */}
      <div className='pm-card__thumb' onClick={() => onSelect(project)}>
        <div className='pm-card__thumb-gradient' />
        <div className='pm-card__thumb-icon'>
          <Film size={32} />
        </div>
        {isSelected && (
          <div className='pm-card__thumb-badge pm-card__thumb-badge--selected'>
            ✓ Active
          </div>
        )}
        {project.updated_at && (
          <div className='pm-card__thumb-badge'>
            <Calendar size={10} />
            {formatRelative(project.updated_at)}
          </div>
        )}
      </div>

      {/* Info */}
      <div className='pm-card__body'>
        <div className='pm-card__title-row'>
          <button className='pm-card__title' onClick={() => onSelect(project)}>
            {project.name}
          </button>
          <div className='pm-card__menu-wrap' ref={menuRef}>
            <button
              className='pm-card__menu-btn'
              onClick={() => setMenuOpen((v) => !v)}
              aria-label='Project options'
            >
              <MoreVertical size={18} />
            </button>
            {menuOpen && (
              <div className='pm-card__dropdown'>
                <button
                  className='pm-card__dropdown-item'
                  onClick={() => { onSelect(project); setMenuOpen(false); }}
                >
                  <Edit size={15} /> Open
                </button>
                <button
                  className='pm-card__dropdown-item pm-card__dropdown-item--danger'
                  onClick={() => { onDelete(project.id); setMenuOpen(false); }}
                  disabled={deleting === project.id}
                >
                  <Trash2 size={15} />
                  {deleting === project.id ? 'Deleting…' : 'Delete'}
                </button>
              </div>
            )}
          </div>
        </div>

        {project.description && (
          <div className='pm-card__meta'>
            <Music size={14} />
            <span>{project.description}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default function ProjectManager({ onContinue }) {
  const { user } = useAuth();
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

  const handleDelete = async (projectId) => {
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
    <div className='pm-page'>
      <div className='pm-container'>

        {/* Header */}
        <div className='pm-header'>
          <div>
            <h1 className='pm-title'>My Projects</h1>
            {user && <p className='pm-welcome'>Welcome back, <strong>@{user.username}</strong></p>}
          </div>
          <button
            className='pm-btn-new'
            onClick={() => setShowCreate((v) => !v)}
          >
            <Plus size={18} />
            {showCreate ? 'Cancel' : 'New Project'}
          </button>
        </div>

        {/* Create form */}
        {showCreate && (
          <form className='pm-create-form' onSubmit={handleCreate}>
            <h3 className='pm-create-form__title'>Create New Project</h3>
            <input
              className='pm-input'
              type='text'
              placeholder='Project name (e.g. Summer EP, Song 1)'
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              required
              autoFocus
            />
            <input
              className='pm-input'
              type='text'
              placeholder='Description (optional)'
              value={newDesc}
              onChange={(e) => setNewDesc(e.target.value)}
            />
            {error && <p className='pm-error'>{error}</p>}
            <div className='pm-create-form__actions'>
              <button
                type='submit'
                className='pm-btn-primary'
                disabled={creating || !newName.trim()}
              >
                {creating ? 'Creating…' : 'Create Project'}
              </button>
              <button
                type='button'
                className='pm-btn-ghost'
                onClick={() => setShowCreate(false)}
              >
                Cancel
              </button>
            </div>
          </form>
        )}

        {/* Loading */}
        {loadingProjects && (
          <div className='pm-loading'>Loading projects…</div>
        )}

        {/* Empty state */}
        {!loadingProjects && projects.length === 0 && !showCreate && (
          <div className='pm-empty'>
            <FolderOpen className='pm-empty__icon' />
            <h3>No projects yet</h3>
            <p>Create your first Symphovie project to get started</p>
            <button className='pm-btn-primary' onClick={() => setShowCreate(true)}>
              <Plus size={16} /> Create Project
            </button>
          </div>
        )}

        {/* Project grid */}
        {!loadingProjects && projects.length > 0 && (
          <div className='pm-grid'>
            {projects.map((project) => (
              <ProjectCard
                key={project.id}
                project={project}
                isSelected={currentProject?.id === project.id}
                onSelect={selectProject}
                onDelete={handleDelete}
                deleting={deleting}
              />
            ))}
          </div>
        )}

        {/* Active project continue bar */}
        {currentProject && (
          <div className='pm-continue-bar'>
            <div className='pm-continue-bar__info'>
              <Film size={18} />
              <span>Active: <strong>{currentProject.name}</strong></span>
            </div>
            <button className='pm-btn-primary' onClick={() => onContinue ? onContinue() : null}>
              Continue →
            </button>
          </div>
        )}

      </div>
    </div>
  );
}
