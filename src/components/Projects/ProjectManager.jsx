import { useState, useRef, useEffect } from 'react';
import PropTypes from 'prop-types';
import {
  AlertTriangle,
  CheckCircle2,
  Film,
  ImageIcon,
  LayoutGrid,
  LoaderCircle,
  Music,
  Calendar,
  MoreVertical,
  Trash2,
  Edit,
  Plus,
  FolderOpen,
  Share2,
} from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useProject } from '../../context/ProjectContext';
import {
  DEFAULT_RENDER_PRESET,
  RENDER_PRESETS,
} from '../../../shared/renderPresets.js';
import './ProjectManager.css';

function formatRelative(dateStr) {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `${mins || 1}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(dateStr).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
  });
}

const WORKFLOW_META = {
  draft: {
    label: 'Draft',
    description: 'Start by adding MIDI and clips.',
    tone: 'draft',
  },
  building: {
    label: 'In Progress',
    description: 'The arrangement and assets are taking shape.',
    tone: 'building',
  },
  ready: {
    label: 'Ready to Render',
    description: 'This project has the core pieces for a full render.',
    tone: 'ready',
  },
  rendering: {
    label: 'Rendering',
    description: 'A final composition is currently processing.',
    tone: 'rendering',
  },
  render_failed: {
    label: 'Render Failed',
    description: 'The last render failed and needs another pass.',
    tone: 'danger',
  },
  rendered: {
    label: 'Rendered',
    description: 'A final composition is available for this project.',
    tone: 'success',
  },
  shared: {
    label: 'Shared',
    description: 'A version of this project has been published to the feed.',
    tone: 'shared',
  },
};

const layoutPreviewShape = PropTypes.shape({
  columns: PropTypes.number,
  rows: PropTypes.number,
  items: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      x: PropTypes.number.isRequired,
      y: PropTypes.number.isRequired,
      w: PropTypes.number.isRequired,
      h: PropTypes.number.isRequired,
      type: PropTypes.string,
    }),
  ),
});

const projectSummaryShape = PropTypes.shape({
  clipCount: PropTypes.number,
  hasBackground: PropTypes.bool,
  hasMidi: PropTypes.bool,
  layoutItemCount: PropTypes.number,
  renderPreset: PropTypes.string,
  renderStatus: PropTypes.string,
  renderProgress: PropTypes.number,
  hasRenderOutput: PropTypes.bool,
  sharedCount: PropTypes.number,
  hasSharedComposition: PropTypes.bool,
  workflowStage: PropTypes.string,
  workflowProgress: PropTypes.number,
});

const projectShape = PropTypes.shape({
  id: PropTypes.string.isRequired,
  name: PropTypes.string.isRequired,
  description: PropTypes.string,
  updated_at: PropTypes.string,
  layoutPreview: layoutPreviewShape,
  summary: projectSummaryShape,
});

function getProjectSummary(project) {
  return project.summary || {};
}

function getProjectFormat(project) {
  const summary = getProjectSummary(project);
  return (
    RENDER_PRESETS[summary.renderPreset] ||
    RENDER_PRESETS[DEFAULT_RENDER_PRESET]
  );
}

function getProjectProgress(summary) {
  const renderProgress = Number(summary.renderProgress) || 0;
  if (
    summary.renderStatus === 'processing' ||
    summary.renderStatus === 'queued'
  ) {
    return Math.max(8, Math.min(100, renderProgress));
  }

  return Math.max(0, Math.min(100, Number(summary.workflowProgress) || 0));
}

function getProjectStageMeta(project) {
  const summary = getProjectSummary(project);
  const meta = WORKFLOW_META[summary.workflowStage] || WORKFLOW_META.draft;

  if (
    summary.renderStatus === 'processing' ||
    summary.renderStatus === 'queued'
  ) {
    return {
      ...meta,
      label: `Rendering ${Math.max(0, Number(summary.renderProgress) || 0)}%`,
    };
  }

  return meta;
}

function ProjectThumbnailPreview({ project }) {
  const summary = getProjectSummary(project);
  const preview = project.layoutPreview;
  const renderPreset = summary.renderPreset || DEFAULT_RENDER_PRESET;

  return (
    <div
      className={`pm-card__mini-stage pm-card__mini-stage--${renderPreset}${summary.hasBackground ? ' pm-card__mini-stage--background' : ''}`}
    >
      {preview?.items?.length ? (
        <div
          className='pm-card__mini-grid'
          style={{
            gridTemplateColumns: `repeat(${preview.columns || 12}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${preview.rows || 12}, minmax(0, 1fr))`,
          }}
        >
          {preview.items.map((item) => (
            <div
              key={item.id}
              className={`pm-card__mini-cell pm-card__mini-cell--${item.type || 'track'}`}
              style={{
                gridColumn: `${item.x + 1} / span ${item.w}`,
                gridRow: `${item.y + 1} / span ${item.h}`,
              }}
            />
          ))}
        </div>
      ) : (
        <div className='pm-card__thumb-icon pm-card__thumb-icon--empty'>
          <Film size={32} />
        </div>
      )}
    </div>
  );
}

ProjectThumbnailPreview.propTypes = {
  project: projectShape.isRequired,
};

function ProjectCard({ project, isSelected, onSelect, onDelete, deleting }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef(null);
  const summary = getProjectSummary(project);
  const stageMeta = getProjectStageMeta(project);
  const progress = getProjectProgress(summary);
  const format = getProjectFormat(project);

  useEffect(() => {
    if (!menuOpen) return;
    const close = (e) => {
      if (!menuRef.current?.contains(e.target)) setMenuOpen(false);
    };
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, [menuOpen]);

  return (
    <div className={`pm-card${isSelected ? ' pm-card--selected' : ''}`}>
      {/* Thumbnail */}
      <div className='pm-card__thumb' onClick={() => onSelect(project)}>
        <div className='pm-card__thumb-gradient' />
        <ProjectThumbnailPreview project={project} />
        {isSelected && (
          <div className='pm-card__thumb-badge pm-card__thumb-badge--selected'>
            ✓ Active
          </div>
        )}
        <div
          className={`pm-card__thumb-status pm-card__thumb-status--${stageMeta.tone}`}
        >
          {stageMeta.label}
        </div>
        <div className='pm-card__thumb-format'>{format.shortLabel}</div>
        {project.updated_at && (
          <div className='pm-card__thumb-badge pm-card__thumb-badge--time'>
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
                  onClick={() => {
                    onSelect(project);
                    setMenuOpen(false);
                  }}
                >
                  <Edit size={15} /> Open
                </button>
                <button
                  className='pm-card__dropdown-item pm-card__dropdown-item--danger'
                  onClick={() => {
                    onDelete(project.id);
                    setMenuOpen(false);
                  }}
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

        <div className='pm-card__status-copy'>{stageMeta.description}</div>

        <div className='pm-card__progress-row'>
          <span>Project stage</span>
          <span>{progress}%</span>
        </div>
        <div className='pm-card__progress'>
          <span style={{ width: `${progress}%` }} />
        </div>

        <div className='pm-card__stats'>
          <div className='pm-card__stat'>
            <Film size={13} />
            <span>{summary.clipCount || 0} clips</span>
          </div>
          <div className='pm-card__stat'>
            <LayoutGrid size={13} />
            <span>{summary.layoutItemCount || 0} tiles</span>
          </div>
          <div className='pm-card__stat'>
            <Music size={13} />
            <span>{summary.hasMidi ? 'MIDI loaded' : 'No MIDI'}</span>
          </div>
          <div className='pm-card__stat'>
            <Share2 size={13} />
            <span>
              {summary.hasSharedComposition
                ? `${summary.sharedCount} shared`
                : 'Not shared'}
            </span>
          </div>
        </div>

        <div className='pm-card__footer'>
          {summary.hasBackground && (
            <span className='pm-card__pill'>
              <ImageIcon size={12} /> Background
            </span>
          )}
          {summary.hasRenderOutput && (
            <span className='pm-card__pill pm-card__pill--success'>
              <CheckCircle2 size={12} /> Final render
            </span>
          )}
          {(summary.renderStatus === 'processing' ||
            summary.renderStatus === 'queued') && (
            <span className='pm-card__pill pm-card__pill--info'>
              <LoaderCircle size={12} /> In queue
            </span>
          )}
          {summary.renderStatus === 'failed' && (
            <span className='pm-card__pill pm-card__pill--danger'>
              <AlertTriangle size={12} /> Retry render
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

ProjectCard.propTypes = {
  project: projectShape.isRequired,
  isSelected: PropTypes.bool,
  onSelect: PropTypes.func.isRequired,
  onDelete: PropTypes.func.isRequired,
  deleting: PropTypes.string,
};

export default function ProjectManager({ onContinue }) {
  const { user } = useAuth();
  const {
    projects,
    currentProject,
    loadingProjects,
    selectProject,
    createProject,
    deleteProject,
  } = useProject();

  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [newRenderPreset, setNewRenderPreset] = useState(DEFAULT_RENDER_PRESET);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState('');
  const [deleting, setDeleting] = useState(null);

  const handleCreate = async (e) => {
    e.preventDefault();
    if (!newName.trim()) return;
    setCreating(true);
    setError('');
    try {
      await createProject(newName.trim(), newDesc.trim(), newRenderPreset);
      setNewName('');
      setNewDesc('');
      setNewRenderPreset(DEFAULT_RENDER_PRESET);
      setShowCreate(false);
    } catch (err) {
      setError(err.message);
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (projectId) => {
    if (
      !window.confirm(
        'Delete this project and all its uploaded clips? This cannot be undone.',
      )
    )
      return;
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
            {user && (
              <p className='pm-welcome'>
                Welcome back, <strong>@{user.username}</strong>
              </p>
            )}
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
            <label className='pm-label' htmlFor='project-render-preset'>
              Video Format
            </label>
            <select
              id='project-render-preset'
              className='pm-input'
              value={newRenderPreset}
              onChange={(e) => setNewRenderPreset(e.target.value)}
            >
              {Object.values(RENDER_PRESETS).map((preset) => (
                <option key={preset.id} value={preset.id}>
                  {preset.label} - {preset.description}
                </option>
              ))}
            </select>
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
        {loadingProjects && <div className='pm-loading'>Loading projects…</div>}

        {/* Empty state */}
        {!loadingProjects && projects.length === 0 && !showCreate && (
          <div className='pm-empty'>
            <FolderOpen className='pm-empty__icon' />
            <h3>No projects yet</h3>
            <p>Create your first Symphovie project to get started</p>
            <button
              className='pm-btn-primary'
              onClick={() => setShowCreate(true)}
            >
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
              <span>
                Active: <strong>{currentProject.name}</strong>
              </span>
            </div>
            <button
              className='pm-btn-primary'
              onClick={() => (onContinue ? onContinue() : null)}
            >
              Continue →
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

ProjectManager.propTypes = {
  onContinue: PropTypes.func,
};
