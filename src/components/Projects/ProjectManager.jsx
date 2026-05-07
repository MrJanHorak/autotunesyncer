import { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import PropTypes from 'prop-types';
import {
  AlertTriangle,
  CheckCircle2,
  Crown,
  Filter,
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
  Search,
  Share2,
  UserPlus,
  Users,
  X,
} from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useProject } from '../../context/ProjectContext';
import {
  DEFAULT_RENDER_PRESET,
  RENDER_PRESETS,
} from '../../../shared/renderPresets.js';
import './ProjectManager.css';

const API_BASE = 'http://localhost:3000/api';

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

const PROJECT_SCOPES = [
  { id: 'all', label: 'All projects' },
  { id: 'mine', label: 'Mine' },
  { id: 'collab', label: 'Collaborating' },
];

const PROJECT_FILTERS = [
  { id: 'all', label: 'All' },
  { id: 'shared', label: 'Published' },
  { id: 'rendered', label: 'Rendered' },
  { id: 'active', label: 'In Progress' },
];

const PROJECT_SORTS = [
  { id: 'recent', label: 'Recently updated' },
  { id: 'name', label: 'Name A-Z' },
  { id: 'clips', label: 'Most clips' },
  { id: 'progress', label: 'Most complete' },
];

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
  collaboratorCount: PropTypes.number,
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
  accessRole: PropTypes.string,
  name: PropTypes.string.isRequired,
  ownerUsername: PropTypes.string,
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

function matchesProjectFilter(project, filterMode) {
  const summary = getProjectSummary(project);

  switch (filterMode) {
    case 'shared':
      return Boolean(summary.hasSharedComposition);
    case 'rendered':
      return Boolean(summary.hasRenderOutput);
    case 'active':
      return !summary.hasSharedComposition && !summary.hasRenderOutput;
    default:
      return true;
  }
}

function matchesProjectScope(project, scopeMode) {
  const summary = getProjectSummary(project);

  switch (scopeMode) {
    case 'mine':
      return project.accessRole === 'owner';
    case 'collab':
      return (
        (project.accessRole && project.accessRole !== 'owner') ||
        (project.accessRole === 'owner' && (summary.collaboratorCount || 0) > 0)
      );
    default:
      return true;
  }
}

function compareProjects(a, b, sortMode) {
  const summaryA = getProjectSummary(a);
  const summaryB = getProjectSummary(b);

  switch (sortMode) {
    case 'name':
      return a.name.localeCompare(b.name);
    case 'clips':
      return (summaryB.clipCount || 0) - (summaryA.clipCount || 0);
    case 'progress':
      return getProjectProgress(summaryB) - getProjectProgress(summaryA);
    case 'recent':
    default:
      return (
        new Date(b.updated_at || 0).getTime() -
        new Date(a.updated_at || 0).getTime()
      );
  }
}

function ProjectThumbnailPreview({ project }) {
  const summary = getProjectSummary(project);
  const preview = project.layoutPreview;
  const renderPreset = summary.renderPreset || DEFAULT_RENDER_PRESET;
  const isPortrait = renderPreset === 'portrait';
  const stageClassName = `pm-card__mini-stage pm-card__mini-stage--${renderPreset}${summary.hasBackground ? ' pm-card__mini-stage--background' : ''}${isPortrait ? ' pm-card__mini-stage--framed' : ''}`;
  const previewGridStyle = {
    gridTemplateColumns: `repeat(${preview?.columns || 12}, minmax(0, 1fr))`,
    gridTemplateRows: `repeat(${preview?.rows || 12}, minmax(0, 1fr))`,
  };
  const previewCells = preview?.items?.map((item) => (
    <div
      key={item.id}
      className={`pm-card__mini-cell pm-card__mini-cell--${item.type || 'track'}`}
      style={{
        gridColumn: `${item.x + 1} / span ${item.w}`,
        gridRow: `${item.y + 1} / span ${item.h}`,
      }}
    />
  ));
  const stageBody = preview?.items?.length ? (
    <div className='pm-card__mini-grid' style={previewGridStyle}>
      {previewCells}
    </div>
  ) : (
    <div className='pm-card__thumb-icon pm-card__thumb-icon--empty'>
      <Film size={32} />
    </div>
  );

  if (isPortrait) {
    return (
      <div className='pm-card__mini-stage-stack'>
        {preview?.items?.length ? (
          <div
            className={`pm-card__mini-ambient${summary.hasBackground ? ' pm-card__mini-ambient--background' : ''}`}
          >
            <div
              className='pm-card__mini-grid pm-card__mini-grid--ambient'
              style={previewGridStyle}
            >
              {previewCells}
            </div>
          </div>
        ) : null}
        <div className={stageClassName}>{stageBody}</div>
      </div>
    );
  }

  return <div className={stageClassName}>{stageBody}</div>;
}

ProjectThumbnailPreview.propTypes = {
  project: projectShape.isRequired,
};

function ProjectCard({
  project,
  isSelected,
  onSelect,
  onDelete,
  deleting,
  onInvite,
}) {
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
          <div className='pm-card__actions'>
            {project.accessRole === 'owner' && onInvite && (
              <button
                type='button'
                className='pm-card__invite-btn'
                onClick={() => onInvite(project)}
              >
                <UserPlus size={14} /> Invite
              </button>
            )}
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
                  {project.accessRole === 'owner' && onInvite && (
                    <button
                      className='pm-card__dropdown-item'
                      onClick={() => {
                        onInvite(project);
                        setMenuOpen(false);
                      }}
                    >
                      <UserPlus size={15} /> Invite collaborator
                    </button>
                  )}
                  {project.accessRole === 'owner' && (
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
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {project.description && (
          <div className='pm-card__meta'>
            <Music size={14} />
            <span>{project.description}</span>
          </div>
        )}

        {project.accessRole !== 'owner' && project.ownerUsername && (
          <div className='pm-card__meta pm-card__meta--owner'>
            <Users size={14} />
            <span>Shared by @{project.ownerUsername}</span>
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
          {summary.collaboratorCount > 0 && (
            <span className='pm-card__pill'>
              <Users size={12} />
              {summary.collaboratorCount} collaborator
              {summary.collaboratorCount === 1 ? '' : 's'}
            </span>
          )}
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
  onInvite: PropTypes.func,
};

function InviteRow({ invite, onAccept, onDecline, pendingId }) {
  const isPending = pendingId === invite.id;

  return (
    <div className='pm-invite-row'>
      <div className='pm-invite-row__copy'>
        <strong>{invite.project_name}</strong>
        <span>
          @{invite.inviter_username} invited you to collaborate as an editor.
        </span>
      </div>
      <div className='pm-invite-row__actions'>
        <button
          type='button'
          className='pm-btn-primary'
          onClick={() => onAccept(invite.id)}
          disabled={isPending}
        >
          {isPending ? 'Working…' : 'Accept'}
        </button>
        <button
          type='button'
          className='pm-btn-ghost'
          onClick={() => onDecline(invite.id)}
          disabled={isPending}
        >
          Decline
        </button>
      </div>
    </div>
  );
}

InviteRow.propTypes = {
  invite: PropTypes.shape({
    id: PropTypes.string.isRequired,
    inviter_username: PropTypes.string.isRequired,
    project_name: PropTypes.string.isRequired,
  }).isRequired,
  onAccept: PropTypes.func.isRequired,
  onDecline: PropTypes.func.isRequired,
  pendingId: PropTypes.string,
};

function InviteCollaboratorModal({
  project,
  collaborators,
  pendingInvites,
  inviteUsername,
  onInviteUsernameChange,
  onClose,
  onSubmit,
  loading,
  submitting,
  error,
  success,
}) {
  if (!project) return null;

  return (
    <div className='pm-modal-overlay' onClick={onClose}>
      <div className='pm-modal' onClick={(event) => event.stopPropagation()}>
        <div className='pm-modal__header'>
          <div>
            <h3 className='pm-modal__title'>Invite collaborators</h3>
            <p className='pm-modal__subtitle'>
              Add another Symphovie user to <strong>{project.name}</strong>.
            </p>
          </div>
          <button
            type='button'
            className='pm-modal__close'
            onClick={onClose}
            aria-label='Close collaborator invite dialog'
          >
            <X size={16} />
          </button>
        </div>

        <form className='pm-modal__form' onSubmit={onSubmit}>
          <label className='pm-label' htmlFor='project-collaborator-username'>
            Invite by username
          </label>
          <input
            id='project-collaborator-username'
            className='pm-input'
            type='text'
            value={inviteUsername}
            onChange={(event) => onInviteUsernameChange(event.target.value)}
            placeholder='Enter an existing username'
            autoFocus
          />
          {error ? <p className='pm-error'>{error}</p> : null}
          {success ? <p className='pm-success'>{success}</p> : null}
          <div className='pm-create-form__actions'>
            <button
              type='submit'
              className='pm-btn-primary'
              disabled={submitting || !inviteUsername.trim()}
            >
              {submitting ? 'Sending…' : 'Send Invite'}
            </button>
            <button type='button' className='pm-btn-ghost' onClick={onClose}>
              Close
            </button>
          </div>
        </form>

        <div className='pm-modal__section'>
          <h4 className='pm-modal__section-title'>Current collaborators</h4>
          {loading ? (
            <div className='pm-modal__empty'>Loading collaborators…</div>
          ) : (
            <div className='pm-collab-list'>
              <div className='pm-collab-chip pm-collab-chip--owner'>
                <Crown size={13} /> @{project.ownerUsername || 'owner'}
              </div>
              {collaborators.map((collaborator) => (
                <div key={collaborator.id} className='pm-collab-chip'>
                  <Users size={13} /> @{collaborator.username}
                </div>
              ))}
              {collaborators.length === 0 && (
                <div className='pm-modal__empty'>No collaborators yet.</div>
              )}
            </div>
          )}
        </div>

        <div className='pm-modal__section'>
          <h4 className='pm-modal__section-title'>Pending invites</h4>
          {loading ? (
            <div className='pm-modal__empty'>Loading pending invites…</div>
          ) : pendingInvites.length > 0 ? (
            <div className='pm-pending-list'>
              {pendingInvites.map((invite) => (
                <div key={invite.id} className='pm-pending-chip'>
                  <UserPlus size={13} /> @{invite.invitee_username}
                </div>
              ))}
            </div>
          ) : (
            <div className='pm-modal__empty'>No pending invites.</div>
          )}
        </div>
      </div>
    </div>
  );
}

InviteCollaboratorModal.propTypes = {
  project: projectShape,
  collaborators: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      username: PropTypes.string.isRequired,
    }),
  ).isRequired,
  pendingInvites: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      invitee_username: PropTypes.string.isRequired,
    }),
  ).isRequired,
  inviteUsername: PropTypes.string.isRequired,
  onInviteUsernameChange: PropTypes.func.isRequired,
  onClose: PropTypes.func.isRequired,
  onSubmit: PropTypes.func.isRequired,
  loading: PropTypes.bool,
  submitting: PropTypes.bool,
  error: PropTypes.string,
  success: PropTypes.string,
};

export default function ProjectManager({ onContinue }) {
  const { user, token } = useAuth();
  const {
    projects,
    currentProject,
    loadingProjects,
    fetchProjects,
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
  const [scopeMode, setScopeMode] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [filterMode, setFilterMode] = useState('all');
  const [sortMode, setSortMode] = useState('recent');
  const [projectInvites, setProjectInvites] = useState([]);
  const [loadingInvites, setLoadingInvites] = useState(false);
  const [inviteActionPendingId, setInviteActionPendingId] = useState(null);
  const [collabError, setCollabError] = useState('');
  const [inviteProject, setInviteProject] = useState(null);
  const [inviteUsername, setInviteUsername] = useState('');
  const [inviteSubmitting, setInviteSubmitting] = useState(false);
  const [inviteError, setInviteError] = useState('');
  const [inviteSuccess, setInviteSuccess] = useState('');
  const [collaboratorSnapshot, setCollaboratorSnapshot] = useState({
    collaborators: [],
    pendingInvites: [],
  });
  const [loadingCollaborators, setLoadingCollaborators] = useState(false);

  const authFetch = useCallback(
    (path, options = {}) =>
      fetch(`${API_BASE}${path}`, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
          ...options.headers,
        },
      }),
    [token],
  );

  const fetchPendingInvites = useCallback(async () => {
    if (!token) return;
    setLoadingInvites(true);
    try {
      const res = await authFetch('/projects/invites');
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to load invites');
      setProjectInvites(data.invites || []);
    } catch (err) {
      console.warn('[projects] Failed to load project invites:', err);
    } finally {
      setLoadingInvites(false);
    }
  }, [token, authFetch]);

  const loadCollaborators = useCallback(
    async (projectId) => {
      if (!projectId || !token) return;
      setLoadingCollaborators(true);
      try {
        const res = await authFetch(`/projects/${projectId}/collaborators`);
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || 'Failed to load collaborators');
        }
        setCollaboratorSnapshot({
          collaborators: data.collaborators || [],
          pendingInvites: data.pendingInvites || [],
        });
      } catch (err) {
        setInviteError(err.message);
      } finally {
        setLoadingCollaborators(false);
      }
    },
    [token, authFetch],
  );

  useEffect(() => {
    fetchPendingInvites();
  }, [fetchPendingInvites]);

  useEffect(() => {
    if (!inviteProject?.id) {
      setCollaboratorSnapshot({ collaborators: [], pendingInvites: [] });
      setInviteError('');
      setInviteSuccess('');
      return;
    }
    void loadCollaborators(inviteProject.id);
  }, [inviteProject?.id, loadCollaborators]);

  const visibleProjects = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();

    return [...projects]
      .filter((project) => matchesProjectScope(project, scopeMode))
      .filter((project) => matchesProjectFilter(project, filterMode))
      .filter((project) => {
        if (!query) return true;

        const summary = getProjectSummary(project);
        const searchFields = [
          project.name,
          project.description,
          project.ownerUsername,
          project.accessRole === 'owner' ? 'mine' : 'collaborating',
          WORKFLOW_META[summary.workflowStage]?.label,
          summary.hasSharedComposition ? 'shared' : '',
          summary.hasRenderOutput ? 'rendered' : '',
          summary.hasMidi ? 'midi' : '',
        ]
          .filter(Boolean)
          .join(' ')
          .toLowerCase();

        return searchFields.includes(query);
      })
      .sort((a, b) => compareProjects(a, b, sortMode));
  }, [projects, filterMode, scopeMode, searchQuery, sortMode]);

  const openInviteModal = useCallback((project) => {
    setInviteProject(project);
    setInviteUsername('');
    setInviteError('');
    setInviteSuccess('');
  }, []);

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

  const handleInviteSubmit = async (event) => {
    event.preventDefault();
    if (!inviteProject?.id || !inviteUsername.trim()) return;

    setInviteSubmitting(true);
    setInviteError('');
    setInviteSuccess('');
    try {
      const res = await authFetch(`/projects/${inviteProject.id}/invites`, {
        method: 'POST',
        body: JSON.stringify({ username: inviteUsername.trim() }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to send invite');
      setInviteSuccess(`Invite sent to @${data.invite.inviteeUsername}.`);
      setInviteUsername('');
      await loadCollaborators(inviteProject.id);
      await fetchProjects();
    } catch (err) {
      setInviteError(err.message);
    } finally {
      setInviteSubmitting(false);
    }
  };

  const handleAcceptInvite = async (inviteId) => {
    setInviteActionPendingId(inviteId);
    setCollabError('');
    try {
      const res = await authFetch(`/projects/invites/${inviteId}/accept`, {
        method: 'POST',
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to accept invite');
      await Promise.all([fetchPendingInvites(), fetchProjects()]);
      setScopeMode('collab');
    } catch (err) {
      setCollabError(err.message);
    } finally {
      setInviteActionPendingId(null);
    }
  };

  const handleDeclineInvite = async (inviteId) => {
    setInviteActionPendingId(inviteId);
    setCollabError('');
    try {
      const res = await authFetch(`/projects/invites/${inviteId}/decline`, {
        method: 'POST',
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to decline invite');
      await fetchPendingInvites();
    } catch (err) {
      setCollabError(err.message);
    } finally {
      setInviteActionPendingId(null);
    }
  };

  return (
    <div className='pm-page'>
      <div className='pm-container'>
        {/* Header */}
        <div className='pm-header'>
          <div>
            <h1 className='pm-title'>Projects</h1>
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

        {collabError ? <p className='pm-error'>{collabError}</p> : null}

        {projectInvites.length > 0 && (
          <div className='pm-invites-panel'>
            <div className='pm-invites-panel__header'>
              <div>
                <h3 className='pm-invites-panel__title'>Pending invites</h3>
                <p className='pm-invites-panel__subtitle'>
                  Accept an invite to bring a shared project into your
                  workspace.
                </p>
              </div>
              <span className='pm-invites-panel__count'>
                {loadingInvites
                  ? 'Loading…'
                  : `${projectInvites.length} pending`}
              </span>
            </div>
            <div className='pm-invites-panel__list'>
              {projectInvites.map((invite) => (
                <InviteRow
                  key={invite.id}
                  invite={invite}
                  onAccept={handleAcceptInvite}
                  onDecline={handleDeclineInvite}
                  pendingId={inviteActionPendingId}
                />
              ))}
            </div>
          </div>
        )}

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

        {!loadingProjects && projects.length > 0 && (
          <div className='pm-toolbar'>
            <div className='pm-toolbar__scopes'>
              {PROJECT_SCOPES.map((scope) => (
                <button
                  key={scope.id}
                  type='button'
                  className={`pm-scope-tab${scopeMode === scope.id ? ' pm-scope-tab--active' : ''}`}
                  onClick={() => setScopeMode(scope.id)}
                >
                  {scope.label}
                </button>
              ))}
            </div>

            <div className='pm-toolbar__search'>
              <Search size={16} />
              <input
                className='pm-toolbar__search-input'
                type='search'
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder='Search projects, status, or notes'
                aria-label='Search projects'
              />
            </div>

            <div className='pm-toolbar__filters'>
              <span className='pm-toolbar__filters-label'>
                <Filter size={14} /> Filter
              </span>
              {PROJECT_FILTERS.map((filter) => (
                <button
                  key={filter.id}
                  type='button'
                  className={`pm-filter-chip${filterMode === filter.id ? ' pm-filter-chip--active' : ''}`}
                  onClick={() => setFilterMode(filter.id)}
                >
                  {filter.label}
                </button>
              ))}
            </div>

            <div className='pm-toolbar__sort'>
              <span className='pm-toolbar__count'>
                {visibleProjects.length} of {projects.length} projects
              </span>
              <select
                className='pm-toolbar__sort-select'
                value={sortMode}
                onChange={(e) => setSortMode(e.target.value)}
                aria-label='Sort projects'
              >
                {PROJECT_SORTS.map((sort) => (
                  <option key={sort.id} value={sort.id}>
                    {sort.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
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
        {!loadingProjects &&
          projects.length > 0 &&
          visibleProjects.length > 0 && (
            <div className='pm-grid'>
              {visibleProjects.map((project) => (
                <ProjectCard
                  key={project.id}
                  project={project}
                  isSelected={currentProject?.id === project.id}
                  onSelect={selectProject}
                  onDelete={handleDelete}
                  deleting={deleting}
                  onInvite={openInviteModal}
                />
              ))}
            </div>
          )}

        {!loadingProjects &&
          projects.length > 0 &&
          visibleProjects.length === 0 && (
            <div className='pm-empty pm-empty--filtered'>
              <FolderOpen className='pm-empty__icon' />
              <h3>No projects match these filters</h3>
              <p>
                Try a different search term or switch back to a broader view.
              </p>
              <button
                type='button'
                className='pm-btn-ghost'
                onClick={() => {
                  setSearchQuery('');
                  setScopeMode('all');
                  setFilterMode('all');
                  setSortMode('recent');
                }}
              >
                Reset filters
              </button>
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

        <InviteCollaboratorModal
          project={inviteProject}
          collaborators={collaboratorSnapshot.collaborators}
          pendingInvites={collaboratorSnapshot.pendingInvites}
          inviteUsername={inviteUsername}
          onInviteUsernameChange={setInviteUsername}
          onClose={() => setInviteProject(null)}
          onSubmit={handleInviteSubmit}
          loading={loadingCollaborators}
          submitting={inviteSubmitting}
          error={inviteError}
          success={inviteSuccess}
        />
      </div>
    </div>
  );
}

ProjectManager.propTypes = {
  onContinue: PropTypes.func,
};
