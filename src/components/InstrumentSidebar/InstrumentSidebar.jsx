import PropTypes from 'prop-types';
import './InstrumentSidebar.css';

const toClipKey = (instrument) => {
  if (instrument.isDrum) {
    return `drum_${(instrument.group || '').toLowerCase().replace(/\s+/g, '_')}`;
  }
  return (instrument.name || '').toLowerCase().replace(/\s+/g, '_');
};

const familyEmoji = (instrument) => {
  if (instrument.isDrum) return '🥁';
  const fam = (instrument.family || '').toLowerCase();
  if (fam.includes('piano') || fam.includes('organ') || fam.includes('keyboard')) return '🎹';
  if (fam.includes('guitar') || fam.includes('bass')) return '🎸';
  if (fam.includes('string')) return '🎻';
  if (fam.includes('brass')) return '🎺';
  if (fam.includes('reed') || fam.includes('wind') || fam.includes('saxophone') || fam.includes('harmonica')) return '🎷';
  if (fam.includes('flute') || fam.includes('pipe')) return '🪈';
  if (fam.includes('vocal') || fam.includes('choir')) return '🎤';
  return '🎵';
};

const displayName = (instrument) => {
  if (instrument.isDrum) {
    const g = instrument.group || '';
    return 'Drum – ' + g.charAt(0).toUpperCase() + g.slice(1);
  }
  return instrument.name || 'Unknown';
};

export default function InstrumentSidebar({
  instruments,
  instrumentVideos,
  longestNotes,
  onRecordClick,
  isOpen,
  onToggle,
}) {
  const recorded = instruments.filter((i) => !!instrumentVideos?.[toClipKey(i)]).length;
  const total = instruments.length;
  const pct = total > 0 ? Math.round((recorded / total) * 100) : 0;

  return (
    <div className={`instrument-sidebar${isOpen ? '' : ' instrument-sidebar--collapsed'}`}>
      {/* Panel header */}
      <div className='instrument-sidebar__header'>
        {isOpen && <span className='instrument-sidebar__title'>Instruments</span>}
        {isOpen && (
          <span className='instrument-sidebar__count'>
            {recorded}/{total}
          </span>
        )}
        <button
          className='panel-toggle-btn instrument-sidebar__toggle'
          onClick={onToggle}
          title={isOpen ? 'Collapse instrument panel' : 'Expand instrument panel'}
        >
          {isOpen ? '◀' : '▶'}
        </button>
      </div>

      {/* Progress strip (full width when open, tiny dot when collapsed) */}
      {isOpen && total > 0 && (
        <div className='instrument-sidebar__progress-bar'>
          <div
            className='instrument-sidebar__progress-fill'
            style={{ width: `${pct}%` }}
          />
          <span className='instrument-sidebar__progress-label'>{pct}% ready</span>
        </div>
      )}

      {/* Instrument list */}
      <div className='instrument-sidebar__list'>
        {instruments.map((instrument, idx) => {
          const clipKey = toClipKey(instrument);
          const hasVideo = !!instrumentVideos?.[clipKey];
          const emoji = familyEmoji(instrument);
          const name = displayName(instrument);
          const instrKey = instrument.isDrum
            ? `drum_${instrument.group}`
            : instrument.name;
          const minSec = longestNotes?.[instrKey] || 0;
          const recSec = Math.ceil(minSec + 1);

          return (
            <button
              key={idx}
              className={`instrument-sidebar__item${hasVideo ? ' instrument-sidebar__item--recorded' : ''}`}
              onClick={() => onRecordClick(instrument)}
              title={isOpen ? undefined : name}
            >
              <span className='instrument-sidebar__status'>
                {hasVideo ? '✅' : '⏺'}
              </span>
              {isOpen && (
                <>
                  <span className='instrument-sidebar__emoji'>{emoji}</span>
                  <span className='instrument-sidebar__name'>{name}</span>
                  <span className='instrument-sidebar__hint'>{recSec}s</span>
                </>
              )}
            </button>
          );
        })}

        {instruments.length === 0 && isOpen && (
          <p className='instrument-sidebar__empty'>Load a MIDI file to see instruments</p>
        )}
      </div>
    </div>
  );
}

InstrumentSidebar.propTypes = {
  instruments: PropTypes.array.isRequired,
  instrumentVideos: PropTypes.object,
  longestNotes: PropTypes.object,
  onRecordClick: PropTypes.func.isRequired,
  isOpen: PropTypes.bool.isRequired,
  onToggle: PropTypes.func.isRequired,
};
