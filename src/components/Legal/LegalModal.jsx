import { useEffect, useState } from 'react';
import PropTypes from 'prop-types';
import { FileText, X } from 'lucide-react';
import {
  LEGAL_DOCUMENT_ORDER,
  getLegalDocument,
} from '../../../shared/legalDocuments.js';

const overlayStyle = {
  position: 'fixed',
  inset: 0,
  zIndex: 1200,
  background: 'rgba(15, 23, 42, 0.72)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  padding: '1.5rem',
};

const panelStyle = {
  width: 'min(960px, 100%)',
  maxHeight: '90vh',
  overflow: 'hidden',
  display: 'flex',
  flexDirection: 'column',
  background: '#fffdf8',
  borderRadius: '20px',
  boxShadow: '0 24px 80px rgba(15, 23, 42, 0.35)',
};

export default function LegalModal({ documentKey, onClose }) {
  const [activeKey, setActiveKey] = useState(documentKey);

  useEffect(() => {
    setActiveKey(documentKey);
  }, [documentKey]);

  const document = getLegalDocument(activeKey);

  return (
    <div
      style={overlayStyle}
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <div style={panelStyle}>
        <div
          style={{
            padding: '1.25rem 1.5rem 1rem',
            borderBottom: '1px solid #e7e5e4',
            background:
              'linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(15, 52, 96, 0.08))',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'flex-start',
              justifyContent: 'space-between',
              gap: '1rem',
            }}
          >
            <div>
              <div
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  color: '#8a5a00',
                  fontSize: '0.82rem',
                  fontWeight: 700,
                  letterSpacing: '0.06em',
                  textTransform: 'uppercase',
                }}
              >
                <FileText size={14} /> Draft Legal Copy
              </div>
              <h2
                style={{
                  margin: '0.55rem 0 0.35rem',
                  fontSize: '1.65rem',
                  color: '#111827',
                }}
              >
                {document.title}
              </h2>
              <p
                style={{
                  margin: 0,
                  color: '#4b5563',
                  lineHeight: 1.55,
                  maxWidth: '72ch',
                }}
              >
                {document.summary}
              </p>
              <div
                style={{
                  marginTop: '0.7rem',
                  fontSize: '0.82rem',
                  color: '#6b7280',
                }}
              >
                Version {document.version}
              </div>
            </div>
            <button
              type='button'
              onClick={onClose}
              aria-label='Close legal document'
              style={{
                border: 'none',
                background: 'rgba(255,255,255,0.82)',
                borderRadius: '999px',
                width: 38,
                height: 38,
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                color: '#374151',
              }}
            >
              <X size={18} />
            </button>
          </div>
          <div
            style={{
              display: 'flex',
              gap: '0.55rem',
              flexWrap: 'wrap',
              marginTop: '1rem',
            }}
          >
            {LEGAL_DOCUMENT_ORDER.map((key) => {
              const item = getLegalDocument(key);
              const isActive = key === activeKey;
              return (
                <button
                  key={key}
                  type='button'
                  onClick={() => setActiveKey(key)}
                  style={{
                    border: isActive
                      ? '1px solid #0f3460'
                      : '1px solid #d6d3d1',
                    background: isActive ? '#0f3460' : '#fff',
                    color: isActive ? '#fff' : '#374151',
                    borderRadius: '999px',
                    padding: '0.5rem 0.9rem',
                    fontSize: '0.88rem',
                    cursor: 'pointer',
                    fontWeight: 600,
                  }}
                >
                  {item.title}
                </button>
              );
            })}
          </div>
        </div>

        <div
          style={{
            padding: '1.25rem 1.5rem 1.5rem',
            overflowY: 'auto',
            color: '#1f2937',
          }}
        >
          {document.sections.map((section) => (
            <section key={section.title} style={{ marginBottom: '1.25rem' }}>
              <h3
                style={{
                  margin: '0 0 0.55rem',
                  fontSize: '1rem',
                  color: '#111827',
                }}
              >
                {section.title}
              </h3>
              {section.body.map((paragraph) => (
                <p
                  key={paragraph}
                  style={{
                    margin: '0 0 0.7rem',
                    lineHeight: 1.65,
                    color: '#374151',
                  }}
                >
                  {paragraph}
                </p>
              ))}
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}

LegalModal.propTypes = {
  documentKey: PropTypes.string.isRequired,
  onClose: PropTypes.func.isRequired,
};
