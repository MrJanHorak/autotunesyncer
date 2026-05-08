import React, { useState } from 'react';
import LegalModal from './Legal/LegalModal.jsx';

export default function LandingPage({ onLogin }) {
  const [activeLegalDocument, setActiveLegalDocument] = useState(null);

  return (
    <div
      className='landing-page'
      style={{
        minHeight: '100vh',
        background: '#10101a',
        color: '#fff',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      {activeLegalDocument && (
        <LegalModal
          documentKey={activeLegalDocument}
          onClose={() => setActiveLegalDocument(null)}
        />
      )}
      <h1
        style={{
          fontSize: '3rem',
          fontWeight: 'bold',
          color: '#e87cff',
          marginBottom: 16,
        }}
      >
        Create Visual Symphonies
      </h1>
      <p
        style={{
          fontSize: '1.25rem',
          color: '#bdb7d2',
          marginBottom: 32,
          maxWidth: 600,
          textAlign: 'center',
        }}
      >
        Upload your video clips, map them to MIDI instruments, and compose
        stunning video collages that dance to the rhythm of music.
      </p>
      <button
        style={{
          background: '#a259ff',
          color: '#fff',
          border: 'none',
          borderRadius: 8,
          padding: '16px 32px',
          fontSize: '1.2rem',
          fontWeight: 'bold',
          cursor: 'pointer',
          marginBottom: 40,
        }}
        onClick={onLogin}
      >
        Start Creating
      </button>
      <p
        style={{
          margin: '0 0 32px',
          maxWidth: 700,
          textAlign: 'center',
          color: '#fbbf24',
          lineHeight: 1.6,
          fontSize: '0.98rem',
        }}
      >
        Private collaboration and local downloads are the focus of this
        release. Upload only content you own or are authorized to use.
      </p>
      <div style={{ display: 'flex', gap: 32 }}>
        <div
          style={{
            background: '#181828',
            borderRadius: 12,
            padding: 32,
            minWidth: 260,
            textAlign: 'center',
          }}
        >
          <div style={{ fontSize: 32, marginBottom: 12 }}>🎥</div>
          <h2 style={{ fontSize: '1.2rem', color: '#fff', marginBottom: 8 }}>
            Upload Clips
          </h2>
          <p style={{ color: '#bdb7d2', fontSize: '1rem' }}>
            Upload video clips to represent each instrument and drum in your
            composition.
          </p>
        </div>
        <div
          style={{
            background: '#181828',
            borderRadius: 12,
            padding: 32,
            minWidth: 260,
            textAlign: 'center',
          }}
        >
          <div style={{ fontSize: 32, marginBottom: 12 }}>🎵</div>
          <h2 style={{ fontSize: '1.2rem', color: '#fff', marginBottom: 8 }}>
            Map to MIDI
          </h2>
          <p style={{ color: '#bdb7d2', fontSize: '1rem' }}>
            Assign your clips to MIDI tracks and let our engine synchronize them
            perfectly.
          </p>
        </div>
        <div
          style={{
            background: '#181828',
            borderRadius: 12,
            padding: 32,
            minWidth: 260,
            textAlign: 'center',
          }}
        >
          <div style={{ fontSize: 32, marginBottom: 12 }}>🤝</div>
          <h2 style={{ fontSize: '1.2rem', color: '#fff', marginBottom: 8 }}>
            Private Collaboration
          </h2>
          <p style={{ color: '#bdb7d2', fontSize: '1rem' }}>
            Invite collaborators into projects, work together in the editor,
            and keep outputs private by default.
          </p>
        </div>
      </div>
      <div
        style={{
          display: 'flex',
          gap: 16,
          flexWrap: 'wrap',
          justifyContent: 'center',
          marginTop: 32,
        }}
      >
        <button
          type='button'
          onClick={() => setActiveLegalDocument('terms')}
          style={{
            border: 'none',
            background: 'none',
            color: '#f8fafc',
            textDecoration: 'underline',
            cursor: 'pointer',
          }}
        >
          Terms of Use
        </button>
        <button
          type='button'
          onClick={() => setActiveLegalDocument('privacy')}
          style={{
            border: 'none',
            background: 'none',
            color: '#f8fafc',
            textDecoration: 'underline',
            cursor: 'pointer',
          }}
        >
          Privacy Policy
        </button>
        <button
          type='button'
          onClick={() => setActiveLegalDocument('copyright')}
          style={{
            border: 'none',
            background: 'none',
            color: '#f8fafc',
            textDecoration: 'underline',
            cursor: 'pointer',
          }}
        >
          Copyright and DMCA
        </button>
      </div>
    </div>
  );
}
