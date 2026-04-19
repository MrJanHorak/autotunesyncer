import React from 'react';

export default function LandingPage({ onLogin }) {
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
          <div style={{ fontSize: 32, marginBottom: 12 }}>🔗</div>
          <h2 style={{ fontSize: '1.2rem', color: '#fff', marginBottom: 8 }}>
            Share & Discover
          </h2>
          <p style={{ color: '#bdb7d2', fontSize: '1rem' }}>
            Share your creations with the community and discover amazing visual
            symphonies.
          </p>
        </div>
      </div>
    </div>
  );
}
