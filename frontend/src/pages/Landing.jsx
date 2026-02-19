import React from 'react';
import '../index.css';

export default function Landing() {
  return (
    <div className="container text-center mt-5">
      <h1>Spotify Analytics</h1>
      <p>Analyze your listening habits and generate playlists.</p>
      <p className="text-muted">
        We request Spotify playlist permissions to create recommendation playlists in your account.
      </p>
      <a href="http://127.0.0.1:5000/login" className="btn btn-primary btn-lg" style={{backgroundColor: '#1DB954', border: 'none', color: 'black', borderRadius: '25px'}}>
        Connect Spotify
      </a>
    </div>
  );
}
