import React from 'react';

export default function Landing() {
  return (
    <div className="container text-center mt-5">
      <h1>Spotify Analytics</h1>
      <p>Analyze your listening habits and generate playlists.</p>
      <a href="http://127.0.0.1:5000/login" className="btn btn-primary btn-lg">
        Connect Spotify
      </a>
    </div>
  );
}
