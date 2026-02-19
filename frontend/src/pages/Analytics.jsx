import React, { useEffect, useState } from 'react';

export default function Analytics() {
  const [data, setData] = useState(null);
  const [loadingError, setLoadingError] = useState('');
  const [isCreatingPlaylist, setIsCreatingPlaylist] = useState(false);
  const [playlistError, setPlaylistError] = useState('');
  const [playlistResult, setPlaylistResult] = useState(null);
  const [topK, setTopK] = useState(50);
  const [dedupeMode, setDedupeMode] = useState('track_name');
  const [recencyHalflifeDays, setRecencyHalflifeDays] = useState(14);

  useEffect(() => {
    fetch('http://127.0.0.1:5000/get-info', { credentials: 'include' })
      .then(async (res) => {
        const payload = await res.json().catch(() => ({}));
        if (!res.ok) {
          throw new Error(payload.error || 'Failed to load analytics data');
        }
        return payload;
      })
      .then(json => setData(json))
      .catch(err => setLoadingError(err.message || 'Failed to load analytics data'));
  }, []);

  const createPlaylist = async () => {
    setIsCreatingPlaylist(true);
    setPlaylistError('');
    setPlaylistResult(null);
    try {
      const response = await fetch('http://127.0.0.1:5000/api/recommendations/create-playlist', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          top_k: Number(topK),
          recency_halflife_days: Number(recencyHalflifeDays),
          dedupe_mode: dedupeMode,
        }),
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || 'Failed to create playlist');
      }
      setPlaylistResult(payload);
    } catch (err) {
      setPlaylistError(err.message || 'Failed to create playlist');
    } finally {
      setIsCreatingPlaylist(false);
    }
  };

  if (!data) {
    return (
      <div className="container mt-5">
        {!loadingError ? (
          <p className="text-center">Loading...</p>
        ) : (
          <div className="alert alert-warning" role="alert">
            {loadingError}
            <div className="mt-2">
              <a href="http://127.0.0.1:5000/login" className="btn btn-sm btn-primary">
                Connect Spotify
              </a>
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="container mt-5">
      <h2 className="mb-3">Recommended Playlist</h2>
      <p className="text-muted">
        Create a Spotify playlist from your current model recommendations and open it directly in Spotify.
      </p>

      <div className="row g-3 align-items-end mb-3">
        <div className="col-md-3">
          <label className="form-label">Top K</label>
          <input
            type="number"
            className="form-control"
            min="1"
            max="500"
            value={topK}
            onChange={(e) => setTopK(e.target.value)}
          />
        </div>
        <div className="col-md-3">
          <label className="form-label">Recency Halflife (days)</label>
          <input
            type="number"
            className="form-control"
            min="1"
            max="365"
            value={recencyHalflifeDays}
            onChange={(e) => setRecencyHalflifeDays(e.target.value)}
          />
        </div>
        <div className="col-md-4">
          <label className="form-label">Deduplicate by</label>
          <select
            className="form-select"
            value={dedupeMode}
            onChange={(e) => setDedupeMode(e.target.value)}
          >
            <option value="track_name">Track name</option>
            <option value="track_name_artists">Track name + artists</option>
            <option value="track_id">Track ID</option>
          </select>
        </div>
        <div className="col-md-5">
          <button
            className="btn btn-success w-100"
            disabled={isCreatingPlaylist}
            onClick={createPlaylist}
          >
            {isCreatingPlaylist ? 'Creating Playlist...' : 'Create Recommended Playlist'}
          </button>
        </div>
      </div>

      {playlistError && (
        <div className="alert alert-danger" role="alert">
          {playlistError}
        </div>
      )}

      {playlistResult && (
        <div className="alert alert-success" role="alert">
          <div className="mb-2">
            Playlist created with <strong>{playlistResult.tracks_added}</strong> tracks.
          </div>
          {playlistResult.playlist_url && (
            <a
              href={playlistResult.playlist_url}
              target="_blank"
              rel="noreferrer"
              className="btn btn-outline-dark"
            >
              Open in Spotify
            </a>
          )}
        </div>
      )}
    </div>
  );
}
