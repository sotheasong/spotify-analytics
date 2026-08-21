import React, { useEffect, useState } from 'react';

export default function Analytics() {
  const [data, setData] = useState(null);
  const [loadingError, setLoadingError] = useState('');
  const [isCreatingPlaylist, setIsCreatingPlaylist] = useState(false);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [playlistError, setPlaylistError] = useState('');
  const [playlistResult, setPlaylistResult] = useState(null);
  const [previewError, setPreviewError] = useState('');
  const [previewResult, setPreviewResult] = useState(null); // { run_id, meta, recommendations }
  const [hiddenTrackIds, setHiddenTrackIds] = useState(() => new Set());
  const [model, setModel] = useState('cosine');
  const [moodId, setMoodId] = useState('');
  const [moods, setMoods] = useState([]);
  const [topK, setTopK] = useState(50);
  const [dedupeMode, setDedupeMode] = useState('track_name');
  const [recencyHalflifeDays, setRecencyHalflifeDays] = useState(14);
  const [genreWeight, setGenreWeight] = useState(0.0);
  const [popularityWeight, setPopularityWeight] = useState(0.0);
  const [perTrackK, setPerTrackK] = useState(40);
  const [maxUserTracks, setMaxUserTracks] = useState(0);
  const [minSimilarity, setMinSimilarity] = useState(0.0);

  const buildRequestBody = () => {
    const requestBody = {
      model,
      top_k: Number(topK),
      recency_halflife_days: Number(recencyHalflifeDays),
      dedupe_mode: dedupeMode,
      genre_weight: Number(genreWeight),
      popularity_weight: Number(popularityWeight),
    };

    if (moodId !== '') {
      requestBody.mood_id = Number(moodId);
      requestBody.restrict_to_mood = true;
    }

    if (model === 'knn') {
      requestBody.per_track_k = Number(perTrackK);
      requestBody.max_user_tracks = Number(maxUserTracks);
      requestBody.min_similarity = Number(minSimilarity);
    }

    return requestBody;
  };

  const previewRecommendations = async () => {
    setIsPreviewing(true);
    setPreviewError('');
    setPreviewResult(null);
    setHiddenTrackIds(new Set());
    setPlaylistError('');
    setPlaylistResult(null);

    try {
      const requestBody = buildRequestBody();
      const response = await fetch('http://127.0.0.1:5000/api/recommendations/preview', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || 'Failed to preview recommendations');
      }
      setPreviewResult(payload);
    } catch (err) {
      setPreviewError(err.message || 'Failed to preview recommendations');
    } finally {
      setIsPreviewing(false);
    }
  };

  const toggleHidden = (trackId) => {
    if (!trackId) return;
    setHiddenTrackIds((prev) => {
      const next = new Set(prev);
      if (next.has(trackId)) next.delete(trackId);
      else next.add(trackId);
      return next;
    });
  };

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

  useEffect(() => {
    fetch('http://127.0.0.1:5000/api/moods', { credentials: 'include' })
      .then(async (res) => {
        const payload = await res.json().catch(() => ({}));
        if (!res.ok) return null;
        return payload;
      })
      .then((json) => {
        if (json && Array.isArray(json.moods)) setMoods(json.moods);
      })
      .catch(() => {});
  }, []);

  const createPlaylist = async () => {
    setIsCreatingPlaylist(true);
    setPlaylistError('');
    setPlaylistResult(null);
    try {
      const requestBody = buildRequestBody();
      if (previewResult && previewResult.run_id) {
        requestBody.run_id = previewResult.run_id;
        requestBody.exclude_track_ids = Array.from(hiddenTrackIds);
      }

      const response = await fetch('http://127.0.0.1:5000/api/recommendations/create-playlist', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
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
          <label className="form-label">Model</label>
          <select
            className="form-select"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            <option value="cosine">Cosine (profile baseline)</option>
            <option value="knn">KNN (item-to-item)</option>
          </select>
        </div>
        <div className="col-md-3">
          <label className="form-label">Mood (optional)</label>
          <select
            className="form-select"
            value={moodId}
            onChange={(e) => setMoodId(e.target.value)}
          >
            <option value="">All moods</option>
            {moods.map((m) => (
              <option key={m.mood_id} value={String(m.mood_id)}>
                {m.mood_id} — {m.name}
              </option>
            ))}
          </select>
        </div>
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
        <div className="col-md-3">
          <label className="form-label">Genre weight (0 = off)</label>
          <input
            type="number"
            className="form-control"
            min="0"
            max="1"
            step="0.05"
            value={genreWeight}
            onChange={(e) => setGenreWeight(e.target.value)}
          />
          <div className="form-text">
            Blends audio similarity with genre similarity.
          </div>
        </div>
        <div className="col-md-3">
          <label className="form-label">Popularity weight (0 = off)</label>
          <input
            type="number"
            className="form-control"
            min="0"
            max="1"
            step="0.05"
            value={popularityWeight}
            onChange={(e) => setPopularityWeight(e.target.value)}
          />
          <div className="form-text">
            Re-ranks toward globally popular tracks.
          </div>
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
        {model === 'knn' && (
          <>
            <div className="col-md-3">
              <label className="form-label">Per-track K</label>
              <input
                type="number"
                className="form-control"
                min="1"
                max="200"
                value={perTrackK}
                onChange={(e) => setPerTrackK(e.target.value)}
              />
            </div>
            <div className="col-md-3">
              <label className="form-label">Max user tracks (0 = all)</label>
              <input
                type="number"
                className="form-control"
                min="0"
                max="5000"
                value={maxUserTracks}
                onChange={(e) => setMaxUserTracks(e.target.value)}
              />
            </div>
            <div className="col-md-3">
              <label className="form-label">Min similarity</label>
              <input
                type="number"
                className="form-control"
                min="0"
                max="1"
                step="0.01"
                value={minSimilarity}
                onChange={(e) => setMinSimilarity(e.target.value)}
              />
            </div>
          </>
        )}
        <div className="col-md-3">
          <button
            className="btn btn-outline-primary w-100"
            disabled={isPreviewing}
            onClick={previewRecommendations}
          >
            {isPreviewing ? 'Loading Preview...' : 'Preview Recommendations'}
          </button>
        </div>
        <div className="col-md-4">
          <button
            className="btn btn-success w-100"
            disabled={isCreatingPlaylist || !previewResult}
            onClick={createPlaylist}
          >
            {isCreatingPlaylist ? 'Creating Playlist...' : 'Create Playlist (Reviewed)'}
          </button>
          <div className="form-text">
            Preview first, then hide tracks that don’t fit the vibe.
          </div>
        </div>
      </div>

      {previewError && (
        <div className="alert alert-danger" role="alert">
          {previewError}
        </div>
      )}

      {previewResult && Array.isArray(previewResult.recommendations) && (
        <div className="card mb-3">
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-center mb-2">
              <h5 className="card-title mb-0">Preview</h5>
              <div className="text-muted small">
                Hidden: {hiddenTrackIds.size} / {previewResult.recommendations.length}
              </div>
            </div>
            <div className="table-responsive">
              <table className="table table-sm align-middle">
                <thead>
                  <tr>
                    <th style={{ width: 90 }}>Hide</th>
                    <th>Track</th>
                    <th style={{ width: 110 }}>Score</th>
                    <th style={{ width: 140 }}>Genre</th>
                  </tr>
                </thead>
                <tbody>
                  {previewResult.recommendations.map((r, idx) => {
                    const tid = String(r.track_id ?? r.id ?? '');
                    const hidden = tid && hiddenTrackIds.has(tid);
                    return (
                      <tr key={`${tid}-${idx}`} className={hidden ? 'table-secondary' : ''}>
                        <td>
                          <input
                            type="checkbox"
                            checked={hidden}
                            onChange={() => toggleHidden(tid)}
                            disabled={!tid}
                          />
                        </td>
                        <td>
                          <div className="fw-semibold">{r.track_name || '(unknown)'}</div>
                          <div className="text-muted small">{r.artists || ''}</div>
                        </td>
                        <td>{typeof r.score === 'number' ? r.score.toFixed(4) : ''}</td>
                        <td className="text-muted small">{r.track_genre || ''}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

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
