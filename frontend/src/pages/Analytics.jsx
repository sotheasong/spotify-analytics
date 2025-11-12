import React, { useEffect, useState } from 'react';

export default function Analytics() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch('http://127.0.0.1:5000/api/user-data', { credentials: 'include' })
      .then(res => res.json())
      .then(json => setData(json))
      .catch(err => console.error(err));
  }, []);

  if (!data) return <p className="text-center mt-5">Loading...</p>;

  return (
    <div className="container mt-5">
      <h2>Your Top Tracks</h2>
      <ul className="list-group mb-4">
        {data.top_tracks.map((track, idx) => (
          <li key={idx} className="list-group-item">
            {track.name} — {track.artists.join(', ')}
          </li>
        ))}
      </ul>

      <h2>Your Top Artists</h2>
      <ul className="list-group">
        {data.top_artists.map((artist, idx) => (
          <li key={idx} className="list-group-item">
            {artist.name}
          </li>
        ))}
      </ul>
    </div>
  );
}
