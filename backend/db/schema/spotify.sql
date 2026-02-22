CREATE TABLE IF NOT EXISTS recent_tracks_audio_features (
    id TEXT,
    name TEXT,
    acousticness DOUBLE PRECISION,
    danceability DOUBLE PRECISION,
    energy DOUBLE PRECISION,
    instrumentalness DOUBLE PRECISION,
    key DOUBLE PRECISION,
    liveness DOUBLE PRECISION,
    mode DOUBLE PRECISION,
    speechiness DOUBLE PRECISION,
    valence DOUBLE PRECISION,
    tempo DOUBLE PRECISION,
    collection_date DATE NOT NULL,
    PRIMARY KEY (id, collection_date)
);

CREATE TABLE IF NOT EXISTS recent_tracks (
    track_id TEXT,
    track_name TEXT,
    album_name TEXT,
    played_at DATE,
    artist_name TEXT,
    collection_date DATE NOT NULL,
    PRIMARY KEY (track_id, collection_date)
);

-- recent_track_artists: normalized track ↔ artist mapping for recents
CREATE TABLE IF NOT EXISTS recent_track_artists (
    track_id TEXT,
    artist_id TEXT,
    artist_name TEXT,
    collection_date DATE NOT NULL,
    PRIMARY KEY (track_id, artist_id, collection_date)
);

-- artist_genres: cache artist genres (Spotify exposes genres on artists, not tracks)
CREATE TABLE IF NOT EXISTS artist_genres (
    id TEXT,
    name TEXT,
    popularity INT,
    genres TEXT,
    follower_count INT,
    collection_date DATE NOT NULL,
    PRIMARY KEY (id, collection_date)
);

-- top_artists
CREATE TABLE IF NOT EXISTS top_artists (
    "Unnamed: 0" INT,  -- original index column
    id TEXT,
    name TEXT,
    popularity INT,
    genres TEXT,
    follower_count INT,
    collection_date DATE,
    PRIMARY KEY (id, collection_date)
);

-- top_tracks
CREATE TABLE IF NOT EXISTS top_tracks (
    "Unnamed: 0" INT,  -- original index column
    id TEXT,
    track_name TEXT,
    artist_name TEXT,
    album_name TEXT,
    release_date DATE,
    duration_min DOUBLE PRECISION,
    popularity INT,
    explicit BOOLEAN,
    collection_date DATE,
    PRIMARY KEY (id, collection_date) 
);

-- top_tracks_audio_features
CREATE TABLE IF NOT EXISTS top_tracks_audio_features (
    "Unnamed: 0" INT,  -- original index column
    id TEXT,
    name TEXT,
    acousticness DOUBLE PRECISION,
    danceability DOUBLE PRECISION,
    energy DOUBLE PRECISION,
    instrumentalness DOUBLE PRECISION,
    key DOUBLE PRECISION,
    liveness DOUBLE PRECISION,
    mode DOUBLE PRECISION,
    speechiness DOUBLE PRECISION,
    valence DOUBLE PRECISION,
    temp DOUBLE PRECISION,
    collection_date DATE,
    PRIMARY KEY (id, collection_date)
);
