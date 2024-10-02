CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    color TEXT NOT NULL
);

-- Update the default cluster to include a color
INSERT OR IGNORE INTO clusters (id, name, color) VALUES (-1, 'Noise', '#CCCCCC');

CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER DEFAULT -1,
    timestamp DATETIME,
    raw_data TEXT,
    preprocessed_text TEXT,
    embedding BLOB,
    sentiment INTEGER,
    tsne_x REAL,
    tsne_y REAL,
    tsne_z REAL,
    FOREIGN KEY (cluster_id) REFERENCES clusters(id)
);

-- Insert default cluster
INSERT OR IGNORE INTO clusters (id, name) VALUES (-1, 'Noise');