import numpy as np
from sklearn.cluster import DBSCAN
from PyQt6.QtCore import QThread, pyqtSignal
from src.db_manager import DatabaseManager

class ClusteringThread(QThread):
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, db_name, epsilon=0.5, min_samples=5):
        super().__init__()
        self.db_name = db_name
        self.epsilon = epsilon
        self.min_samples = min_samples

    def run(self):
        self.status_update.emit("Starting clustering process...")
        self.progress_update.emit(0)
        db_manager = DatabaseManager(self.db_name)

        # Fetch embeddings
        embeddings, log_ids = self.fetch_embeddings(db_manager)
        if len(embeddings) == 0:
            self.status_update.emit("No embeddings found in the database.")
            return

        # Perform clustering
        self.status_update.emit(f"Performing DBSCAN clustering on {len(embeddings)} embeddings...")
        self.progress_update.emit(25)  # 25% progress after fetching embeddings

        dbscan = DBSCAN(eps=self.epsilon, min_samples=self.min_samples)
        cluster_labels = dbscan.fit_predict(embeddings)

        self.progress_update.emit(75)  # 75% progress after clustering

        # Update database with clustering results
        self.update_database_with_clusters(db_manager, log_ids, cluster_labels)

        self.status_update.emit("Clustering completed and database updated.")
        self.progress_update.emit(100)  # 100% progress when finished
        self.finished.emit()

    def fetch_embeddings(self, db_manager):
        embeddings = []
        log_ids = []
        logs = db_manager.get_all_logs()
        print(f"Total logs retrieved: {len(logs)}")
        for log in logs:
            if log[4] is not None:  # Assuming the embedding is in the 16th column
                try:
                    if isinstance(log[5], bytes):
                        # If it's already bytes, use it directly
                        embedding = np.frombuffer(log[5], dtype=np.float32)
                    elif isinstance(log[5], str):
                        # If it's a string, try to convert from string representation
                        embedding = np.fromstring(log[5].strip('[]'), sep=',', dtype=np.float32)
                    else:
                        print(f"Unexpected embedding type for log {log[0]}: {type(log[5])}")
                        continue
                    if embedding.size > 0:
                        embeddings.append(embedding)
                        log_ids.append(log[0])  # Log ID
                    else:
                        print(f"Skipping log {log[0]}: Empty embedding")
                except Exception as e:
                    print(f"Error processing embedding for log {log[0]}: {str(e)}")
        
        print(f"Total valid embeddings: {len(embeddings)}")
        if not embeddings:
            print("No valid embeddings found in the database.")
            return np.array([]), []

        # Ensure all embeddings have the same dimensionality
        embedding_dim = embeddings[0].shape[0]
        valid_embeddings = [emb for emb in embeddings if emb.shape[0] == embedding_dim]
        valid_log_ids = [log_id for emb, log_id in zip(embeddings, log_ids) if emb.shape[0] == embedding_dim]
        
        if len(valid_embeddings) != len(embeddings):
            print(f"Removed {len(embeddings) - len(valid_embeddings)} embeddings with inconsistent dimensions.")
        
        return np.array(valid_embeddings), valid_log_ids

    def update_database_with_clusters(self, db_manager, log_ids, cluster_labels):
        cluster_ids = {}
        default_cluster_id = -1
        unique_clusters = set(cluster_labels) - {-1}  # Exclude -1 from unique clusters
        total_clusters = len(unique_clusters)
        
        for i, cluster_label in enumerate(unique_clusters):
            cluster_name = f'Cluster {cluster_label}'
            cluster_id = db_manager.create_cluster(cluster_name)
            cluster_ids[cluster_label] = cluster_id
            progress = 75 + int((i + 1) / total_clusters * 20)  # Progress from 75% to 95%
            self.progress_update.emit(progress)

        total_logs = len(log_ids)
        for i, (log_id, cluster_label) in enumerate(zip(log_ids, cluster_labels)):
            if cluster_label == -1:
                # Assume the default cluster ID is known or can be retrieved
                db_manager.assign_to_cluster(log_id, default_cluster_id)
            else:
                db_manager.assign_to_cluster(log_id, cluster_ids[cluster_label])
            if i % 100 == 0:  # Update progress every 100 logs
                progress = 95 + int((i + 1) / total_logs * 5)  # Progress from 95% to 100%
                self.progress_update.emit(progress)

        self.status_update.emit(f"Created {total_clusters} clusters. Points labeled -1 assigned to default cluster.")