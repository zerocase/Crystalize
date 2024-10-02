import sqlite3
import json
import os
from datetime import datetime
import threading
import numpy as np
import random
import logging
import colorsys

class DatabaseManager:
    _local = threading.local()

    def __init__(self, db_name):
        self.db_name = db_name
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"DatabaseManager initialized with database: {db_name}")

    def get_connection(self):
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.db_name)
        return self._local.connection

    def get_cursor(self):
        return self.get_connection().cursor()

    def create_tables(self):
        cursor = self.get_cursor()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        schema_path = os.path.join(project_root, 'database_schema.sql')
        with open(schema_path, 'r') as schema_file:
            cursor.executescript(schema_file.read())
        self.get_connection().commit()

    def insert_log(self, log):
        cursor = self.get_cursor()
        try:
            cursor.execute('''
            INSERT INTO logs (cluster_id, raw_data)
            VALUES (?, ?)
            ''', (
                -1,  # Default cluster_id
                json.dumps(log),  # Store the entire log as JSON in raw_data
            ))
            return cursor.lastrowid
        except Exception as e:
            print(f"Error inserting log: {str(e)}")
            print(f"Log data: {log}")
            raise
    
    def check_embeddings_exist(self):
        cursor = self.get_cursor()
        cursor.execute("SELECT COUNT(*) FROM logs WHERE embedding IS NOT NULL")
        count = cursor.fetchone()[0]
        return count > 0

    def get_logs_without_embeddings(self):
        cursor = self.get_cursor()
        cursor.execute("SELECT id, method, url, body FROM logs WHERE embedding IS NULL")
        return [{'id': row[0], 'method': row[1], 'url': row[2], 'body': row[3]} for row in cursor.fetchall()]
    
    
    
    def reset_clusters(self):
        cursor = self.get_cursor()
        try:
            # Move all logs to cluster -1
            cursor.execute("UPDATE logs SET cluster_id = -1")
            
            # Delete all existing clusters except -1
            cursor.execute("DELETE FROM clusters WHERE id != -1")
            
            # Ensure cluster -1 exists
            cursor.execute("INSERT OR IGNORE INTO clusters (id, name, color) VALUES (-1, 'Noise', '#808080')")
            
            self.get_connection().commit()
            logging.info("Clusters reset completed successfully.")
        except Exception as e:
            logging.error(f"Error resetting clusters: {e}")
            self.get_connection().rollback()
    
    def clear_database(self):
        cursor = self.get_cursor()
        try:
            # Delete all logs
            cursor.execute("DELETE FROM logs")
            
            # Delete all clusters except the default one (id = -1)
            cursor.execute("DELETE FROM clusters WHERE id != -1")
            
            # Reset the auto-increment counter for logs and clusters
            cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('logs', 'clusters')")
            
            # Ensure the default cluster exists
            cursor.execute("INSERT OR IGNORE INTO clusters (id, name, color) VALUES (-1, 'Noise', '#CCCCCC')")
            
            self.get_connection().commit()
            logging.info("Database cleared successfully.")
        except Exception as e:
            logging.error(f"Error clearing database: {e}")
            self.get_connection().rollback()

    def delete_coordinates(self):
        cursor = self.get_cursor()
        try:
            # Set tsne_x, tsne_y, and tsne_z to NULL for all logs
            cursor.execute("UPDATE logs SET tsne_x = NULL, tsne_y = NULL, tsne_z = NULL")
            
            self.get_connection().commit()
            logging.info("Coordinates deleted successfully.")
        except Exception as e:
            logging.error(f"Error deleting coordinates: {e}")
            self.get_connection().rollback()

    def prepare_for_embedding_regeneration(self):
        try:
            self.reset_clusters()
            self.delete_coordinates()
            logging.info("Preparation for embedding regeneration completed.")
        except Exception as e:
            logging.error(f"Error during preparation for embedding regeneration: {e}")


    def update_log_embedding(self, log_id, embedding):
        cursor = self.get_cursor()
        try:
            logging.info(f"Updating log ID {log_id} with embedding.")
            cursor.execute("UPDATE logs SET embedding = ? WHERE id = ?", (embedding.tobytes(), log_id))
            self.get_connection().commit()
            logging.info(f"Successfully updated log ID {log_id} with embedding.")
        except Exception as e:
            logging.error(f"Error updating log ID {log_id}: {e}")

    
    def check_preprocessed_text_exists(self):
        cursor = self.get_cursor()
        cursor.execute("SELECT COUNT(*) FROM logs WHERE preprocessed_text IS NOT NULL AND preprocessed_text != ''")
        count = cursor.fetchone()[0]
        return count > 0
    
    def update_preprocessed_text(self, log_id, preprocessed_text):
        cursor = self.get_cursor()
        try:
            cursor.execute("UPDATE logs SET preprocessed_text = ? WHERE id = ?", (preprocessed_text, log_id))
            self.get_connection().commit()
        except Exception as e:
            print(f"Error updating preprocessed_text for log_id {log_id}: {e}")

    def insert_preprocessed_log(self, log_id, preprocessed_data):
        try:
            # Assuming you have a table 'preprocessed_logs' with 'log_id' and 'data'
            query = "INSERT INTO preprocessed_logs (log_id, data) VALUES (?, ?)"
            self.conn.execute(query, (log_id, preprocessed_data))
            self.conn.commit()
            print(f"Inserted preprocessed data for log {log_id}")
        except Exception as e:
            print(f"Error inserting preprocessed log {log_id}: {e}")

    def get_sample_log(self):
        cursor = self.get_cursor()
        cursor.execute("SELECT * FROM logs LIMIT 1")
        return cursor.fetchone()

    def get_clusters(self):
        cursor = self.get_cursor()
        cursor.execute('SELECT id, name, color FROM clusters')
        return cursor.fetchall()

    def get_logs_in_cluster(self, cluster_id):
        cursor = self.get_cursor()
        cursor.execute("SELECT * FROM logs WHERE cluster_id = ?", (cluster_id,))
        return cursor.fetchall()

    def get_all_logs(self):
        cursor = self.get_cursor()
        cursor.execute('SELECT * FROM logs')
        return cursor.fetchall()
    
    def get_all_logs_with_coordinates(self):
        cursor = self.get_cursor()
        cursor.execute('SELECT id, cluster_id, tsne_x, tsne_y, tsne_z FROM logs WHERE tsne_x IS NOT NULL')
        return cursor.fetchall()

    def create_cluster(self, name):
        cursor = self.get_cursor()
        color = self.generate_random_color()
        cursor.execute('INSERT INTO clusters (name, color) VALUES (?, ?)', (name, color))
        self.get_connection().commit()
        return cursor.lastrowid
    
    def get_cluster_log_counts(self):
        cursor = self.get_cursor()
        cursor.execute('''
        SELECT c.id, c.name, c.color, COUNT(l.id) as log_count
        FROM clusters c
        LEFT JOIN logs l ON c.id = l.cluster_id
        GROUP BY c.id
        ORDER BY c.id
        ''')
        return cursor.fetchall()

    def generate_random_color(self):
        r = random.randint(100, 255)
        g = random.randint(100, 255)
        b = random.randint(100, 255)
        
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    def get_cluster_color(self, cluster_id):
        cursor = self.get_cursor()
        cursor.execute('SELECT color FROM clusters WHERE id = ?', (cluster_id,))
        result = cursor.fetchone()
        return result[0] if result else None

    def assign_to_cluster(self, log_id, cluster_id):
        cursor = self.get_cursor()
        cursor.execute('UPDATE logs SET cluster_id = ? WHERE id = ?', (cluster_id, log_id))
        self.get_connection().commit()

    def commit(self):
        self.get_connection().commit()

    def close(self):
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            del self._local.connection