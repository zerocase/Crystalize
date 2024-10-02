from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.manifold import TSNE
import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal
from .db_manager import DatabaseManager

class EmbeddingGeneratorThread(QThread):
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)

    def __init__(self, db_path, model_name):
        super().__init__()
        self.db_path = db_path
        self.model_name = model_name
        self.semantic_model = SentenceTransformer(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.semantic_model.to(self.device)
        self.sentiment_model.to(self.device)

    def run(self):
        self.status_update.emit("Starting embedding generation...")
        db_manager = DatabaseManager(self.db_path)
        
        # Prepare for embedding regeneration
        self.status_update.emit("Preparing for embedding regeneration...")
        db_manager.prepare_for_embedding_regeneration()

        cursor = db_manager.get_cursor()

        # Get all logs
        cursor.execute("SELECT id, preprocessed_text FROM logs")
        logs = [{'id': row[0], 'text': row[1]} for row in cursor.fetchall()]
        total_logs = len(logs)
        embeddings = []
        log_ids = []

        for i, log in enumerate(logs):
            if log['text'] is None or log['text'].strip() == '':
                self.status_update.emit(f"Skipping log {i+1} due to empty text")
                continue

            semantic_embedding = self.get_semantic_embedding(log['text'])
            sentiment = self.get_sentiment(log['text'])
            sentiment_value = 1 if sentiment['label'] == 'POSITIVE' else 0
            combined_embedding = np.concatenate([semantic_embedding, np.array([sentiment_value])])
            embeddings.append(combined_embedding)
            log_ids.append(log['id'])
            db_manager.update_log_embedding(log['id'], combined_embedding)
            db_manager.get_cursor().execute("UPDATE logs SET sentiment = ? WHERE id = ?",
                                            (sentiment_value, log['id']))
            progress = int((i + 1) / total_logs * 50)  # First half of progress
            self.progress_update.emit(progress)
            self.status_update.emit(f"Generated embedding for log {i+1} of {total_logs}")

        db_manager.get_connection().commit()

        # Perform t-SNE
        self.status_update.emit("Performing dimensionality reduction...")
        tsne = TSNE(n_components=3, random_state=42)
        reduced_embeddings = tsne.fit_transform(np.array(embeddings))

        # Save reduced coordinates
        for i, (x, y, z) in enumerate(reduced_embeddings):
            cursor.execute("UPDATE logs SET tsne_x = ?, tsne_y = ?, tsne_z = ? WHERE id = ?",
                           (float(x), float(y), float(z), log_ids[i]))
            progress = 50 + int((i + 1) / total_logs * 50)  # Second half of progress
            self.progress_update.emit(progress)
            self.status_update.emit(f"Saved reduced coordinates for log {i+1} of {total_logs}")

        db_manager.get_connection().commit()
        db_manager.close()
        self.status_update.emit("Embedding generation and dimensionality reduction completed!")

    def get_semantic_embedding(self, text):
        return self.semantic_model.encode(text, convert_to_numpy=True)

    def get_sentiment(self, text):
        encoded_input = self.tokenizer(text, truncation=True, max_length=512, return_tensors='pt', padding=True)
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        with torch.no_grad():
            output = self.sentiment_model(**encoded_input)
        
        predicted_class = output.logits.argmax().item()
        score = torch.softmax(output.logits, dim=1)[0, predicted_class].item()
        
        label = 'POSITIVE' if predicted_class == 1 else 'NEGATIVE'
        return {'label': label, 'score': score}