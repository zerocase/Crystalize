import json
import jsonlines
import os
from PyQt6.QtCore import QThread, pyqtSignal
from .db_manager import DatabaseManager

class ImportThread(QThread):
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    common_fields_found = pyqtSignal(list)  # Signal to emit common fields

    def __init__(self, file_paths, db_path):
        super().__init__()
        self.file_paths = file_paths
        self.db_path = db_path
        self.common_fields = None  # To store the common fields
    
    def run(self):
        db_manager = DatabaseManager(self.db_path)
        db_manager.create_tables()
        total_files = len(self.file_paths)
        total_logs_processed = 0
        total_logs_inserted = 0
        first_log = True
        common_fields = []

        for file_index, file_path in enumerate(self.file_paths):
            self.status_update.emit(f"Processing file {file_index + 1} of {total_files}: {os.path.basename(file_path)}")
            try:
                logs = self.parse_log_file(file_path)
                self.status_update.emit(f"Parsed {len(logs)} logs from file")

                for log_index, log in enumerate(logs):
                    try:
                        if first_log:
                            common_fields = list(log.keys())  # Initialize with the first log's keys, preserving order
                            first_log = False
                        else:
                            # Update common_fields to keep only fields present in all logs, preserving order
                            common_fields = [field for field in common_fields if field in log]
                        
                        log_id = db_manager.insert_log(log)
                        if log_id:
                            total_logs_inserted += 1
                        total_logs_processed += 1

                        if log_index % 100 == 0:  # Update status every 100 logs
                            self.status_update.emit(f"Processed {log_index + 1} logs in current file")

                    except Exception as e:
                        self.status_update.emit(f"Error processing log in file {file_path}, index {log_index}: {str(e)}")

                progress = int((file_index * 100 + (log_index + 1) / len(logs) * 100) / total_files)
                self.progress_update.emit(progress)
                db_manager.commit()
                self.status_update.emit(f"Committed changes for file {file_index + 1}")

            except Exception as e:
                self.status_update.emit(f"Error processing file {file_path}: {str(e)}")

        db_manager.close()

        # Emit the common fields (now ordered)
        self.common_fields_found.emit(common_fields)
        self.status_update.emit(f"Import completed. Processed {total_logs_processed} logs, inserted {total_logs_inserted} logs.")

    def parse_log_file(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        
        if file_extension.lower() in ('.json', '.jsonl'):
            return self.parse_json_logs(file_path)
        elif file_extension.lower() in ('.log', '.txt'):
            return self.parse_text_logs(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def parse_json_logs(self, file_path):
        logs = []
        with jsonlines.open(file_path) as reader:
            for log in reader:
                logs.append(log)
        return logs

    def parse_text_logs(self, file_path):
        logs = []
        with open(file_path, 'r') as file:
            for line_index, line in enumerate(file, start=1):
                try:
                    log = json.loads(line.strip())
                    logs.append(log)
                except json.JSONDecodeError:
                    # Emit a status update with the line number instead of the full line
                    truncated_line = (line.strip()[:75] + '...') if len(line) > 75 else line.strip()
                    self.status_update.emit(f"Error parsing line {line_index}: {truncated_line}")
        return logs
