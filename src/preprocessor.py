import json
import re

def preprocess_logs(logs, selected_fields, db_manager, update_progress, update_status):
    total_logs = len(logs)
    for i, log in enumerate(logs):
        log_id = log[0]  # Assuming the log ID is in the first column
        raw_data = json.loads(log[3])  # Assuming raw_data is in the fourth column

        preprocessed_text = ""
        for field in selected_fields:
            if field in raw_data:
                value = raw_data[field]
                # Check if the value is not None and not an empty string
                if value not in (None, "", "None"):
                    preprocessed_text += f"{value}\n"  # Only append the value, not the field name

        # Remove special characters using regex
        preprocessed_text = re.sub(r'[^\w\s]', '', preprocessed_text)

        # Only update if preprocessed_text is not empty after processing
        if preprocessed_text.strip():
            db_manager.update_preprocessed_text(log_id, preprocessed_text)

        # Update progress
        progress = int((i + 1) / total_logs * 100)
        update_progress(progress)
        update_status(f"Preprocessed {i + 1}/{total_logs} logs")

    db_manager.commit()
