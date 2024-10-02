from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QSplitter, QScrollArea,
                             QFileDialog, QProgressBar, QLabel, QTreeWidget, QTreeWidgetItem, QHBoxLayout, QComboBox, QSpacerItem, QSizePolicy, QCheckBox, QMessageBox, QGridLayout)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QBrush, QFont, QFontDatabase, QPainter, QPen, QIcon
from src.import_logic import ImportThread
from src.db_manager import DatabaseManager
from src.embedding_generator import EmbeddingGeneratorThread
from src.clustering import ClusteringThread
from src.visualization import Visualization3D
from src.preprocessor import preprocess_logs
import sys
import json
import os
import numpy as np

class ArrowLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(50)  # Adjust this value to change arrow length

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        pen = QPen(Qt.GlobalColor.gray, 2)  # Adjust color and thickness here
        painter.setPen(pen)
        
        # Draw the line
        painter.drawLine(self.width() // 2, 0, self.width() // 2, self.height() - 15)
        
        # Draw the arrowhead
        painter.drawLine(self.width() // 2, self.height() - 15, self.width() // 2 - 7, self.height() - 25)
        painter.drawLine(self.width() // 2, self.height() - 15, self.width() // 2 + 7, self.height() - 25)




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crystalize")
        self.setWindowIcon(QIcon('Crystalize.png'))
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(800, 600)

        self.embedding_models = [
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/all-mpnet-base-v2'
        ]

        # Set the database path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(current_dir, 'logs.db')
        
        # Initialize DatabaseManager
        self.db_manager = DatabaseManager(self.db_path)

        # Initialize common fields
        self.common_fields = []
        self.common_fields_checkboxes = []

        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Create left panel
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Create scroll area for upper controls (vertical scroll only)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Add all upper controls to the scroll layout
        self.add_upper_controls(scroll_layout)

        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)

        # Add scroll area to left layout
        left_layout.addWidget(scroll_area)

        # Add status label
        self.status_label = QLabel("Ready")
        left_layout.addWidget(self.status_label)

        # Add visualization
        self.visualization = Visualization3D()
        self.visualization.setMinimumSize(300, 300)
        left_layout.addWidget(self.visualization, 1)  # Add stretch factor
        self.visualization.hide()  # Hide it initially

        # Create right panel
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Add tree widget to a scroll area
        tree_scroll_area = QScrollArea()
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Clusters/Logs", "Visibility", "Details"])
        self.tree_widget.setColumnWidth(0, 300)
        tree_scroll_area.setWidget(self.tree_widget)
        tree_scroll_area.setWidgetResizable(True)
        right_layout.addWidget(tree_scroll_area)

        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(20)
        right_layout.addWidget(self.progress_bar)

        # Add widgets to splitter
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(0, 2)  # Left panel
        main_splitter.setStretchFactor(1, 3)  # Right panel

        # Set the main_splitter as the central widget
        self.setCentralWidget(main_splitter)

        # Initialize database manager
        self.db_manager = DatabaseManager('log_data.db')
        self.db_manager.create_tables()

        # After setting up all UI components, but before populating the tree:
        self.check_and_show_common_fields()

        # Populate tree with existing data
        self.populate_tree()

        # Call this method to update the button text initially
        self.update_preprocess_button_text()
        self.update_generate_embeddings_button()

        # Check for embeddings and start visualization if present
        self.check_and_start_visualization()

        # Connect the tree widget's item selection changed signal
        self.tree_widget.itemSelectionChanged.connect(self.on_tree_selection_changed)
    
    def add_upper_controls(self, layout):
        # Add buttons
        button_layout = QHBoxLayout()
        
        self.file_button = QPushButton("Open File")
        self.file_button.clicked.connect(self.open_file)
        self.file_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button_layout.addWidget(self.file_button)

        self.folder_button = QPushButton("Open Folder")
        self.folder_button.clicked.connect(self.open_folder)
        self.folder_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button_layout.addWidget(self.folder_button)

        # Add clear database button
        self.clear_database_button = QPushButton("Clear Database")
        self.clear_database_button.clicked.connect(self.clear_database_and_reset)
        self.clear_database_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.clear_database_button)
        
        layout.addLayout(button_layout)

        # Add arrow label
        layout.addWidget(self.create_arrow_label())

        # Add common fields section
        self.common_fields_layout = QVBoxLayout()
        layout.addLayout(self.common_fields_layout)

        # Add preprocessing section
        preprocessing_section = self.create_preprocessing_section()
        layout.addLayout(preprocessing_section)

        # Add arrow label
        layout.addWidget(self.create_arrow_label())

        # Add model selection dropdown
        layout.addWidget(QLabel("Embedding Model:"))
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(self.embedding_models)
        self.model_dropdown.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.model_dropdown.setMinimumContentsLength(20)  # Adjust this value as needed
        layout.addWidget(self.model_dropdown)

        self.generate_embeddings_button = QPushButton("Generate Embeddings")
        self.generate_embeddings_button.clicked.connect(self.generate_embeddings)
        self.generate_embeddings_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.generate_embeddings_button)

        # Add arrow label
        layout.addWidget(self.create_arrow_label())

        self.perform_clustering_button = QPushButton("Perform Clustering")
        self.perform_clustering_button.clicked.connect(self.perform_clustering)
        self.perform_clustering_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.perform_clustering_button)

        # Add stretch to push everything to the top
        layout.addStretch(1)
    
    def check_and_show_common_fields(self):
        logs = self.db_manager.get_all_logs()
        if logs:
            # Get common fields from the first log (assuming all logs have the same structure)
            raw_data = json.loads(logs[0][3])  # Assuming raw_data is in the fourth column
            common_fields = list(raw_data.keys())
            self.add_common_fields_checkboxes(common_fields)

    def check_for_existing_logs(self):
        logs = self.db_manager.get_all_logs()
        if logs:
            # Get common fields from the first log (assuming all logs have the same structure)
            raw_data = json.loads(logs[0][3])  # Assuming raw_data is in the fourth column
            common_fields = list(raw_data.keys())
            self.add_common_fields_checkboxes(common_fields)
            self.preprocess_button.show()  # Show the preprocess button
        else:
            self.preprocess_button.hide()  # Hide the preprocess button if no logs


    def create_arrow_label(self):
        return ArrowLabel()


    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Log File", "", "Log Files (*.log);;All Files (*)")
        if file_path:
            self.import_logs([file_path])

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Open Log Folder")
        if folder_path:
            file_paths = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.log') or file.endswith('.json') or file.endswith('.jsonl'):
                        file_paths.append(os.path.join(root, file))
            
            if file_paths:
                self.import_logs(file_paths)
            else:
                QMessageBox.warning(self, "No Files Found", "No suitable log files found in the selected folder.")

    def create_preprocessing_section(self):
        preprocessing_layout = QVBoxLayout()
        
        self.preprocess_button = QPushButton("Preprocess Data")
        self.preprocess_button.clicked.connect(self.preprocess_data)
        self.preprocess_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        preprocessing_layout.addWidget(self.preprocess_button)

        # Create a layout to hold dynamic checkboxes
        self.common_fields_layout = QGridLayout()
        preprocessing_layout.addLayout(self.common_fields_layout)

        return preprocessing_layout
    
    def update_preprocess_button_text(self):
        if self.db_manager.check_preprocessed_text_exists():
            self.preprocess_button.setText("Re-Preprocess Data")
        else:
            self.preprocess_button.setText("Preprocess Data")

    def preprocess_data(self):
        selected_fields = [checkbox.text() for checkbox in self.common_fields_checkboxes if checkbox.isChecked()]
        if not selected_fields:
            QMessageBox.warning(self, "No Fields Selected", "Please select at least one field for preprocessing.")
            return

        reply = QMessageBox.question(self, 'Confirm Preprocessing',
                                    "This will overwrite any existing preprocessed data. Continue?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                    QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            return

        self.status_label.setText("Preprocessing data...")
        self.progress_bar.setValue(0)

        # Get all logs from the database
        logs = self.db_manager.get_all_logs()

        # Call the preprocess_logs function
        preprocess_logs(logs, selected_fields, self.db_manager, self.update_progress, self.update_status)

        self.status_label.setText("Preprocessing completed!")
        self.progress_bar.setValue(100)
        self.update_preprocess_button_text()


    def import_logs(self, file_paths):
        self.import_thread = ImportThread(file_paths, 'log_data.db')
        self.import_thread.progress_update.connect(self.update_progress)
        self.import_thread.status_update.connect(self.update_status)
        self.import_thread.common_fields_found.connect(self.add_common_fields_checkboxes)
        self.import_thread.finished.connect(self.on_import_finished)
        self.import_thread.start()
    
    def add_common_fields_section(self, layout):
        layout.addWidget(QLabel("Common Fields:"))  # Add label for common fields
        for field in self.common_fields:
            checkbox = QCheckBox(field)
            layout.addWidget(checkbox)
            self.common_fields_checkboxes.append(checkbox)
    
    def add_common_fields_checkboxes(self, common_fields):
        # Clear existing checkboxes
        for checkbox in self.common_fields_checkboxes:
            checkbox.setParent(None)
        self.common_fields_checkboxes.clear()

        # Remove existing layout if it exists
        while self.common_fields_layout.count():
            item = self.common_fields_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add checkboxes to the grid layout
        for i, field in enumerate(common_fields):
            checkbox = QCheckBox(field)
            checkbox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            row = i // 3  # Integer division to determine the row
            col = i % 3   # Modulo to determine the column
            self.common_fields_layout.addWidget(checkbox, row, col)
            self.common_fields_checkboxes.append(checkbox)
        
        # Show the preprocessing button
        if hasattr(self, 'preprocess_button'):
            self.preprocess_button.show()
        
    def on_import_finished(self):
        self.populate_tree()
        self.preprocess_button.show()  # Show the preprocess button after import

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, message):
        self.status_label.setText(message)

    def update_generate_embeddings_button(self):
        embeddings_exist = self.db_manager.check_embeddings_exist()
        if embeddings_exist:
            self.generate_embeddings_button.setText("Re-generate Embeddings")
        else:
            self.generate_embeddings_button.setText("Generate Embeddings")

    def generate_embeddings(self):
        embeddings_exist = self.db_manager.check_embeddings_exist()
        
        if embeddings_exist:
            reply = QMessageBox.question(self, 'Warning',
                                         "Existing embeddings will be overwritten. Are you sure you want to continue?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return

        selected_model = self.model_dropdown.currentText()
        self.embedding_thread = EmbeddingGeneratorThread(self.db_manager.db_name, selected_model)
        self.embedding_thread.progress_update.connect(self.update_progress)
        self.embedding_thread.status_update.connect(self.update_status)
        self.embedding_thread.finished.connect(self.on_embedding_generation_finished)
        self.embedding_thread.start()

    def on_embedding_generation_finished(self):
        self.status_label.setText("Embedding generation completed!")
        self.populate_tree()  # Refresh the tree view
        self.update_generate_embeddings_button()  # Update button text
        self.show_visualization()  # Automatically show visualization

    def perform_clustering(self):
        self.clustering_thread = ClusteringThread(self.db_manager.db_name)
        self.clustering_thread.progress_update.connect(self.update_progress)
        self.clustering_thread.status_update.connect(self.update_status)
        self.clustering_thread.finished.connect(self.on_clustering_finished)
        self.clustering_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, message):
        self.status_label.setText(message)

    def on_clustering_finished(self):
        self.status_label.setText("Clustering completed!")
        self.populate_tree()  # Refresh the tree view
        self.show_visualization()  # Automatically show visualization

    def clear_database_and_reset(self):
        reply = QMessageBox.question(self, 'Confirm Clear',
                                    "This will delete all logs and clusters. Are you sure?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                    QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.db_manager.clear_database()
            self.populate_tree()  # Refresh the tree view
            self.visualization.clear_data()  # Clear the visualization
            self.visualization.hide()
            self.status_label.setText("Database cleared and UI reset.")
            self.update_preprocess_button_text()
            self.update_generate_embeddings_button()    

    def show_visualization(self):
        logs = self.db_manager.get_all_logs_with_coordinates()
        
        if not logs:
            self.status_label.setText("No data available for visualization. Please generate embeddings and perform clustering first.")
            return

        points = [(log[2], log[3], log[4]) for log in logs]  # tsne_x, tsne_y, tsne_z
        clusters = [log[1] for log in logs]  # cluster_id
        
        # Get color map
        color_map = {}
        for cluster in self.db_manager.get_clusters():
            cluster_id, _, color = cluster
            color_map[cluster_id] = color
        
        self.visualization.set_data(self.db_manager)
        self.visualization.show()
        self.status_label.setText("Visualization updated successfully.")

    def get_visualization_data(self):
        embeddings = []
        cluster_ids = []
        color_map = {}
        logs = self.db_manager.get_all_logs()
        for log in logs:
            if log[16] is not None:  # Assuming the embedding is in the 16th column
                try:
                    embedding = np.frombuffer(log[16], dtype=np.float32)
                    cluster_id = log[1]  # Assuming cluster_id is in the 2nd column
                    cluster_color = self.db_manager.get_cluster_color(cluster_id)
                    
                    embeddings.append(embedding)
                    cluster_ids.append(cluster_id)
                    color_map[cluster_id] = cluster_color
                except Exception as e:
                    print(f"Error processing embedding for log {log[0]}: {str(e)}")
        
        if not embeddings:
            return None, None, None
        
        embeddings_array = np.array(embeddings)
        
        # Standardize the embeddings
        scaler = StandardScaler()
        embeddings_standardized = scaler.fit_transform(embeddings_array)
        
        return embeddings_standardized, cluster_ids, color_map

    def populate_tree(self):
        self.tree_widget.clear()
        self.tree_widget.setHeaderLabels(["Clusters/Logs", "Visibility", "Details"])
        clusters = self.db_manager.get_cluster_log_counts()
        
        # Set a larger base font size
        base_font = QFont()
        base_font.setPointSize(11)  # Adjust this value as needed
        self.tree_widget.setFont(base_font)
        
        # Set white text color
        white_text = QColor(255, 255, 255)  # Pure white
        
        # Determine common fields
        all_logs = self.db_manager.get_all_logs()
        if all_logs:
            first_log = json.loads(all_logs[0][3])  # Assuming raw_data is in index 3
            common_fields = list(first_log.keys())[:3]  # Get the first 3 fields
        else:
            common_fields = []
        
        for cluster_id, cluster_name, cluster_color, log_count in clusters:
            cluster_item = QTreeWidgetItem(self.tree_widget)
            
            # Create less saturated background color
            bg_color = QColor(cluster_color)
            bg_color.setAlpha(80)  # Increase opacity to 80% for better contrast with white text
            
            if cluster_id == -1:
                cluster_text = f"Noise Cluster: {log_count} logs"
                cluster_item.setData(0, Qt.ItemDataRole.UserRole, None)
            else:
                cluster_text = f"Cluster{cluster_id}: {log_count} logs)"
                cluster_item.setData(0, Qt.ItemDataRole.UserRole, cluster_id)
            
            cluster_item.setText(0, cluster_text)
            cluster_item.setExpanded(False)

            # Set background color and white text color
            for column in range(3):
                cluster_item.setBackground(column, QBrush(bg_color))
                cluster_item.setForeground(column, QBrush(white_text))

            # Set bold and larger font for cluster items
            font = QFont(base_font)
            font.setBold(True)
            font.setPointSize(base_font.pointSize() + 1)  # Make cluster font slightly larger
            cluster_item.setFont(0, font)

            # Create and set visibility checkbox
            checkbox_widget = self.create_visibility_checkbox(cluster_id)
            self.tree_widget.setItemWidget(cluster_item, 1, checkbox_widget)

            if log_count > 0:
                logs = self.db_manager.get_logs_in_cluster(cluster_id)
                for log in logs:
                    log_item = QTreeWidgetItem(cluster_item)
                    log_item.setText(0, f"Log {log[0]}")  # Assuming log[0] is the log ID

                    # Use a slightly darker shade of the cluster color for log items
                    log_bg_color = bg_color.darker(110)
                    for column in range(3):
                        log_item.setBackground(column, QBrush(log_bg_color))
                        log_item.setForeground(column, QBrush(white_text))

                    # Parse the raw_data JSON
                    raw_data = log[3]  # Assuming raw_data is in index 3
                    content = json.loads(raw_data)

                    # Add summary of log details to the "Details" column
                    details = []
                    for field in common_fields:
                        value = content.get(field, 'N/A')
                        # Truncate long values
                        if isinstance(value, str) and len(value) > 30:
                            value = value[:27] + "..."
                        details.append(str(value))
                    summary = " | ".join(details)
                    log_item.setText(2, summary)

                    # Add child items for each detail in the content
                    for key, value in content.items():
                        self.add_detail_item(log_item, key, value)

        # Connect the tree widget's item selection changed signal
        self.tree_widget.itemSelectionChanged.connect(self.on_tree_selection_changed)

    def add_detail_item(self, parent, key, value):
        item = QTreeWidgetItem(parent)
        item.setText(0, key)
        
        # Inherit background color from parent and set white text
        white_text = QColor(255, 255, 255)  # Pure white
        for column in range(3):
            item.setBackground(column, parent.background(column))
            item.setForeground(column, QBrush(white_text))

        if key == "Raw Data":
            try:
                parsed_value = json.loads(value)
                item.setText(2, json.dumps(parsed_value, indent=2))
            except json.JSONDecodeError:
                item.setText(2, str(value))
        else:
            if len(str(value)) > 100:
                item.setText(2, str(value)[:100] + "...")
            else:
                item.setText(2, str(value))

    def on_tree_selection_changed(self):
        selected_items = self.tree_widget.selectedItems()
        if selected_items:
            item = selected_items[0]
            cluster_id = item.data(0, Qt.ItemDataRole.UserRole)
            if cluster_id is not None and cluster_id != -1:
                self.visualization.set_selected_cluster(cluster_id)
            else:
                self.visualization.set_selected_cluster(None)
        else:
            self.visualization.set_selected_cluster(None)
    
    def create_visibility_checkbox(self, cluster_id):
        checkbox = QCheckBox("Visible")
        checkbox.setChecked(True)
        checkbox.stateChanged.connect(lambda state, cid=cluster_id: self.toggle_cluster_visibility(cid, state))
        return checkbox
    
    def check_and_start_visualization(self):
        if self.db_manager.check_embeddings_exist():
            self.show_visualization()
            self.status_label.setText("Visualization started automatically.")
        else:
            self.status_label.setText("No embeddings found. Generate embeddings to see visualization.")

    def toggle_cluster_visibility(self, cluster_id, state):
        is_visible = state == Qt.CheckState.Checked.value
        self.visualization.set_cluster_visibility(cluster_id, is_visible)

    def add_detail_item(self, parent, key, value):
        item = QTreeWidgetItem(parent)
        item.setText(0, key)
        if key in ["Query Params", "Headers", "Cookies", "Form Data", "Files", "JSON"]:
            try:
                parsed_value = json.loads(value)
                item.setText(2, json.dumps(parsed_value, indent=2))
            except json.JSONDecodeError:
                item.setText(2, str(value))
        else:
            item.setText(2, str(value))

def load_custom_font():
    font_path = os.path.join('fonts', 'MesloLGS NF Regular.ttf')
    font_id = QFontDatabase.addApplicationFont(font_path)
    if font_id != -1:
        font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
        custom_font = QFont(font_family)
        QApplication.setFont(custom_font)
    else:
        print("Error: Failed to load the custom font.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    load_custom_font()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
