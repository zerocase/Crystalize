from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GLU import *
from sklearn.manifold import TSNE
import numpy as np
import colorsys

class Visualization3D(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 600)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.points = []
        self.colors = []
        self.clusters = []
        self.highlighted_cluster = None
        self.rotation = [0, 0, 0]
        self.zoom = -5
        self.last_pos = None
        self.cluster_visibility = {}
        self.selected_cluster = None

    @staticmethod
    def hex_to_rgb(hex_color):
        saturation_increase = 0.5
        # Convert hex to RGB
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join(c*2 for c in hex_color)
        if len(hex_color) != 6:
            raise ValueError("Invalid hex color format")
        r, g, b = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
        return (r, g, b)

    def set_data(self, db_manager):
        logs_data = db_manager.get_all_logs_with_coordinates()
        
        if not logs_data:
            print("No data available for visualization.")
            return

        # Extract points, clusters, and log IDs
        points = [(log[2], log[3], log[4]) for log in logs_data]  # tsne_x, tsne_y, tsne_z
        clusters = [log[1] for log in logs_data]  # cluster_id
        self.log_ids = [log[0] for log in logs_data]  # log_id
        
        # Convert points to numpy array
        self.points = np.array(points)
        
        # Normalize to [-1, 1] range
        self.points = (self.points - self.points.min()) / (self.points.max() - self.points.min()) * 2 - 1
        
        self.clusters = clusters
        
        # Get color map from database
        color_map = {cluster[0]: cluster[2] for cluster in db_manager.get_clusters()}
        self.color_map = {k: self.hex_to_rgb(v) for k, v in color_map.items()}
        self.colors = [self.color_map.get(c, (0.5, 0.5, 0.5)) for c in clusters]  # Default to gray if no color found
        
        # Initialize cluster visibility
        self.cluster_visibility = {cluster: True for cluster in set(clusters)}
        
        self.update()

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / height, 0.1, 100.0)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, 0, self.zoom)
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        
        base_size = 5
        point_size = max(base_size, min(50, 2000 / len(self.points)))
        
        # Draw main points
        for point, color, cluster in zip(self.points, self.colors, self.clusters):
            if self.cluster_visibility.get(cluster, True):
                if cluster == self.selected_cluster and cluster != -1:
                    # Scale up the size for selected cluster and slightly reduce opacity
                    glPointSize(point_size * 2.5)  # Increased scaling factor
                    glColor4f(*color, 0.7)  # Slightly reduced opacity
                else:
                    glPointSize(point_size)
                    if cluster == -1:
                        glColor4f(color[0], color[1], color[2], 0.3)  # Lower opacity for noise cluster
                    else:
                        glColor4f(*color, 1.0)
                
                glBegin(GL_POINTS)
                glVertex3f(*point)
                glEnd()

        # Reset OpenGL state
        glPointSize(1.0)
        glColor4f(1.0, 1.0, 1.0, 1.0)

    def set_selected_cluster(self, cluster):
        self.selected_cluster = cluster
        self.update()
    
    def set_cluster_visibility(self, cluster_id, is_visible):
        if cluster_id in self.cluster_visibility:
            self.cluster_visibility[cluster_id] = is_visible
            self.update()  # Trigger a redraw of the visualization

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            self.zoom += 0.1
        else:
            self.zoom -= 0.1
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_pos = event.position()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.MouseButton.LeftButton and self.last_pos is not None:
            dx = event.position().x() - self.last_pos.x()
            dy = event.position().y() - self.last_pos.y()
            self.rotation[0] += dy
            self.rotation[1] += dx
            self.last_pos = event.position()
            self.update()
    
    def clear_data(self):
        self.points = []
        self.colors = []
        self.cluster_ids = []
        self.update()