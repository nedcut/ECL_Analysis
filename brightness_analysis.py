import sys
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui, QtCore
import os

class VideoAnalyzer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Video Brightness Sorcerer')
        self.setAcceptDrops(True)

        # Set a fun, dark theme for the whole app
        self.setStyleSheet("""
            QWidget {
                background-color: #23272e;
                color: #f8f8f2;
                font-family: 'Comic Sans MS', 'Comic Neue', Arial, sans-serif;
                font-size: 15px;
            }
            QLabel#mascotLabel {
                font-size: 32px;
                color: #ffeb3b;
                qproperty-alignment: AlignCenter;
            }
            QLabel {
                color: #f8f8f2;
            }
            QGroupBox {
                border: 2px solid #44475a;
                border-radius: 8px;
                margin-top: 10px;
                background: #282a36;
                font-weight: bold;
                font-size: 17px;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                color: #50fa7b;
            }
            QPushButton {
                background-color: #44475a;
                color: #f8f8f2;
                border: 2px solid #6272a4;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #50fa7b;
                color: #23272e;
                border: 2px solid #f1fa8c;
            }
            QListWidget {
                background: #282a36;
                border: 1px solid #44475a;
                color: #f8f8f2;
                font-size: 15px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #44475a;
                height: 8px;
                background: #44475a;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #ff79c6;
                border: 1px solid #bd93f9;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #50fa7b;
                border-radius: 4px;
            }
            QSlider::add-page:horizontal {
                background: #282a36;
                border-radius: 4px;
            }
        """)

        # Initialize variables
        self.video_path = None
        self.frame = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.cap = None

        # Rectangle management
        self.rects = []  # List of rectangles: [(pt1, pt2), ...]
        self.selected_rect_idx = None  # Index of selected rectangle
        self.drawing = False
        self.moving = False
        self.resizing = False
        self.start_point = None
        self.end_point = None
        self.move_offset = None
        self.resize_corner = None
        self._current_image_size = None

        # Frame range management
        self.start_frame = 0
        self.end_frame = None

        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.left_layout = QtWidgets.QVBoxLayout()

        # Mascot label for personality
        self.mascot_label = QtWidgets.QLabel("🧙 Brightness Sorcerer 🪄", self)
        self.mascot_label.setObjectName("mascotLabel")
        self.left_layout.addWidget(self.mascot_label, stretch=0)

        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.image_label.setStyleSheet("border: 2px solid #ff79c6; background: #181920; border-radius: 12px;")
        self.left_layout.addWidget(self.image_label, stretch=4)
        self._current_image_size = self.image_label.size()

        self.slider_layout = QtWidgets.QHBoxLayout()
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.slider_frame_changed)
        self.slider_layout.addWidget(self.frame_slider)

        self.frame_label = QtWidgets.QLabel("🎞️ Frame: 0 / 0")
        self.frame_label.setMinimumWidth(120)
        self.frame_label.setStyleSheet("font-size: 16px; color: #8be9fd; font-weight: bold;")
        self.slider_layout.addWidget(self.frame_label)
        self.left_layout.addLayout(self.slider_layout, stretch=1)

        # Start/End frame buttons
        btn_frame_layout = QtWidgets.QHBoxLayout()
        self.set_start_btn = QtWidgets.QPushButton("Set Start Frame")
        self.set_start_btn.clicked.connect(self.set_start_frame)
        btn_frame_layout.addWidget(self.set_start_btn)
        self.set_end_btn = QtWidgets.QPushButton("Set End Frame")
        self.set_end_btn.clicked.connect(self.set_end_frame)
        btn_frame_layout.addWidget(self.set_end_btn)
        self.left_layout.addLayout(btn_frame_layout, stretch=0)

        # Analysis name input
        name_layout = QtWidgets.QHBoxLayout()
        name_layout.addWidget(QtWidgets.QLabel("Analysis Name:"))
        self.analysis_name_input = QtWidgets.QLineEdit()
        self.analysis_name_input.setPlaceholderText("MyAnalysis")
        name_layout.addWidget(self.analysis_name_input)
        self.left_layout.addLayout(name_layout, stretch=0)

        self.analyze_btn = QtWidgets.QPushButton('🔍 Analyze Brightness', self)
        self.analyze_btn.clicked.connect(self.analyze_top_decile)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setToolTip("Click me to let BrightBot do the math magic!")
        self.left_layout.addWidget(self.analyze_btn, stretch=1)

        # Plot button
        self.plot_btn = QtWidgets.QPushButton('📈 Plot Results', self)
        self.plot_btn.setEnabled(False)
        self.plot_btn.clicked.connect(self.plot_results)
        self.left_layout.addWidget(self.plot_btn, stretch=1)

        self.main_layout.addLayout(self.left_layout, stretch=3)

        # --- Right Layout (Results & Rectangle Controls) ---
        self.right_layout = QtWidgets.QVBoxLayout()

        self.brightness_groupbox = QtWidgets.QGroupBox("✨ Average Brightness ✨")
        self.brightness_groupbox_layout = QtWidgets.QVBoxLayout()
        self.brightness_display_label = QtWidgets.QLabel("N/A")
        self.brightness_display_label.setAlignment(QtCore.Qt.AlignCenter)
        self.brightness_display_label.setStyleSheet("font-size: 32px; border: 2px solid #f1fa8c; padding: 10px; color: #f1fa8c; background: #181920; border-radius: 10px;")
        self.brightness_groupbox_layout.addWidget(self.brightness_display_label)
        self.brightness_groupbox.setLayout(self.brightness_groupbox_layout)
        self.brightness_groupbox.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.right_layout.addWidget(self.brightness_groupbox)

        # Rectangle List and Controls
        self.rect_groupbox = QtWidgets.QGroupBox("🟩 Rectangles")
        self.rect_groupbox_layout = QtWidgets.QVBoxLayout()
        self.rect_list = QtWidgets.QListWidget()
        self.rect_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.rect_list.currentRowChanged.connect(self.select_rectangle)
        self.rect_groupbox_layout.addWidget(self.rect_list)

        btn_layout = QtWidgets.QHBoxLayout()
        self.add_rect_btn = QtWidgets.QPushButton("➕ Add Rectangle")
        self.add_rect_btn.setCheckable(True)
        self.add_rect_btn.clicked.connect(self.start_add_rectangle)
        btn_layout.addWidget(self.add_rect_btn)
        self.del_rect_btn = QtWidgets.QPushButton("❌ Delete Selected")
        self.del_rect_btn.clicked.connect(self.delete_selected_rectangle)
        btn_layout.addWidget(self.del_rect_btn)
        self.clear_rect_btn = QtWidgets.QPushButton("🧹 Clear All")
        self.clear_rect_btn.clicked.connect(self.clear_all_rectangles)
        btn_layout.addWidget(self.clear_rect_btn)
        self.rect_groupbox_layout.addLayout(btn_layout)
        self.rect_groupbox.setLayout(self.rect_groupbox_layout)
        self.right_layout.addWidget(self.rect_groupbox)

        self.results_label = QtWidgets.QLabel("🦄 Drag & drop a video file to start.\nBrightBot is ready for action!")
        self.results_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.results_label.setWordWrap(True)
        self.results_label.setMaximumWidth(400)
        self.results_label.setStyleSheet("font-size: 15px; color: #bd93f9; background: #181920; border-radius: 8px; padding: 8px;")
        self.results_label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.right_layout.addWidget(self.results_label, stretch=1)

        self.main_layout.addLayout(self.right_layout, stretch=1)

        self.resize(1100, 800)
        self._initial_window_size = self.size()
        self.setFixedSize(self.size())

    def update_rect_list(self):
        self.rect_list.blockSignals(True)
        self.rect_list.clear()
        for idx, (pt1, pt2) in enumerate(self.rects):
            x1, y1 = pt1
            x2, y2 = pt2
            self.rect_list.addItem(f"Rect {idx+1}: ({x1},{y1}) - ({x2},{y2})")
        if self.selected_rect_idx is not None and 0 <= self.selected_rect_idx < len(self.rects):
            self.rect_list.setCurrentRow(self.selected_rect_idx)
        self.rect_list.blockSignals(False)

    def start_add_rectangle(self):
        self.drawing = True
        self.selected_rect_idx = None
        self.add_rect_btn.setChecked(True)                      # highlight button
        self.update_rect_list()
        self.show_frame()

    def select_rectangle(self, idx):
        if 0 <= idx < len(self.rects):
            self.selected_rect_idx = idx
        else:
            self.selected_rect_idx = None
        self.show_frame()

    def delete_selected_rectangle(self):
        if self.selected_rect_idx is not None and 0 <= self.selected_rect_idx < len(self.rects):
            del self.rects[self.selected_rect_idx]
            self.selected_rect_idx = None
            self.update_rect_list()
            self.show_frame()

    def clear_all_rectangles(self):
        self.rects.clear()
        self.selected_rect_idx = None
        self.update_rect_list()
        self.show_frame()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            self.video_path = url.toLocalFile()
            self.load_first_frame()
            break

    def load_first_frame(self):
        if self.cap:
            self.cap.release()
            self.cap = None

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, 'Error', f'Could not open video file: {self.video_path}')
            self.cap = None
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = self.cap.read()

        if ret:
            self.frame = frame
            self.current_frame_index = 0
            self.frame_slider.setEnabled(True)
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(self.total_frames - 1 if self.total_frames > 0 else 0)
            self.frame_slider.setValue(0)
            self.update_frame_label()
            self.show_frame()
            self.analyze_btn.setEnabled(True)
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Could not read the first frame.')
            self.cap.release()
            self.cap = None
            self.frame_slider.setEnabled(False)
            self.analyze_btn.setEnabled(False)
            self.update_frame_label(reset=True)

    def slider_frame_changed(self, value):
        if self.cap and self.cap.isOpened() and value != self.current_frame_index:
            self.current_frame_index = value
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                self.show_frame()
                self.update_frame_label()

    def update_frame_label(self, reset=False):
        if reset or self.total_frames == 0:
            self.frame_label.setText("🎞️ Frame: 0 / 0")
        else:
            self.frame_label.setText(f"🎞️ Frame: {self.current_frame_index + 1} / {self.total_frames}")

    def show_frame(self):
        if self.frame is None:
            self.image_label.setText("No video loaded")
            return
        frame_copy = self.frame.copy()

        # Draw all rectangles
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 128, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 255, 0)
        ]
        for idx, (pt1, pt2) in enumerate(self.rects):
            color = colors[idx % len(colors)]
            thickness = 3 if idx == self.selected_rect_idx else 2
            cv2.rectangle(frame_copy, pt1, pt2, color, thickness)
            # Draw index label
            label = f"{idx+1}"
            cv2.putText(frame_copy, label, (pt1[0]+5, pt1[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Draw current drawing rectangle
        if self.drawing and self.start_point and self.end_point:
            pt1, pt2 = self._map_label_points_to_frame(self.start_point, self.end_point)
            cv2.rectangle(frame_copy, pt1, pt2, (0, 255, 255), 2)

        rgb_image = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)

        target_size = self._current_image_size if self._current_image_size is not None else self.image_label.size()
        scaled_pixmap = pixmap.scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def _map_label_points_to_frame(self, start, end):
        label_size = self.image_label.size()
        pixmap_original_size = self.frame.shape[1::-1]
        temp_pixmap = QtGui.QPixmap(pixmap_original_size[0], pixmap_original_size[1])
        pixmap_scaled_size = temp_pixmap.scaled(label_size, QtCore.Qt.KeepAspectRatio).size()
        offset_x = (label_size.width() - pixmap_scaled_size.width()) / 2
        offset_y = (label_size.height() - pixmap_scaled_size.height()) / 2
        start_x_rel = start.x() - offset_x
        start_y_rel = start.y() - offset_y
        end_x_rel = end.x() - offset_x
        end_y_rel = end.y() - offset_y
        scale_w = self.frame.shape[1] / pixmap_scaled_size.width()
        scale_h = self.frame.shape[0] / pixmap_scaled_size.height()
        x1 = int(min(start_x_rel, end_x_rel) * scale_w)
        y1 = int(min(start_y_rel, end_y_rel) * scale_h)
        x2 = int(max(start_x_rel, end_x_rel) * scale_w)
        y2 = int(max(start_y_rel, end_y_rel) * scale_h)
        frame_h, frame_w = self.frame.shape[:2]
        x1 = max(0, min(x1, frame_w))
        y1 = max(0, min(y1, frame_h))
        x2 = max(0, min(x2, frame_w))
        y2 = max(0, min(y2, frame_h))
        return (x1, y1), (x2, y2)

    def mousePressEvent(self, event):
        if self.frame is not None and event.button() == QtCore.Qt.LeftButton:
            pos = self.image_label.mapFromParent(event.pos())
            if self.image_label.rect().contains(pos):
                pixmap = self.image_label.pixmap()
                if pixmap and not pixmap.isNull():
                    label_rect = self.image_label.rect()
                    pixmap_size = pixmap.size()
                    pixmap_x = int((label_rect.width() - pixmap_size.width()) / 2)
                    pixmap_y = int((label_rect.height() - pixmap_size.height()) / 2)
                    pixmap_rect_in_label = QtCore.QRect(pixmap_x, pixmap_y, pixmap_size.width(), pixmap_size.height())
                    if pixmap_rect_in_label.contains(pos):
                        if self.drawing:
                            self.start_point = pos
                            self.end_point = pos
                        elif self.selected_rect_idx is not None:
                            # Check if near a corner for resizing
                            pt1, pt2 = self.rects[self.selected_rect_idx]
                            corners = [
                                (pt1[0], pt1[1]), (pt2[0], pt1[1]),
                                (pt1[0], pt2[1]), (pt2[0], pt2[1])
                            ]
                            mouse_x, mouse_y = self._map_label_points_to_frame(pos, pos)[0]
                            for i, (cx, cy) in enumerate(corners):
                                if abs(mouse_x - cx) < 10 and abs(mouse_y - cy) < 10:
                                    self.resizing = True
                                    self.resize_corner = i
                                    break
                            else:
                                # Otherwise, start moving
                                self.moving = True
                                self.move_offset = (mouse_x - pt1[0], mouse_y - pt1[1])
                        self.show_frame()
                        return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.start_point:
            pos = self.image_label.mapFromParent(event.pos())
            pos.setX(max(0, min(pos.x(), self.image_label.size().width() - 1)))
            pos.setY(max(0, min(pos.y(), self.image_label.size().height() - 1)))
            self.end_point = pos
            self.show_frame()
        elif self.moving and self.selected_rect_idx is not None:
            pos = self.image_label.mapFromParent(event.pos())
            mouse_x, mouse_y = self._map_label_points_to_frame(pos, pos)[0]
            pt1, pt2 = self.rects[self.selected_rect_idx]
            w = pt2[0] - pt1[0]
            h = pt2[1] - pt1[1]
            new_pt1 = (mouse_x - self.move_offset[0], mouse_y - self.move_offset[1])
            new_pt2 = (new_pt1[0] + w, new_pt1[1] + h)
            frame_h, frame_w = self.frame.shape[:2]
            new_pt1 = (max(0, min(new_pt1[0], frame_w-1)), max(0, min(new_pt1[1], frame_h-1)))
            new_pt2 = (max(0, min(new_pt2[0], frame_w-1)), max(0, min(new_pt2[1], frame_h-1)))
            self.rects[self.selected_rect_idx] = (new_pt1, new_pt2)
            self.update_rect_list()
            self.show_frame()
        elif self.resizing and self.selected_rect_idx is not None:
            pos = self.image_label.mapFromParent(event.pos())
            mouse_x, mouse_y = self._map_label_points_to_frame(pos, pos)[0]
            pt1, pt2 = self.rects[self.selected_rect_idx]
            pts = [list(pt1), [pt2[0], pt1[1]], [pt1[0], pt2[1]], list(pt2)]
            pts[self.resize_corner] = [mouse_x, mouse_y]
            x_coords = [p[0] for p in pts]
            y_coords = [p[1] for p in pts]
            new_pt1 = (min(x_coords), min(y_coords))
            new_pt2 = (max(x_coords), max(y_coords))
            self.rects[self.selected_rect_idx] = (new_pt1, new_pt2)
            self.update_rect_list()
            self.show_frame()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing and event.button() == QtCore.Qt.LeftButton and self.start_point and self.end_point:
            pt1, pt2 = self._map_label_points_to_frame(self.start_point, self.end_point)
            if pt1 != pt2:
                self.rects.append((pt1, pt2))
                self.selected_rect_idx = len(self.rects) - 1
                self.update_rect_list()
            self.drawing = False
            self.start_point = None
            self.end_point = None
            self.add_rect_btn.setChecked(False)                  # clear highlight
            self.show_frame()
        elif self.moving or self.resizing:
            self.moving = False
            self.resizing = False
            self.move_offset = None
            self.resize_corner = None
            self.show_frame()
        else:
            super().mouseReleaseEvent(event)

    def set_start_frame(self):
        self.start_frame = self.current_frame_index
        self.results_label.setText(f"Start frame set to {self.start_frame}")

    def set_end_frame(self):
        self.end_frame = self.current_frame_index
        self.results_label.setText(f"End frame set to {self.end_frame}")

    def analyze_video(self):
        if not self.video_path or not self.rects:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Please select a video and draw at least one rectangle.')
            return

        self.brightness_display_label.setText("...")
        self.results_label.setText("Analyzing...")
        QtWidgets.QApplication.processEvents()

        analysis_cap = None
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            analysis_cap = self.cap
        else:
            analysis_cap = cv2.VideoCapture(self.video_path)

        if not analysis_cap or not analysis_cap.isOpened():
            QtWidgets.QMessageBox.critical(self, 'Error', f'Could not open video file for analysis: {self.video_path}')
            self.results_label.setText("Error opening video.")
            self.brightness_display_label.setText("N/A")
            return

        frame_count = int(analysis_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = self.start_frame
        end = self.end_frame if self.end_frame is not None else frame_count - 1
        if start > end:
            start, end = 0, frame_count - 1

        progress = QtWidgets.QProgressDialog("Analyzing video...", "Cancel", 0, end - start + 1, None)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setWindowTitle("Processing")
        progress.show()
        QtWidgets.QApplication.processEvents()

        analysis_cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frame_idx = start
        processed = 0

        all_brightness = []
        for rect_idx, (x1y1, x2y2) in enumerate(self.rects):
            brightness_list = []
            x1, y1 = x1y1
            x2, y2 = x2y2
            while frame_idx <= end:
                ret, frame = analysis_cap.read()
                if not ret:
                    break
                if progress.wasCanceled():
                    analysis_cap.release()
                    self.results_label.setText("Analysis cancelled.")
                    self.brightness_display_label.setText("N/A")
                    return

                frame_height, frame_width = frame.shape[:2]
                vx1, vy1 = max(0, x1), max(0, y1)
                vx2, vy2 = min(frame_width, x2), min(frame_height, y2)
                if vy1 >= vy2 or vx1 >= vx2:
                    frame_idx += 1
                    if rect_idx == 0:
                        processed += 1
                        progress.setValue(processed)
                        QtWidgets.QApplication.processEvents()
                    continue

                roi = frame[vy1:vy2, vx1:vx2]
                if roi.size == 0:
                    frame_idx += 1
                    if rect_idx == 0:
                        processed += 1
                        progress.setValue(processed)
                        QtWidgets.QApplication.processEvents()
                    continue

                try:
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    bright_pixels = gray[gray >= 10]
                    if bright_pixels.size > 0:
                        brightness = float(np.mean(bright_pixels))
                        brightness_list.append(brightness)
                except cv2.error as e:
                    print(f"OpenCV error processing frame {frame_idx}: {e}")

                frame_idx += 1
                if rect_idx == 0:
                    processed += 1
                    progress.setValue(processed)
                    QtWidgets.QApplication.processEvents()

            all_brightness.append(brightness_list)

        if analysis_cap != self.cap:
            analysis_cap.release()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)

        progress.close()

        if not all_brightness or not all_brightness[0]:
            self.results_label.setText("No valid frames found or analysis cancelled.")
            self.brightness_display_label.setText("N/A")
            return

        avg_brightnesses = [np.mean(b) if b else 0 for b in all_brightness]
        self.brightness_display_label.setText(
            ", ".join([f"R{i+1}: {v:.2f}" for i, v in enumerate(avg_brightnesses)])
        )

        script_dir = os.path.dirname(__file__)
        video_filename = os.path.basename(self.video_path)
        base_video_name, _ = os.path.splitext(video_filename)
        analysis_name = self.analysis_name_input.text().strip() or "MyAnalysis"
        out_paths = []
        for idx, brightness_list in enumerate(all_brightness):
            df = pd.DataFrame({'frame': range(len(brightness_list)), 'brightness': brightness_list})
            out_filename = f"{analysis_name}_{base_video_name}_rect{idx+1}_brightness.csv"
            out_path = os.path.join(script_dir, out_filename)
            try:
                df.to_csv(out_path, index=False)
                out_paths.append(out_path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'Error Saving File', f'Could not save CSV file:\n{e}')

        self.results_label.setText(
            f"Analysis complete.\nRectangles: {len(self.rects)}\n" +
            "\n".join([f"Rect {i+1} Avg: {avg_brightnesses[i]:.2f}" for i in range(len(self.rects))]) +
            "\n\nData saved to:\n" + "\n".join(out_paths)
        )
        self.out_paths = out_paths
        self.plot_btn.setEnabled(True)

    def analyze_top_decile(self):
        """
        Analyze video: for each rectangle, record per-frame
        overall average and top-10% average brightness.
        """
        if not self.video_path or not self.rects:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Please select a video and draw at least one rectangle.')
            return

        self.brightness_display_label.setText("...")
        self.results_label.setText("Analyzing top decile...")
        QtWidgets.QApplication.processEvents()

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            QtWidgets.QMessageBox.critical(self, 'Error', 'Could not open video for top-decile analysis.')
            self.brightness_display_label.setText("N/A")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = self.start_frame
        end = self.end_frame if self.end_frame is not None else total_frames - 1
        if start > end:
            start, end = 0, total_frames - 1

        decile_progress = QtWidgets.QProgressDialog(
            "Analyzing top decile...", "Cancel", 0, end - start + 1, None
        )
        decile_progress.setWindowModality(QtCore.Qt.WindowModal)
        decile_progress.setWindowTitle("Top Decile Processing")
        decile_progress.show()
        QtWidgets.QApplication.processEvents()

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frame_idx = start
        processed = 0

        all_results = []
        for rect_idx, (x1y1, x2y2) in enumerate(self.rects):
            avg_list, decile_list = [], []
            x1, y1 = x1y1
            x2, y2 = x2y2
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frame_idx = start
            while frame_idx <= end:
                ret, frame = cap.read()
                if not ret or decile_progress.wasCanceled():
                    break

                h, w = frame.shape[:2]
                vx1, vy1 = max(0, x1), max(0, y1)
                vx2, vy2 = min(w, x2), min(h, y2)
                if vy2 > vy1 and vx2 > vx1:
                    roi = frame[vy1:vy2, vx1:vx2]
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).flatten()
                    if gray.size:
                        avg_list.append(float(gray.mean()))
                        thresh = np.percentile(gray, 90)
                        top = gray[gray >= thresh]
                        decile_list.append(float(top.mean()) if top.size else 0)
                frame_idx += 1
                if rect_idx == 0:
                    processed += 1
                    decile_progress.setValue(processed)
                    QtWidgets.QApplication.processEvents()

            all_results.append((avg_list, decile_list))

        cap.release()
        decile_progress.close()

        if not all_results or not all_results[0][0]:
            self.results_label.setText("No valid data or cancelled.")
            self.brightness_display_label.setText("N/A")
            return

        script_dir = os.path.dirname(__file__)
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        analysis_name = self.analysis_name_input.text().strip() or "MyAnalysis"
        out_paths = []
        for i, (avgs, deciles) in enumerate(all_results):
            df = pd.DataFrame({
                'frame': range(len(avgs)),
                'avg_brightness': avgs,
                'top10pct_brightness': deciles
            })
            fname = f"{analysis_name}_{base_name}_rect{i+1}_combined.csv"
            path = os.path.join(script_dir, fname)
            df.to_csv(path, index=False)
            out_paths.append(path)

        self.out_paths = out_paths
        self.results_label.setText(f"Combined analysis done. Files:\n" + "\n".join(out_paths))
        self.plot_btn.setEnabled(True)

    def plot_results(self):
        for csv in getattr(self, 'out_paths', []):
            df = pd.read_csv(csv)
            fr = df['frame']
            avg = df['avg_brightness']
            top = df['top10pct_brightness']

            # compute peaks and means
            idx_peak_avg = avg.idxmax()
            idx_peak_top = top.idxmax()
            fr_peak_avg, peak_avg = fr.iloc[idx_peak_avg], avg.iloc[idx_peak_avg]
            fr_peak_top, peak_top = fr.iloc[idx_peak_top], top.iloc[idx_peak_top]
            mean_avg = avg.mean()
            mean_top = top.mean()

            plt.figure(figsize=(8, 4))
            # plot curves
            plt.plot(fr, avg, label='Avg', color='#8b001f')
            plt.plot(fr, top, label='Top10%', color='#50fa7b')
            # horizontal lines for means
            plt.axhline(mean_avg, color='#8b001f', linestyle='--', label=f'Mean Avg ({mean_avg:.1f})')
            plt.axhline(mean_top, color='#50fa7b', linestyle='--', label=f'Mean Top10% ({mean_top:.1f})')
            # mark and annotate peaks
            plt.scatter([fr_peak_avg], [peak_avg], color='#8b001f', zorder=5)
            plt.annotate(f'Peak Avg\n({fr_peak_avg}, {peak_avg:.1f})',
                         xy=(fr_peak_avg, peak_avg),
                         xytext=(0, 10),
                         textcoords='offset points',
                         ha='center',
                         color='#8b001f')
            plt.scatter([fr_peak_top], [peak_top], color='#50fa7b', zorder=5)
            plt.annotate(f'Peak Top10%\n({fr_peak_top}, {peak_top:.1f})',
                         xy=(fr_peak_top, peak_top),
                         xytext=(0, -15),
                         textcoords='offset points',
                         ha='center',
                         color='#50fa7b')

            plt.title(os.path.basename(csv))
            plt.xlabel('Frame')
            plt.ylabel('Brightness')
            plt.legend()
            plt.tight_layout()

            plot_path = csv.replace('.csv', '_combined_plot.png')
            plt.savefig(plot_path)
            plt.show()

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        super().closeEvent(event)

    def resizeEvent(self, event):
        self._current_image_size = self.image_label.size()
        self.show_frame()
        super().resizeEvent(event)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = VideoAnalyzer()
    win.show()
    sys.exit(app.exec_())