import json
import numpy as np
import cv2
import os
import moviepy.editor as mpe
import subprocess
import sys
import numpy as np
from datetime import datetime, timedelta
import datetime
import librosa
from scenedetect import SceneManager, open_video, ContentDetector
import scenedetect
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QSlider, QLabel, QStyle, QSizePolicy, QFileDialog, QTreeWidget, QTreeWidgetItem
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QPalette
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer

class InteractiveMediaPlayer(QWidget):
    def __init__(self, video_path, audio_path,json_file_name):
        super().__init__()

        self.scene_threshold = 50
        self.shot_threshold = 35
        self.min_scene_length = 60
        self.min_shot_length = 50

        self.rgb_video_path = rgb_video_path
        self.audio_path = audio_path
        self.dict_of_start_times = {}

        self.setWindowTitle("Interactive Media Player")
        self.setGeometry(0, 0, 1600, 900)

        self.section_map = {}
        self.reverse_map = {}
        self.object_map = {}   

        self.width = 480                     
        self.height = 270                   
        self.fps = 30 

        self.output_video = './Data/video_no_audio.mp4'
        self.video_path = './Data/OutputVideo.mp4'

        print("Making Mp4 Video")
        self.create_video_from_rgb_file(self.rgb_video_path, self.output_video, self.width, self.height, self.fps)
        self.add_audio_to_video(self.output_video, self.audio_path, self.video_path)

        print("Video Successfully Created")
        print("Dividing Video into Scenes, Shots and SubShots")
        self.preprocess_data()
        json_file = open(os.path.join(dirname,'times.json'))
        self.data = json.load(json_file)

        print("Displaying Video")
        self.setup_ui()

    def create_video_from_rgb_file(self,rgb_file, output_video, width, height, fps):
        with open(rgb_file, 'rb') as f:
            raw_data = f.read()

        num_frames = len(raw_data) // (width * height * 3)
        video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for i in range(num_frames):
            start_index = i * width * height * 3
            end_index = start_index + width * height * 3
            frame_data = np.frombuffer(raw_data[start_index:end_index], dtype=np.uint8)
            frame = frame_data.reshape(height, width, 3)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

        video_writer.release()
    
    def add_audio_to_video(self,video_file, audio_file, output_file):
        command = f"ffmpeg -i {video_file} -i {audio_file} -c:v copy -c:a aac -strict experimental {output_file}"
        subprocess.call(command, shell=True)


    def preprocess_data(self):
        def find_scenes(video_path):
            video = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(
                ContentDetector(threshold=self.scene_threshold, min_scene_len=self.min_scene_length))
           
            scene_manager.detect_scenes(video)
            
            return scene_manager.get_scene_list()

        scenes = find_scenes(self.video_path)
        detector = scenedetect.detectors.ContentDetector(threshold=self.shot_threshold, min_scene_len=self.min_shot_length)

        for scene in scenes:
            scene_start = scene[0].get_timecode()
            scene_end = scene[1].get_timecode()
            
            shots = scenedetect.detect(self.video_path, detector=detector, start_time=scene_start,end_time=scene[1].get_timecode())
            
            filtered_shots = [shot for shot in shots if shot[0].get_timecode() != shot[1].get_timecode()]
            self.dict_of_start_times[scene_start] = {}
            for shot in filtered_shots:
                shot_start = shot[0].get_timecode()
                shot_end = shot[1].get_timecode()
                start_delta = datetime.datetime.strptime(shot_start, "%H:%M:%S.%f") - datetime.datetime.strptime("00:00:00.000", "%H:%M:%S.%f")
                end_delta = datetime.datetime.strptime(shot_end, "%H:%M:%S.%f") - datetime.datetime.strptime("00:00:00.000","%H:%M:%S.%f")
                duration = (end_delta - start_delta).total_seconds()
                
                waveform, sample_rate = librosa.load(self.audio_path, sr=None, offset=start_delta.total_seconds(),duration=duration)
                
                diff = np.abs(np.diff(waveform))

                delta_threshold = np.sqrt(np.mean(np.square(waveform))) * 2.5

                start_stop_times = []

                start = 0
                for i in range(1, len(diff)):
                    if diff[i] > delta_threshold:
                        stop = i
                        if (stop - start) / sample_rate > 1:  
                            start_stop_times.append((start / sample_rate + start_delta.total_seconds(),stop / sample_rate + start_delta.total_seconds()))
                            start = i

                if start < len(waveform):
                    stop = len(waveform)
                    if (stop - start) / sample_rate > 1: 
                        start_stop_times.append((start / sample_rate + start_delta.total_seconds(),stop / sample_rate + start_delta.total_seconds()))

                for tuple in start_stop_times:
                    list_of_subshots = [t[0] for t in start_stop_times]
                    self.dict_of_start_times[scene_start][shot_start] = {}
                    self.dict_of_start_times[scene_start][shot_start] = list_of_subshots

        with open('times.json', 'w') as fp:
            json.dump(self.dict_of_start_times, fp, indent=4)

    def setup_ui(self):
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        video_widget = QVideoWidget()

        self.play_pause_btn = QPushButton()
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.setText("Play")
        self.play_pause_btn.setStyleSheet("color: black")
        self.play_pause_btn.clicked.connect(self.play_video)

        self.stop_btn = QPushButton()
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("Stop")
        self.stop_btn.setStyleSheet("color: black")
        self.stop_btn.clicked.connect(self.stop_video)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position)
        self.slider.setStyleSheet("QSlider::groove:horizontal {background: #808080; height: 6px;}"
                                  "QSlider::handle:horizontal {background: #FF0000; width: 18px; margin: -6px 0;}")

        self.label = QLabel()
        self.label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(1)
        self.populate_tree_widget()
        self.tree.itemClicked.connect(self.onItemClicked)

        self.tree.setAutoFillBackground(True)
        tree_palette = self.tree.palette()
        tree_palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#FFFFFF"))
        self.tree.setPalette(tree_palette)

        hboxlayout = QHBoxLayout()
        hboxlayout.setContentsMargins(0, 0, 0, 0)
        hboxlayout.addWidget(self.slider)
        hboxlayout.addWidget(self.play_pause_btn)
        hboxlayout.addWidget(self.stop_btn)

        vboxlayout = QVBoxLayout()
        vboxlayout.addWidget(video_widget)
        vboxlayout.addLayout(hboxlayout)
        vboxlayout.addWidget(self.label)

        hboxrootlayout = QHBoxLayout()
        hboxrootlayout.addWidget(self.tree, 2)
        hboxrootlayout.addLayout(vboxlayout, 8)

        self.setLayout(hboxrootlayout)
        self.media_player.setVideoOutput(video_widget)
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, 'Data', "OutputVideo.mp4")
        # path = "/Users/bhavengore/Desktop/Multimedia/Project/Project-Multimedia/VideoPlayer/Data/OutputVideo.mp4"
        if os.path.isfile(path):
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
            self.play_pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
        else:
            print("Error: Video file not found")

        self.media_player.positionChanged.connect(self.slider_position_changed)
        self.media_player.durationChanged.connect(self.slider_duration_changed)
        self.media_player.error.connect(self.handle_errors)
        self.media_player.mediaStatusChanged.connect(self.media_status_changed)  

    def convert_time_to_milli(self,time_component):
        return ((float(time_component[0]) * 60 + float(time_component[1])) * 60 + float(time_component[2])) * 1000

    def handle_errors(self):
            self.play_pause_btn.setEnabled(False)
            self.label.setText("Error: " + self.media_player.errorString())

    def media_status_changed(self, status):
        if status == QMediaPlayer.BufferedMedia or status == QMediaPlayer.LoadedMedia:
            self.play_video()

    def play_video(self):

        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_pause_btn.setText("Play")
            self.play_pause_btn.setStyleSheet("color: black")
        else:
            self.media_player.play()
            self.play_pause_btn.setText("Pause")
            self.play_pause_btn.setStyleSheet("color: black")

    def stop_video(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            position = self.media_player.position()
            break_point = 'Scene1'
            for key in self.reverse_map.keys():
                if position <= key:
                    break
                else:
                    break_point = self.reverse_map[key]

            new_pos = self.section_map[break_point.split(' ')[0]]
            self.media_player.setPosition(int(new_pos))
            self.media_player.pause()
            self.play_pause_btn.setText("Play")
            self.play_pause_btn.setStyleSheet("color: black")

    def slider_position_changed(self, position):
        self.slider.setValue(position)
        self.paint_item(position)

    def paint_item(self, position):
        for timestamp in self.object_map.keys():
            if timestamp <= position:
                for item in self.object_map.get(timestamp):
                    item.setForeground(0, QtGui.QBrush(QtGui.QColor("#FF0000")))
            else:
                for item in self.object_map.get(timestamp):
                    item.setForeground(0, QtGui.QBrush(QtGui.QColor("#808080")))

    def slider_duration_changed(self, duration):
        self.slider.setRange(0, duration)

    def set_position(self, position):
        self.media_player.setPosition(int(position))

    def insert_into_object_map(self, timestamp, item):
        if timestamp not in self.object_map.keys():
            self.object_map[timestamp] = [item]
        else:
            self.object_map[timestamp].append(item)

    def populate_tree_widget(self):

        parents = [k for k in self.data.keys()]
        time_format = "%H:%M:%S.%f"
        idx = 1
        for scene in parents:
            time_component = scene.split(':')
            time_milli = self.convert_time_to_milli(time_component)

            key1 = 'Scene' +" "+ str(idx)
            self.section_map[key1] = time_milli
            self.reverse_map[time_milli] = key1

            parent_it = QTreeWidgetItem([key1])
            self.insert_into_object_map(time_milli, parent_it)
            idx1 = 1
            self.tree.addTopLevelItem(parent_it)
            for shot in self.data[scene]:
                time_component = shot.split(':')
                time_milli = self.convert_time_to_milli(time_component)
                # key2 = key1 + 'Shot' + str(idx1)
                key2 = "Shot" + " " +str(idx1) + " "+ str(idx)
                self.section_map[key2] = time_milli
                self.reverse_map[time_milli] = key2

                it = QTreeWidgetItem([key2])
                self.insert_into_object_map(time_milli, it)

                idx1 += 1
                idx2 = 1
                parent_it.addChild(it)
                for subshot in self.data[scene][shot]:
                    time_milli = float(subshot) * 1000
                    # key3 = key2 + ' Subshot' + str(idx2)
                    key3 = "Subshot" + " " + str(idx2) + " "+ str(idx1-1) +" "+ str(idx)
                    self.section_map[key3] = time_milli
                    self.reverse_map[time_milli] = key3

                    sit = QTreeWidgetItem([key3])
                    self.insert_into_object_map(time_milli, sit)
                    it.addChild(sit)
                    idx2 += 1

            idx += 1

        self.tree.expandAll()

    @QtCore.pyqtSlot(QtWidgets.QTreeWidgetItem, int)
    def onItemClicked(self, it, col):
        key = it.text(col)
        if key in self.section_map:
            self.set_position(self.section_map[key])


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    rgb_video = sys.argv[1]
    audio = sys.argv[2]
    
    audio_path = os.path.join(dirname, 'Data', audio)
    rgb_video_path = os.path.join(dirname, 'Data', rgb_video)

    app = QApplication(sys.argv)
    window = InteractiveMediaPlayer(rgb_video_path, audio_path,json_file_name='dict_of_start_times.json')

    palette = app.palette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#FFFFFF"))
    app.setPalette(palette)
    window.show()
    sys.exit(app.exec_())
