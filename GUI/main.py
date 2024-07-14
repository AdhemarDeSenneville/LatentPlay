import sys
import os
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # type: ignore

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel, QGraphicsView, QGraphicsScene
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QPainterPath, QTransform, QPen
from PyQt5 import QtCore, QtGui, QtWidgets

from audio import LatentPlayGenerator
from deep_ae import TimeFrequencyLoss

MODEL_PATH = r'models'
DATASET_PATH = r'Dataset\kick_dataset'

class UpdateThread(QThread):
    resultReady = pyqtSignal(np.ndarray)

    def __init__(self, model, p1, p2, p3, p4, p5):
        super().__init__()
        self.model = model
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5

    def run(self):
        result = self.model.decode(self.p1, self.p2, self.p3, self.p4, self.p5)
        self.resultReady.emit(result)


class Ui_Dialog(QMainWindow):
    signalChanged = pyqtSignal(int, int, int, int, int)

    def __init__(self, dataset_path, model_path):
        super(Ui_Dialog, self).__init__()
        
        print("INFO : App started")
        self.bpm = 128
        self.gain = 1
        self.dry = 0
        self.control_precision = 1000
        self.notic_padding = 4000

        self.dataset_path = dataset_path
        self.model_path = model_path

        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.emit_signal)
        
        self.signalChanged.connect(self.update_plot)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.play_sound)

        print("INFO : App initialized")

    def setupUi(self, Dialog):
        print("setupUi")
        Dialog.setObjectName("Dialog")
        Dialog.resize(853, 651)
        Dialog.setFixedSize(853, 651)
        Dialog.setWindowTitle("LatentPlay VST")

        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 85, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 115, 90))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 85, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Highlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.HighlightedText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 85, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Link, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 85, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(159, 159, 159, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 85, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 115, 90))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 85, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Highlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.HighlightedText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 85, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Link, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 85, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 210, 210, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 115, 90))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 115, 90))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 120, 215))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Highlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.HighlightedText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Link, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.LinkVisited, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.PlaceholderText, brush)
        palette_2 = palette

        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 234, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 210, 210))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(26, 26, 26))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(215, 0, 204))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Highlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Link, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 210, 210, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 234, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 210, 210))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(26, 26, 26))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(215, 0, 204))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Highlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Link, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 210, 210, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(26, 26, 26))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(26, 26, 26))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 120, 215))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Highlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Link, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.PlaceholderText, brush)

        Dialog.setPalette(palette)
        Dialog.setWindowOpacity(1.0)

        font9 = QtGui.QFont()
        font9.setFamily("Rubik")
        font9.setPointSize(9)
        font9.setBold(True)
        font9.setWeight(75)

        font10 = QtGui.QFont()
        font10.setFamily("Rubik")
        font10.setPointSize(10)
        font10.setBold(True)
        font10.setWeight(75)

        font12 = QtGui.QFont()
        font12.setFamily("Rubik")
        font12.setPointSize(12)
        font12.setBold(True)
        font12.setWeight(75)

        font14 = QtGui.QFont()
        font14.setFamily("Rubik")
        font14.setPointSize(14)
        font14.setBold(True)
        font14.setWeight(75)

        self.checkBox_autorun = QtWidgets.QCheckBox(Dialog)
        self.checkBox_autorun.setGeometry(QtCore.QRect(20, 220, 101, 51))
        self.checkBox_autorun.stateChanged.connect(self.toggle_timer)
        self.checkBox_autorun.setPalette(palette)
        self.checkBox_autorun.setFont(font12)
        self.checkBox_autorun.setObjectName("checkBox_autorun")

        self.verticalLayoutWidget_2 = QtWidgets.QWidget(Dialog)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(440, 160, 391, 155))
        self.verticalLayoutWidget_2.setPalette(palette)
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")

        self.Latent_Slider_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.Latent_Slider_2.setContentsMargins(0, 0, 0, 0)
        self.Latent_Slider_2.setObjectName("Latent_Slider_2")

        self.cstm_label_freq = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.cstm_label_freq.setPalette(palette)
        self.cstm_label_freq.setFont(font9)
        self.cstm_label_freq.setAlignment(QtCore.Qt.AlignCenter)
        self.cstm_label_freq.setObjectName("cstm_label_freq")
        self.Latent_Slider_2.addWidget(self.cstm_label_freq)
        self.Freq_Slider = QtWidgets.QSlider(self.verticalLayoutWidget_2)
        
        self.Freq_Slider.setPalette(palette)
        self.Freq_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.Freq_Slider.setObjectName("Freq_Slider")
        self.Freq_Slider.setRange(0, self.control_precision)
        self.Freq_Slider.setValue(10)
        self.Freq_Slider.valueChanged.connect(self.slider_changed)
        self.Latent_Slider_2.addWidget(self.Freq_Slider)
        self.cstm_label_attack = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        
        self.cstm_label_attack.setPalette(palette)
        self.cstm_label_attack.setFont(font9)
        self.cstm_label_attack.setAlignment(QtCore.Qt.AlignCenter)
        self.cstm_label_attack.setObjectName("cstm_label_attack")
        self.Latent_Slider_2.addWidget(self.cstm_label_attack)
        self.Attack_Slider = QtWidgets.QSlider(self.verticalLayoutWidget_2)
        
        self.Attack_Slider.setPalette(palette)
        self.Attack_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.Attack_Slider.setObjectName("Attack_Slider")
        self.Attack_Slider.setRange(0, self.control_precision)
        self.Attack_Slider.setValue(10)
        self.Attack_Slider.valueChanged.connect(self.slider_changed)
        self.Latent_Slider_2.addWidget(self.Attack_Slider)
        self.cstm_label_attack_2 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        
        self.cstm_label_attack_2.setPalette(palette)
        self.cstm_label_attack_2.setFont(font9)
        self.cstm_label_attack_2.setAlignment(QtCore.Qt.AlignCenter)
        self.cstm_label_attack_2.setObjectName("cstm_label_attack_2")
        self.Latent_Slider_2.addWidget(self.cstm_label_attack_2)

        self.Release_Slider = QtWidgets.QSlider(self.verticalLayoutWidget_2)
        self.Release_Slider.setPalette(palette)
        self.Release_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.Release_Slider.setObjectName("Release_Slider")
        self.Release_Slider.setRange(0, self.control_precision)
        self.Release_Slider.setValue(10)
        self.Release_Slider.valueChanged.connect(self.slider_changed)
        self.Latent_Slider_2.addWidget(self.Release_Slider)

        self.pca_title = QtWidgets.QLabel(Dialog)
        self.pca_title.setGeometry(QtCore.QRect(180, 130, 201, 41))
        self.pca_title.setPalette(palette)
        self.pca_title.setFont(font14)
        self.pca_title.setAlignment(QtCore.Qt.AlignCenter)
        self.pca_title.setObjectName("pca_title")

        self.cstm_title = QtWidgets.QLabel(Dialog)
        self.cstm_title.setGeometry(QtCore.QRect(440, 120, 391, 41))
        self.cstm_title.setPalette(palette)
        self.cstm_title.setFont(font14)
        self.cstm_title.setAlignment(QtCore.Qt.AlignCenter)
        self.cstm_title.setObjectName("cstm_title")

        self.line = QtWidgets.QFrame(Dialog)
        self.line.setGeometry(QtCore.QRect(140, 110, 711, 20))
        self.line.setPalette(palette)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(Dialog)
        self.line_2.setGeometry(QtCore.QRect(0, 320, 861, 16))
        self.line_2.setPalette(palette)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")

        self.comboBox = QtWidgets.QComboBox(Dialog)
        self.comboBox.setGeometry(QtCore.QRect(180, 30, 441, 31))
        self.comboBox.setPalette(palette_2)
        self.comboBox.setObjectName("comboBox")

        self.pca_label_1 = QtWidgets.QLabel(Dialog)
        self.pca_label_1.setGeometry(QtCore.QRect(170, 290, 91, 20))
        self.pca_label_1.setPalette(palette)
        self.pca_label_1.setFont(font10)
        self.pca_label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.pca_label_1.setObjectName("pca_label_1")
        self.pca_label_2 = QtWidgets.QLabel(Dialog)
        self.pca_label_2.setGeometry(QtCore.QRect(290, 290, 91, 20))
        self.pca_label_2.setPalette(palette)
        self.pca_label_2.setFont(font10)
        self.pca_label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.pca_label_2.setObjectName("pca_label_2")

        self.line_3 = QtWidgets.QFrame(Dialog)
        self.line_3.setGeometry(QtCore.QRect(130, 0, 21, 331))
        self.line_3.setPalette(palette)
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")

        self.label_8 = QtWidgets.QLabel(Dialog)
        self.label_8.setGeometry(QtCore.QRect(20, 260, 101, 41))
        self.label_8.setPalette(palette)
        self.label_8.setFont(font10)
        self.label_8.setObjectName("label_8")

        self.load_title = QtWidgets.QLabel(Dialog)
        self.load_title.setGeometry(QtCore.QRect(160, -1, 431, 31))
        self.load_title.setPalette(palette)
        self.load_title.setFont(font14)
        self.load_title.setAlignment(QtCore.Qt.AlignCenter)
        self.load_title.setObjectName("load_title")

        self.pca_dial_2 = QtWidgets.QDial(Dialog)
        self.pca_dial_2.setGeometry(QtCore.QRect(290, 180, 91, 91))
        self.pca_dial_2.setPalette(palette)
        self.pca_dial_2.setObjectName("pca_dial_2")
        self.pca_dial_2.setRange(-self.control_precision, self.control_precision)
        self.pca_dial_2.setValue(0)
        self.pca_dial_2.valueChanged.connect(self.slider_changed)

        self.pca_dial_1 = QtWidgets.QDial(Dialog)
        self.pca_dial_1.setGeometry(QtCore.QRect(170, 180, 91, 91))
        self.pca_dial_1.setPalette(palette)
        self.pca_dial_1.setObjectName("pca_dial_1")
        self.pca_dial_1.setRange(-self.control_precision, self.control_precision)
        self.pca_dial_1.setValue(0)
        self.pca_dial_1.valueChanged.connect(self.slider_changed)

        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(20, 110, 101, 101))
        self.pushButton.setPalette(palette_2)
        font = QtGui.QFont()
        font.setFamily("Rubik")
        font.setPointSize(48)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.play_sound)

        self.line_4 = QtWidgets.QFrame(Dialog)
        self.line_4.setGeometry(QtCore.QRect(620, 0, 16, 121))
        self.line_4.setPalette(palette)
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.line_5 = QtWidgets.QFrame(Dialog)
        self.line_5.setGeometry(QtCore.QRect(400, 160, 20, 131))
        self.line_5.setPalette(palette)
        self.line_5.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.line_6 = QtWidgets.QFrame(Dialog)
        self.line_6.setGeometry(QtCore.QRect(30, 210, 81, 31))
        self.line_6.setPalette(palette)
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.line_7 = QtWidgets.QFrame(Dialog)
        self.line_7.setGeometry(QtCore.QRect(0, 60, 141, 20))
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")

        self.out_title = QtWidgets.QLabel(Dialog)
        self.out_title.setGeometry(QtCore.QRect(636, -1, 201, 31))
        self.out_title.setPalette(palette)
        self.out_title.setFont(font14)
        self.out_title.setAlignment(QtCore.Qt.AlignCenter)
        self.out_title.setObjectName("out_title")

        self.play_title = QtWidgets.QLabel(Dialog)
        self.play_title.setGeometry(QtCore.QRect(20, 70, 101, 41))
        self.play_title.setPalette(palette)
        self.play_title.setFont(font14)
        self.play_title.setAlignment(QtCore.Qt.AlignCenter)
        self.play_title.setObjectName("play_title")

        self.comboBox_2 = QtWidgets.QComboBox(Dialog)
        self.comboBox_2.setGeometry(QtCore.QRect(200, 70, 421, 31))
        self.comboBox_2.setPalette(palette_2)
        self.comboBox_2.currentIndexChanged.connect(self.load_sample)
        self.comboBox_2.setObjectName("comboBox_2")

        self.load_pack = QtWidgets.QLabel(Dialog)
        self.load_pack.setGeometry(QtCore.QRect(146, 29, 51, 31))
        self.load_pack.setPalette(palette)
        self.load_pack.setFont(font9)
        self.load_pack.setObjectName("load_pack")

        self.load_sample_ = QtWidgets.QLabel(Dialog)
        self.load_sample_.setGeometry(QtCore.QRect(146, 72, 51, 31))
        self.load_sample_.setPalette(palette)
        self.load_sample_.setFont(font9)
        self.load_sample_.setObjectName("load_sample")

        self.out_dial_dry = QtWidgets.QDial(Dialog)
        self.out_dial_dry.setGeometry(QtCore.QRect(660, 30, 61, 64))
        self.out_dial_dry.setPalette(palette)
        self.out_dial_dry.setObjectName("out_dial_dry")
        self.out_dial_dry.setRange(0, self.control_precision)
        self.out_dial_dry.setValue(self.control_precision)
        self.out_dial_dry.valueChanged.connect(self.dry_changed)

        self.out_dial_gain = QtWidgets.QDial(Dialog)
        self.out_dial_gain.setGeometry(QtCore.QRect(760, 30, 61, 64))
        self.out_dial_gain.setPalette(palette)
        self.out_dial_gain.setObjectName("out_dial_gain")
        self.out_dial_gain.setRange(-20, 20)
        self.out_dial_gain.setValue(0)
        self.out_dial_gain.valueChanged.connect(self.gain_changed)

        self.out_label_dry = QtWidgets.QLabel(Dialog)
        self.out_label_dry.setGeometry(QtCore.QRect(650, 90, 100, 31))
        self.out_label_dry.setPalette(palette)
        self.out_label_dry.setFont(font9)
        #self.out_label_dry.setAlignment(QtCore.Qt.AlignCenter)
        self.out_label_dry.setObjectName("out_label_dry")

        self.out_label_gain = QtWidgets.QLabel(Dialog)
        self.out_label_gain.setGeometry(QtCore.QRect(760, 90, 100, 31))
        self.out_label_gain.setPalette(palette)
        self.out_label_gain.setFont(font9)
        #self.out_label_gain.setAlignment(QtCore.Qt.AlignCenter)
        self.out_label_gain.setObjectName("out_label_gain")

        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(30, 5, 81, 21))
        self.label.setFont(font14)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")

        self.comboBox_Model = QtWidgets.QComboBox(Dialog)
        self.comboBox_Model.setPalette(palette_2)
        self.comboBox_Model.setGeometry(QtCore.QRect(8, 30, 121, 31))
        self.comboBox_Model.setObjectName("comboBox_Model")

        self.BPM_Slider = QtWidgets.QSlider(Dialog)
        self.BPM_Slider.setGeometry(QtCore.QRect(20, 290, 101, 22))
        self.BPM_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.BPM_Slider.setObjectName("BPM_Slider")
        self.BPM_Slider.setRange(50, 200)
        self.BPM_Slider.setValue(128)
        self.BPM_Slider.valueChanged.connect(self.bpm_changed)

        self.graphicsView = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(10, 340, 830, 300))
        self.graphicsView.setObjectName("graphicsView")
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        print("INFO : UI set up")

    def retranslateUi(self, Dialog):

        print("retranslateUi")
        _translate = QtCore.QCoreApplication.translate
        self.checkBox_autorun.setText(_translate("Dialog", "Auto Run"))
        self.cstm_label_freq.setText(_translate("Dialog", "Frequency"))
        self.cstm_label_attack.setText(_translate("Dialog", "Attack"))
        self.cstm_label_attack_2.setText(_translate("Dialog", "Release"))
        self.pca_title.setText(_translate("Dialog", "PCA Control"))
        self.cstm_title.setText(_translate("Dialog", "Custom Control"))
        
        folders = os.listdir(self.model_path)    # Populate the comboBox with the folder names
        for index, folder_name in enumerate(folders):
            self.comboBox_Model.addItem("")
            self.comboBox_Model.setItemText(index, _translate("Dialog", folder_name))
        self.comboBox_Model.currentIndexChanged.connect(self.load_model)
        self.comboBox_Model.setCurrentIndex(1)
        self.load_model()

        folders = os.listdir(self.dataset_path)    # Populate the comboBox with the folder names
        for index, folder_name in enumerate(folders):
            self.comboBox.addItem("")
            self.comboBox.setItemText(index, _translate("Dialog", folder_name))
        
        
        self.comboBox.currentIndexChanged.connect(self.populate_combobox2)
        self.comboBox.setCurrentIndex(0)
        self.populate_combobox2()
        print("INFO : Default pack", self.comboBox.currentText())
        
        
        self.pca_label_1.setText(_translate("Dialog", "Latent 1"))
        self.pca_label_2.setText(_translate("Dialog", "Latent 2"))
        self.label_8.setText(_translate("Dialog", "BPM"))
        self.load_title.setText(_translate("Dialog", "Load"))
        self.pushButton.setText(_translate("Dialog", "|>"))
        self.out_title.setText(_translate("Dialog", "Output"))
        self.play_title.setText(_translate("Dialog", "Play"))
        self.load_pack.setText(_translate("Dialog", "Pack"))
        self.load_sample_.setText(_translate("Dialog", "Sample"))
        self.out_label_dry.setText(_translate("Dialog", "Dry / Wet"))
        self.out_label_gain.setText(_translate("Dialog", "Gain"))
        self.label.setText(_translate("Dialog", "Model"))
        self.dry_changed()
        self.gain_changed()

        print("INFO : Text loaded")

    def bpm_changed(self):
        self.bpm = self.BPM_Slider.value()
    
    def gain_changed(self):
        self.gain = self.out_dial_gain.value()
        self.out_label_gain.setText(f"Gain : {self.gain}dB")
        self.slider_changed()
    
    def dry_changed(self):
        self.dry = self.out_dial_dry.value()
        self.out_label_dry.setText(f"Dry/Wet : {int(100*self.dry/self.control_precision)}%")
        self.slider_changed()

    def emit_signal(self):
        p_latent_1 = self.pca_dial_1.value()
        p_latent_2 = self.pca_dial_2.value()
        p_freq = self.Freq_Slider.value()
        p_att = self.Attack_Slider.value()
        p_rel = self.Release_Slider.value()
        
        self.signalChanged.emit(p_latent_1, p_latent_2, p_freq, p_att, p_rel)
    
    def slider_changed(self):
        # Restart the debounce timer
        if self.debounce_timer.isActive():
            self.debounce_timer.stop()
        self.debounce_timer.start(200)  # Set debounce delay to 200 ms

    def update_plot(self, p1, p2, p3, p4, p5):
        self.update_thread = UpdateThread(self.model, 
            p1/self.control_precision, 
            p2/self.control_precision, 
            p3/self.control_precision, 
            p4/self.control_precision, 
            p5/self.control_precision)
        self.update_thread.resultReady.connect(self.on_update_result)
        self.update_thread.start()

    def on_update_result(self, result):
        self.sound_data = result
        self.post_pro_audio()
        self.scene.clear()

        # Optimize to draw a reduced number of points
        nb_points = len(self.sound) - self.notic_padding
        path = QPainterPath()
        path.moveTo(0, 0)
        for i in range(0, len(self.sound) - self.notic_padding,):
            path.lineTo(i/nb_points * 830 , 150*self.sound[i] / 10 ** (self.gain / 20))

        # Create a pen with the desired stroke width
        pen = QPen(Qt.black)  # Set the color to black or any other color you prefer
        pen.setWidth(2)  # Set the stroke width to 1 or any other desired width

        # Add the path with the specified pen to the scene
        self.scene.addPath(path, pen)
        self.update_view_transform()

    def resizeEvent(self, event):
        super(Ui_Dialog, self).resizeEvent(event)
        self.update_view_transform()

    def update_view_transform(self):
        

        view_range = 150 #* 10 ** (self.gain / 20)
        self.scene.setSceneRect(0, -view_range, len(self.sound) - self.notic_padding, 2 * view_range)
    
    def play_sound(self):
        sd.play(self.sound, self.model.sr)
        sd.wait()
    
    def toggle_timer(self, state):
        if state == Qt.Checked:
            self.timer.start(1000 * 60 // self.bpm)  # Start the timer based on BPM
        else:
            self.timer.stop()  # Stop the timer
    
    def populate_combobox2(self):
        selected_folder = self.comboBox.currentText()
        folder_path = os.path.join(self.dataset_path, selected_folder)
        print("INFO : Sample Pack selected :",folder_path)
        # Get all .wav files in the selected folder and its subfolders
        wav_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root,file))
        
        # Populate comboBox2 with the wav files
        self.comboBox_2.blockSignals(True)
        self.comboBox_2.clear()
        self.comboBox_2.blockSignals(False)
        for wav_file in wav_files:
            self.comboBox_2.addItem(os.path.relpath(wav_file, folder_path))
    
    def load_sample(self):

        sample_path = os.path.join(self.dataset_path, self.comboBox.currentText() ,self.comboBox_2.currentText())
        
        print("INFO : Loading Sample ",sample_path)
        if not os.path.exists(sample_path):
            print(f"Error: Sample path '{sample_path}' does not exist.")
            return
        
        latent_pca1, latent_pca2, target_freq, target_attack, target_release = self.model.encode(sample_path)
        
        self.sound_data = self.model.decode(latent_pca1, latent_pca2, target_freq, target_attack, target_release)
        print("Loaded Results",self.sound_data[:10], latent_pca1, latent_pca2, target_freq, target_attack, target_release)

        self.pca_dial_1.setValue(int(latent_pca1*self.control_precision))
        self.pca_dial_2.setValue(int(latent_pca2*self.control_precision))
        self.Freq_Slider.setValue(int(0.5+target_freq*self.control_precision))
        self.Attack_Slider.setValue(int(0.5+target_attack*self.control_precision))
        self.Release_Slider.setValue(int(0.5+target_release*self.control_precision))

        #self.sound_data = self.sound_data/np.max(np.abs(self.sound_data))
        self.post_pro_audio(dry_mode = True)
        
        try:
            pass
        except Exception as e:
            print(e)
            raise e
    
    def post_pro_audio(self, dry_mode = False):

        fade_out_length = 1000
        self.sound_data[-fade_out_length:] *= np.linspace(1, 0, fade_out_length)
        padding = np.zeros(self.notic_padding)

        self.sound_data = np.concatenate((self.sound_data, padding))

        if dry_mode:
            self.sound_data_dry = np.copy(self.sound_data)


        self.sound = (1 - self.dry/self.control_precision) * self.sound_data_dry + (self.dry/self.control_precision) * self.sound_data
        self.sound *= 10**(self.gain / 20)

        print(f"INFO : New audio generated | max:{np.max(self.sound)} | std:{np.std(self.sound)} | shape:{self.sound.shape}")
        #print(self.sound_data[-10:])
        #write('test.wav', 22050 ,self.sound_data)
    
    def load_model(self):
        print("INFO : Load Model")
        model_path = os.path.join(self.model_path,self.comboBox_Model.currentText(),'data')
        print("INFO : Load Model",model_path)
        self.model = LatentPlayGenerator(model_path)


if __name__ == "__main__":

    # Run App
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog(DATASET_PATH, MODEL_PATH)
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())