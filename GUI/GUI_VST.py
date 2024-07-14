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

from Audio import update, LatentPlayGenerator
from deep_ae import TimeFrequencyLoss


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
        self.sound_data = np.zeros(400)
        self.bpm = 128
        self.control_precision = 1000
        self.notic_padding = 4000

        
        self.model = LatentPlayGenerator(model_path)
        self.dataset_path = dataset_path

        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.emit_signal)
        
        self.signalChanged.connect(self.update_plot)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.play_sound)

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1000, 500)

        self.checkBox_autorun = QtWidgets.QCheckBox(Dialog)
        self.checkBox_autorun.setGeometry(QtCore.QRect(10, 90, 81, 31))
        self.checkBox_autorun.setObjectName("checkBox_autorun")
        self.checkBox_autorun.stateChanged.connect(self.toggle_timer) 

        self.verticalLayoutWidget_2 = QtWidgets.QWidget(Dialog)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(190, 120, 291, 71))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")

        self.Latent_Slider_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.Latent_Slider_2.setContentsMargins(0, 0, 0, 0)
        self.Latent_Slider_2.setObjectName("Latent_Slider_2")

        self.Freq_Slider = QtWidgets.QSlider(self.verticalLayoutWidget_2)
        self.Freq_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.Freq_Slider.setObjectName("Freq_Slider")
        self.Freq_Slider.setRange(0, self.control_precision)
        self.Freq_Slider.setValue(10)
        self.Freq_Slider.valueChanged.connect(self.slider_changed)
        self.Latent_Slider_2.addWidget(self.Freq_Slider)

        self.Attack_Slider = QtWidgets.QSlider(self.verticalLayoutWidget_2)
        self.Attack_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.Attack_Slider.setObjectName("Attack_Slider")
        self.Attack_Slider.setRange(0, self.control_precision)
        self.Attack_Slider.setValue(10)
        self.Attack_Slider.valueChanged.connect(self.slider_changed)
        self.Latent_Slider_2.addWidget(self.Attack_Slider)

        self.Release_Slider = QtWidgets.QSlider(self.verticalLayoutWidget_2)
        self.Release_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.Release_Slider.setObjectName("Release_Slider")
        self.Release_Slider.setRange(0, self.control_precision)
        self.Release_Slider.setValue(10)
        self.Release_Slider.valueChanged.connect(self.slider_changed)
        self.Latent_Slider_2.addWidget(self.Release_Slider)

        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(380, 10, 61, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(270, 100, 81, 20))
        self.label_2.setObjectName("label_2")
        self.line = QtWidgets.QFrame(Dialog)
        self.line.setGeometry(QtCore.QRect(0, 80, 491, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(Dialog)
        self.line_2.setGeometry(QtCore.QRect(0, 190, 491, 20))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")

        self.verticalSlider = QtWidgets.QSlider(Dialog)
        self.verticalSlider.setGeometry(QtCore.QRect(90, 110, 20, 90))
        self.verticalSlider.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider.setObjectName("verticalSlider")
        self.verticalSlider.setRange(50, 200)
        self.verticalSlider.setValue(128)
        self.verticalSlider.valueChanged.connect(self.bpm_changed)

        self.comboBox = QtWidgets.QComboBox(Dialog)
        self.comboBox.setGeometry(QtCore.QRect(10, 20, 300, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.currentIndexChanged.connect(self.populate_combobox2)

        self.comboBox2 = QtWidgets.QComboBox(Dialog)
        self.comboBox2.setGeometry(QtCore.QRect(10, 10+45, 300, 22))
        self.comboBox2.setObjectName("comboBox2")
        self.comboBox2.currentIndexChanged.connect(self.load_sample)

        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(345, 75, 39, 11))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(435, 75, 39, 11))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(140, 130, 41, 16))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(140, 150, 39, 11))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(Dialog)
        self.label_7.setGeometry(QtCore.QRect(140, 170, 39, 11))
        self.label_7.setObjectName("label_7")
        self.line_3 = QtWidgets.QFrame(Dialog)
        self.line_3.setGeometry(QtCore.QRect(120, 90, 20, 201-90))
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.line_4 = QtWidgets.QFrame(Dialog)
        self.line_4.setGeometry(QtCore.QRect(320, 0, 20, 80))
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.label_8 = QtWidgets.QLabel(Dialog)
        self.label_8.setGeometry(QtCore.QRect(90, 90, 21, 20))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(Dialog)
        self.label_9.setGeometry(QtCore.QRect(20, 40, 39, 11))
        self.label_9.setObjectName("label_9")

        self.dial_1 = QtWidgets.QDial(Dialog)
        self.dial_1.setGeometry(QtCore.QRect(340, 20, 50, 64))
        self.dial_1.setObjectName("dial_1")
        self.dial_1.setRange(-self.control_precision, self.control_precision)
        self.dial_1.setValue(0)
        self.dial_1.valueChanged.connect(self.slider_changed)

        self.dial_2 = QtWidgets.QDial(Dialog)
        self.dial_2.setGeometry(QtCore.QRect(430, 20, 50, 64))
        self.dial_2.setObjectName("dial_2")
        self.dial_2.setRange(-self.control_precision, self.control_precision)
        self.dial_2.setValue(0)
        self.dial_2.valueChanged.connect(self.slider_changed)

        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(20, 130, 51, 51))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.play_sound)

        self.graphicsView = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(10, 260, 980, 230))
        self.graphicsView.setObjectName("graphicsView")
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.checkBox_autorun.setText(_translate("Dialog", "Auto Run"))
        self.label.setText(_translate("Dialog", "PCA Control"))
        self.label_2.setText(_translate("Dialog", "Custom Control"))

        folders = os.listdir(self.dataset_path)    # Populate the comboBox with the folder names
        for index, folder_name in enumerate(folders):
            self.comboBox.addItem("")
            self.comboBox.setItemText(index, _translate("Dialog", folder_name))

        self.label_3.setText(_translate("Dialog", "Latent 1"))
        self.label_4.setText(_translate("Dialog", "Latent 2"))
        self.label_5.setText(_translate("Dialog", "Frequency"))
        self.label_6.setText(_translate("Dialog", "Attack"))
        self.label_7.setText(_translate("Dialog", "Release"))
        self.label_8.setText(_translate("Dialog", "BPM"))
        self.label_9.setText(_translate("Dialog", ""))

        self.pushButton.setText(_translate("Dialog", "Play"))

    def bpm_changed(self):
        self.bpm = self.verticalSlider.value()

    def emit_signal(self):
        p_latent_1 = self.dial_1.value()
        p_latent_2 = self.dial_2.value()
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
        nb_points = len(self.sound_data) - self.notic_padding
        path = QPainterPath()
        path.moveTo(0, 0)
        for i in range(0, len(self.sound_data) - self.notic_padding,):
            path.lineTo(i , 10*self.sound_data[i])

        # Create a pen with the desired stroke width
        pen = QPen(Qt.black)  # Set the color to black or any other color you prefer
        pen.setWidth(1)  # Set the stroke width to 1 or any other desired width

        # Add the path with the specified pen to the scene
        self.scene.addPath(path, pen)
        self.update_view_transform()

    def resizeEvent(self, event):
        super(Ui_Dialog, self).resizeEvent(event)
        self.update_view_transform()

    def update_view_transform(self):
        rect = self.scene.itemsBoundingRect()
        if rect.width() > 0 and rect.height() > 0:  # Ensure non-zero dimensions
            self.graphicsView.setTransform(QTransform().scale(
                self.graphicsView.width() / rect.width(),
                self.graphicsView.height() / rect.height()
            ))
    
    def play_sound(self):
        sd.play(self.sound_data, self.model.sr)
        sd.wait()
    
    def toggle_timer(self, state):
        if state == Qt.Checked:
            self.timer.start(1000 * 60 // self.bpm)  # Start the timer based on BPM
        else:
            self.timer.stop()  # Stop the timer
    
    def populate_combobox2(self):
        selected_folder = self.comboBox.currentText()
        folder_path = os.path.join(self.dataset_path, selected_folder)
        
        # Get all .wav files in the selected folder and its subfolders
        wav_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
        
        # Populate comboBox2 with the wav files
        self.comboBox2.clear()
        for wav_file in wav_files:
            self.comboBox2.addItem(wav_file)
    
    def load_sample(self):

        print("load_sample")
        sample_path = self.comboBox2.currentText()
        if not os.path.exists(sample_path):
            print(f"Error: Sample path '{sample_path}' does not exist.")
            return

        print("load_sample : ",sample_path)
        latent_pca1, latent_pca2, target_freq, target_attack, target_release = self.model.encode(sample_path)
        
        self.sound_data = self.model.decode(latent_pca1, latent_pca2, target_freq, target_attack, target_release)
        print("Loaded Results",self.sound_data[:10], latent_pca1, latent_pca2, target_freq, target_attack, target_release)

        self.dial_1.setValue(int(latent_pca1*self.control_precision))
        self.dial_2.setValue(int(latent_pca2*self.control_precision))
        self.Freq_Slider.setValue(int(0.5+target_freq*self.control_precision))
        self.Attack_Slider.setValue(int(0.5+target_attack*self.control_precision))
        self.Release_Slider.setValue(int(0.5+target_release*self.control_precision))

        #self.sound_data = self.sound_data/np.max(np.abs(self.sound_data))
        self.post_pro_audio()
        
        try:
            pass
        except Exception as e:
            print(e)
            raise e
    
    def post_pro_audio(self):

        fade_out_length = 1000
        self.sound_data[-fade_out_length:] *= np.linspace(1, 0, fade_out_length)

        padding = np.zeros(self.notic_padding)
        self.sound_data = np.concatenate((self.sound_data, padding))
        
        print(np.max(self.sound_data))
        #print(self.sound_data[-10:])
        #write('test.wav', 22050 ,self.sound_data)


if __name__ == "__main__":
    # Load model
    MODEL_PATH = r'models\RUN_5\data'
    DATASET_PATH = r'Dataset\kick_dataset'

    # Run App
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog(DATASET_PATH, MODEL_PATH)
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
