import AOIWindow_UI
import cv2 as cv
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *
import os
import time
import numpy as np
import pypylon.pylon as py
import imutils
from scipy.spatial import distance
import pytesseract
import time
from omegaconf import OmegaConf

emulation_mode = False

# If using emulation camera
if emulation_mode:
    os.environ["PYLON_CAMEMU"] = "1"


class AutomatedOpticalInspection(AOIWindow_UI.Ui_MainWindow, QMainWindow):
    open_cam_signal = Signal(int)
    close_cam_signal = Signal()
    trigger_cam_signal = Signal()

    def __init__(self):
        super(AutomatedOpticalInspection, self).__init__()
        self.setupUi(self)
        self.conf = OmegaConf.load('./config/config.yaml')
        self.enable_logging = self.conf['datalog']['enable']
        self.post_processing = self.conf['testing_options']['post_processing']
        self.cam = None
        self.tlf = py.TlFactory.GetInstance()
        self.list_devices = self.tlf.EnumerateDevices()
        for device in self.list_devices:
            self.comboBoxListCam.addItem(device.GetModelName())

        # Signal
        self.pushButtonStart.clicked.connect(self.run_camera)
        self.pushButtonStop.clicked.connect(self.stop_camera)
        self.pushButtonTrigger.clicked.connect(self.trigger_camera)

        self.cam_handler = CameraHandler()
        self.inspector = None
        self.cam_handler.send_image_signal.connect(self.inspect)
        self.cam_handler.start()

        self.open_cam_signal.connect(self.cam_handler.open_camera)
        self.close_cam_signal.connect(self.cam_handler.close_camera)
        self.trigger_cam_signal.connect(self.cam_handler.grab_image)
        self.total_pass = 0
        self.total_fail = 0
        self.counter = 0
        self.set_ui_start_grabber()

    @Slot(np.ndarray)
    def inspect(self, test_img):
        self.inspector = AOIInspector(post_processing=self.post_processing)
        self.inspector.send_image_signal.connect(self.update_image)
        self.inspector.save_raw_image_signal.connect(self.save_raw_image)
        self.inspector.log_result_image_signal.connect(self.save_result_image)
        self.inspector.log_result_signal.connect(self.save_result)
        self.inspector.inspection_result_signal.connect(
            self.update_inspection_result)
        self.inspector.misplacement_result_signal.connect(
            self.update_misplacement_result)
        self.inspector.missing_backlit_result_signal.connect(
            self.update_missing_backlit_result)
        self.inspector.test_time_signal.connect(self.update_test_time)
        self.inspector.progress_value_signal.connect(self.update_progress_bar)
        self.inspector.inspection_on_progress.connect(self.set_ui_progress)
        self.inspector.inspection_finish.connect(self.stop_inspection)
        self.inspector.set_test_image(test_img)
        self.inspector.start()

    @Slot()
    def stop_inspection(self):
        self.inspector.quit()

    def set_ui_start_grabber(self):
        self.pushButtonStart.setEnabled(True)
        self.pushButtonStop.setEnabled(False)
        self.pushButtonTrigger.setEnabled(False)

    def set_ui_stop_grabber(self):
        self.pushButtonStart.setEnabled(False)
        self.pushButtonStop.setEnabled(True)
        self.pushButtonTrigger.setEnabled(True)

    def run_camera(self):
        selected_cam_idx = self.comboBoxListCam.currentIndex()
        self.open_cam_signal.emit(selected_cam_idx)
        self.set_ui_stop_grabber()

    def stop_camera(self):
        self.close_cam_signal.emit()
        self.set_ui_start_grabber()

    def trigger_camera(self):
        self.trigger_cam_signal.emit()

    def closeEvent(self, event):
        if self.cam_handler != None:
            self.stop_camera()
        if self.inspector != None:
            self.inspector.quit()
        self.cam_handler.quit()

    @Slot()
    def set_ui_progress(self):
        self.labelInspectionResult.setStyleSheet(
            "background-color: yellow; color: black")
        self.labelInspectionResult.setText(
            '<html><head/><body><p align="center"><span style=" font-size:22pt; font-weight:600;">Running</span></p></body></html>')
        self.labelTestTime.setStyleSheet(
            "background-color: green ; color: yellow")
        self.labelTestTime.setText(
            '<html><head/><body><p align="center"><span style=" font-size:18pt; font-weight:600;">Running</span></p></body></html>')
        self.labelMisplacement.setStyleSheet(
            "background-color: red ; color: yellow")
        self.labelMisplacement.setText(
            '<html><head/><body><p align="center"><span style=" font-size:18pt; font-weight:600;">Running</span></p></body></html>')
        self.labelBacklitAbsense.setStyleSheet(
            "background-color: red ; color: yellow")
        self.labelBacklitAbsense.setText(
            '<html><head/><body><p align="center"><span style=" font-size:18pt; font-weight:600;">Running</span></p></body></html>')
        self.pushButtonTrigger.setEnabled(False)

    @Slot(np.ndarray)
    def update_image(self, frame):
        h, w, c = frame.shape
        image = QImage(frame.data, w, h, 3*w, QImage.Format_RGB888)
        pixmap = QPixmap(image)
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        self.graphicsViewCamera.setScene(scene)

    @Slot(np.ndarray)
    def save_raw_image(self, frame):
        if self.enable_logging:
            filename = os.path.join(
                self.conf['datalog']['raw_location'], str(self.counter+1)+".png")
            cv.imwrite(filename, frame)
            print("Save raw ", filename)

    @Slot(np.ndarray)
    def save_result_image(self, frame):
        if self.enable_logging:
            filename = os.path.join(
                self.conf['datalog']['result_location'], str(self.counter+1)+".png")
            cv.imwrite(filename, frame)
            print("Save result ", filename)

    @Slot(list)
    def save_result(self, list_string):
        if self.enable_logging:
            filename = os.path.join(
                self.conf['datalog']['testing_result'], str(self.counter+1)+".txt")
            with open(filename, 'w') as f:
                for text in list_string:
                    f.write(text)
                    f.write('\n')
            print("Save log result ", filename)

    @Slot(int)
    def update_progress_bar(self, value):
        self.progressBarTesting.setValue(value)

    @Slot(bool)
    def update_inspection_result(self, result_pass):
        if result_pass:
            self.total_pass += 1
            self.labelInspectionResult.setStyleSheet(
                "background-color: green ; color: yellow")
            self.labelInspectionResult.setText(
                '<html><head/><body><p align="center"><span style=" font-size:22pt; font-weight:600;">PASS</span></p></body></html>')
            self.labelTotalPass.setText(
                '<html><head/><body><p align="center"><span style=" font-size:18pt; font-weight:600;">'+str(self.total_pass)+'</span></p></body></html>')
        else:
            self.total_fail += 1
            self.labelInspectionResult.setStyleSheet(
                "background-color: red; color: yellow")
            self.labelInspectionResult.setText(
                '<html><head/><body><p align="center"><span style=" font-size:22pt; font-weight:600;">FAIL</span></p></body></html>')
            self.labelTotalFail.setText(
                '<html><head/><body><p align="center"><span style=" font-size:18pt; font-weight:600;">'+str(self.total_fail)+'</span></p></body></html>')
        self.counter += 1
        self.pushButtonTrigger.setEnabled(True)

    @Slot(str)
    def update_test_time(self, test_time):
        self.labelTestTime.setStyleSheet(
            "background-color: green ; color: yellow")
        self.labelTestTime.setText(
            '<html><head/><body><p align="center"><span style=" font-size:18pt; font-weight:600;">'+test_time+'</span></p></body></html>')

    @Slot(int)
    def update_misplacement_result(self, n_misplacements):
        self.labelMisplacement.setStyleSheet(
            "background-color: red ; color: yellow")
        self.labelMisplacement.setText(
            '<html><head/><body><p align="center"><span style=" font-size:18pt; font-weight:600;">'+str(n_misplacements)+'</span></p></body></html>')

    @Slot(int)
    def update_missing_backlit_result(self, n_missing_backlits):
        self.labelBacklitAbsense.setStyleSheet(
            "background-color: red ; color: yellow")
        self.labelBacklitAbsense.setText(
            '<html><head/><body><p align="center"><span style=" font-size:18pt; font-weight:600;">'+str(n_missing_backlits)+'</span></p></body></html>')


class CameraHandler(QThread, QObject):
    send_image_signal = Signal(np.ndarray)

    def __init__(self) -> None:
        super(CameraHandler, self).__init__()
        self.conf = OmegaConf.load('./config/config.yaml')
        self.camera_mode = self.conf['camera_mode']
        self.converter = py.ImageFormatConverter()
        self.converter.OutputPixelFormat = py.PixelType_BGR8packed
        self.converter.OutputBitAlignment = py.OutputBitAlignment_MsbAligned
        self.cam = None
        with open('mtx.npy', 'rb') as f:
            self.mtx = np.load(f)
        with open('dist.npy', 'rb') as f:
            self.dist = np.load(f)

    @Slot(int)
    def open_camera(self, id):
        if self.camera_mode:
            print("Open camera")
            self.tlf = py.TlFactory.GetInstance()
            self.list_devices = self.tlf.EnumerateDevices()
            cam_dev = self.list_devices[id]
            self.cam = py.InstantCamera(self.tlf.CreateDevice(cam_dev))

            try:
                self.cam.Open()
            except:
                print("Error open camera !")
            if self.cam.IsOpen():
                try:
                    self.cam.TriggerMode.SetValue("On")
                    self.cam.TriggerSelector.SetValue("FrameStart")
                    self.cam.TriggerSource.SetValue("Software")
                    self.cam.StartGrabbing(py.GrabStrategy_LatestImageOnly)
                except:
                    print("Camera grabber error !")

    @Slot()
    def close_camera(self):
        if self.camera_mode:
            print("Close camera")
            if self.cam != None:
                if self.cam.IsGrabbing():
                    self.cam.StopGrabbing()
                if self.cam.IsOpen():
                    self.cam.Close()

    # Setelah image diperoleh dikirimkan ke main window
    @Slot()
    def grab_image(self):
        if self.camera_mode:
            print("Grab image")
            if self.cam.IsGrabbing():
                self.cam.ExecuteSoftwareTrigger()
                grab = self.cam.RetrieveResult(
                    5000, py.TimeoutHandling_ThrowException)
                if grab.GrabSucceeded():
                    print("Grab success")
                    img = self.converter.Convert(grab)
                    grab.Release()
                    bgr_img = img.GetArray()
                    h,  w = bgr_img.shape[:2]
                    new_cam_mtx, roi = cv.getOptimalNewCameraMatrix(
                        self.mtx, self.dist, (w, h), 1, (w, h))
                    undistort_img = cv.undistort(
                        bgr_img, self.mtx, self.dist, None, new_cam_mtx)
                    test_img = imutils.rotate(undistort_img, angle=180).copy()
                    self.send_image_signal.emit(test_img)
                else:
                    print("Grab failed")
        else:
            cwd = os.getcwd()
            filename = os.path.join(cwd, "./dummy-images/12.png")
            test_img = cv.imread(filename)
            self.send_image_signal.emit(test_img)


class AOIInspector(QThread, QObject):
    send_image_signal = Signal(np.ndarray)
    save_raw_image_signal = Signal(np.ndarray)
    log_result_image_signal = Signal(np.ndarray)
    log_result_signal = Signal(list)
    inspection_result_signal = Signal(bool)
    misplacement_result_signal = Signal(int)
    missing_backlit_result_signal = Signal(int)
    test_time_signal = Signal(str)
    inspection_finish = Signal()
    inspection_on_progress = Signal()
    progress_value_signal = Signal(int)

    def __init__(self, post_processing=False):
        super(AOIInspector, self).__init__()
        # Load ROI
        with open('ROI.npy', 'rb') as f:
            self.ROI = np.load(f)
        # Load Backlit Template
        with open('backlit.npy', 'rb') as f:
            self.backlit_template = np.load(f)
        self.post_processing = post_processing
        self.red_color = (0, 0, 255)
        self.green_color = (0, 255, 0)

        self.threshold_OCR = [145, 149, 147, 147, 147,
                              142, 143, 140, 138, 147,
                              140, 140, 150, 145, 145,
                              130, 127, 127, 127, 132,
                              127, 127, 127, 127, 120,
                              117, 127, 120, 130, 133,
                              137, 120, 120, 120, 145,
                              137, 139, 127, 171, 180,
                              180, 180, 170]

        self.correct_text = ["SMARTINSRT", "APPND", "RIPLO/WR", "ESC", "SYNCBIN",
                             "AUDIOLEVEL", "FULLVIEW", "SOURCE", "TIMELINE", "CLOSEUP",
                             "PLACEONTOP", "SRCO/WR", "TRANS", "SPLIT", "SNAP",
                             "RIPLDEL", "CAM7", "CAM8", "CAM9", "LIVEO/WR",
                             "TRIMIN", "TRIMOUT", "ROLL", "CAM4", "CAM5",
                             "CAM6", "VIDEOONLY", "SLIPSRC", "SLIPDEST", "TRANSDUR",
                             "CAM1", "CAM2", "CAM3", "AUDIOONLY", "CUT",
                             "DIS", "SMTHCUT", "STOP/PLAY", "IN", "OUT",
                             "SHTL", "JOG", "SCRL"]

        self.threshold_backlit = 0.14

    def set_test_image(self, test_img):
        self.test_img = test_img

    def run(self):
        self.inspection_on_progress.emit()
        self.save_raw_image_signal.emit(self.test_img)
        result_img = self.test_img.copy()
        print("Inspection Running")
        print("Post processing : ", self.post_processing)
        # disini tampilin gambar capture
        test_img_gray = cv.cvtColor(self.test_img, cv.COLOR_BGR2GRAY)
        fail_number = 0
        pass_number = 0
        missing_backlit = 0
        missplacement = 0
        n_ocr_text = len(self.correct_text)
        list_result = []

        for i in range(0, self.ROI.shape[0]):
            x = self.ROI[i][0]
            y = self.ROI[i][1]
            w = self.ROI[i][2]
            h = self.ROI[i][3]

            start_point = (x, y)
            end_point = (x+w, y+h)
            roi_img = test_img_gray[y:y+h, x:x+w].copy()

            # Check OCR
            if i < n_ocr_text:
                roi_img = cv.resize(roi_img, None, fx=8, fy=8,
                                    interpolation=cv.INTER_CUBIC)
                kernel = np.ones((3, 3), np.uint8)
                roi_img = cv.dilate(roi_img, kernel, iterations=4)
                roi_img = cv.erode(roi_img, kernel, iterations=1)

                custom_config = r'--oem 3 --psm 6'

                # Black Caps
                if i < 38:
                    _, binary_img = cv.threshold(
                        roi_img, self.threshold_OCR[i], 255, cv.THRESH_BINARY_INV)
                    text = pytesseract.image_to_string(
                        binary_img, config=custom_config)
                    text = text.replace(" ", "")
                    text = text.replace("\r", "")
                    text = text.replace("\n", "")
                    text = text.strip()
                    if self.post_processing:
                        text = text.capitalize()

                    if text == self.correct_text[i]:
                        cv.rectangle(result_img, start_point, end_point,
                                     self.green_color, thickness=2)
                        pass_number = pass_number + 1
                    else:
                        cv.rectangle(result_img, start_point,
                                     end_point, self.red_color, thickness=2)
                        fail_number = fail_number + 1
                        missplacement += 1

                # White Caps
                else:
                    _, binary_img = cv.threshold(
                        roi_img, self.threshold_OCR[i], 255, cv.THRESH_BINARY)
                    text = pytesseract.image_to_string(
                        binary_img, config=custom_config)
                    text = text.replace(" ", "")
                    text = text.replace("\r", "")
                    text = text.replace("\n", "")
                    text = text.strip()
                    if self.post_processing:
                        text = text.capitalize()
                    if text == self.correct_text[i]:
                        cv.rectangle(result_img, start_point,
                                     end_point, self.green_color, thickness=2)
                        pass_number += 1
                    else:
                        cv.rectangle(result_img, start_point,
                                     end_point, self.red_color, thickness=2)
                        fail_number += 1
                        missplacement += 1
                log_text = "ROI " + str(i) + " OCR : " + text
            # Check Backlit
            else:
                roi_img = cv.resize(roi_img, (100, 100))
                roi_histogram = cv.calcHist(
                    [roi_img], [0], None, [256], [0, 256])
                norm_roi_histogram = roi_histogram/np.sum(roi_histogram)
                # Calc cosine distance
                dist = distance.cosine(
                    self.backlit_template[i-n_ocr_text], norm_roi_histogram)

                if dist <= self.threshold_backlit:
                    cv.rectangle(result_img, start_point,
                                 end_point, self.green_color, thickness=2)
                    pass_number += 1
                else:
                    cv.rectangle(result_img, start_point,
                                 end_point, self.red_color, thickness=2)
                    fail_number += 1
                    missing_backlit += 1
                log_text = "ROI " + str(i) + " Cosine Distance : " + str(dist)
            print(log_text)
            list_result.append(log_text)
            inspection_img = cv.resize(result_img.copy(), (800, 600))
            inspection_img = cv.cvtColor(inspection_img, cv.COLOR_BGR2RGB)
            self.send_image_signal.emit(inspection_img)
            progress_value = int((i/63.0) * 100.0)
            self.progress_value_signal.emit(progress_value)
        self.log_result_signal.emit(list_result)
        self.log_result_image_signal.emit(result_img)
        if fail_number == 0:
            self.inspection_result_signal.emit(True)
        else:
            self.inspection_result_signal.emit(False)
        self.misplacement_result_signal.emit(missplacement)
        self.missing_backlit_result_signal.emit(missing_backlit)
        self.test_time_signal.emit(str(time.ctime()))
        self.inspection_finish.emit()


if __name__ == "__main__":
    cam_viewer = QApplication()
    cam_viewer_gui = AutomatedOpticalInspection()
    cam_viewer_gui.show()
    cam_viewer.exec_()
