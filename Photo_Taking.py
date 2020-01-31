import sys
from PyQt5.QtWidgets import QDialog, QApplication, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import sys
import os
import dlib
import glob
import cv2
from matplotlib import pyplot as plt
import psycopg2 as pg

import random

class Window(QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.face_rec = 'dlib_face_recognition_resnet_model_v1.dat' # модель для преобразования в вектор лица
        self.predictor = 'shape_predictor_5_face_landmarks.dat' # модель для поиска глаз и носа на фото

        self.label = QLabel(f'Press button to take a photo',self)
        self.button = QPushButton(self)
        self.button.clicked.connect(self.timer_start)
        self.time_passed_qll = QLabel()

        layout = QVBoxLayout()

        layout.addWidget(self.label)
        layout.addWidget(self.button)
        layout.addWidget(self.time_passed_qll)

        self.setLayout(layout)

        # трехсекундный таймер до фото

    def timer_start(self):
        self.time_left_int = 3

        self.my_qtimer = QTimer(self)
        self.my_qtimer.timeout.connect(self.timer_timeout)
        self.my_qtimer.start(1000)

        self.update_gui()

    def timer_timeout(self):
        self.time_left_int -= 1

        if self.time_left_int == 0:
            self.facerec(self.predictor, self.face_rec)
            self.my_qtimer.stop()

        self.update_gui()

    def update_gui(self):
        self.time_passed_qll.setText(str(self.time_left_int))


    def facerec(self, predictor_path, face_rec_model_path):
        detector = dlib.get_frontal_face_detector() # модель для поиска лица
        sp = dlib.shape_predictor(predictor_path) # модель для поиска глаз и носа на фото
        facerec = dlib.face_recognition_model_v1(face_rec_model_path) # модель для преобразования лица в векторное представление

        cam = cv2.VideoCapture(0)
        color_green = (0,255,0)
        line_width = 3

        connection = pg.connect("dbname=cv_recommender")
        cursor = connection.cursor()

        while True:
            ret_val, img = cam.read()
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dets = detector(rgb_image) # ищем лица в потоке видео с камеры
            if dets:
                if len(dets)>1: # если лиц в кадре больше одного
                    print('please one person at a time!')
                    break
                else:
                    print('a face is detected')
                    frameRGB = img[:,:,::-1] # BGR => RGB

                    shape = sp(frameRGB, dets[0])
                    face_descriptor = list(facerec.compute_face_descriptor(frameRGB, shape)) # по особым точкам лица составляем вектор

                    # смотрим, есть ли пользователь в базе (сравниваем с пороговым значением расстояния между текущим вектором и всеми в базе)
                    cursor.execute(f"SELECT * FROM (SELECT user_id, vector, (cube_distance(users.vector, cube(array{face_descriptor}))) as dist FROM users) as t WHERE dist < 0.45 ORDER BY dist LIMIT 1;")
                    # если таких совпадений по порогу несколько (что практически невозможно), то берем самого похожего
                    result = cursor.fetchone()
                    if result:
                        print(f'you are in the base: id {result[0]}')
                        # если пользователь есть в базе, то обновляем его векторное представление новой фотографией (усредняем, чтобы лучше распознавалось потом)
                        res = np.fromstring(result[1][1:-1], dtype=float, sep=', ')
                        upd = list(np.mean([res,face_descriptor],axis=0))
                        cursor.execute(f"UPDATE users SET vector = cube(array{upd}) WHERE user_id = {result[0]}")#update
                        connection.commit()
                    else:
                        print('you are not in the base yet')
                        # если нет в базе, то просто добавляем его
                        cursor.execute(f"INSERT INTO users(vector) VALUES(cube(array{face_descriptor}))")
                        connection.commit()
                    break
        cursor.close()
        connection.close()




if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
