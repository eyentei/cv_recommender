import dlib
import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
import cv2
import numpy as np

class VideoProcessing(object):
    def __init__(self):
        #camera
        self.isActive=False
        self.video=None
        #models
        self.detector = dlib.get_frontal_face_detector() # модель для поиска лица
        self.sp = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat') # модель для поиска глаз и носа на фото
        self.facerec = dlib.face_recognition_model_v1( 'models/dlib_face_recognition_resnet_model_v1.dat') #

        self.session = tf.Session(config=tf.ConfigProto())
        keras.backend.set_session(self.session)
        self.model = keras.models.load_model('models/model.h5')

        #person data
        self.gender_list = ['Female','Male']
        self.user = None

    def start(self):
        self.isActive=True
        self.video = cv2.VideoCapture(0)

    def stop(self):
        self.isActive=False
        self.video.release()
        self.video = None

    def get_frame(self, user):
        self.user = user
        ret, frame = self.video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes(), self.user

    def process(self, rgb_image):
        dets = self.detector(rgb_image)
        if dets:
            if len(dets)>1:
                cv2.putText(rgb_image, "Too many people in view", (10,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            else:
                shape = self.sp(rgb_image, dets[0])
                h,w,_=rgb_image.shape
                cropped = rgb_image[max(0, dets[0].top()): min(dets[0].bottom(), h),
                        max(0, dets[0].left()): min(dets[0].right(), w)].copy()
                descriptor = list(self.facerec.compute_face_descriptor(rgb_image, shape))
                self.user.descriptors.append(descriptor)
                self.user.descriptors = [np.mean(self.user.descriptors,axis=0)]
                img = cv2.resize(cropped, (224, 224))
                img = np.expand_dims(img/255,axis=0)

                with self.session.as_default():
                    with self.session.graph.as_default():
                        ages,genders,_ = self.model.predict(img)

                n_age = np.dot(ages[0], list(range(101)))
                n_gender = np.argmax(genders[0])

                if self.user.age is not None:
                    self.user.age = 0.95 * self.user.age + 0.05 * n_age
                else:
                    self.user.age = n_age

                if self.user.gender is not None:
                    self.user.gender = 0.8 * self.user.gender + 0.2 * n_gender
                else:
                    self.user.gender = n_gender

                cv2.putText(rgb_image, f"Age : {int(np.round(self.user.age))}", (10,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 100, 255),5)
                cv2.putText(rgb_image, f"Gender : {self.gender_list[int(np.round(self.user.gender))]}", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 100, 255),5)
        return rgb_image
