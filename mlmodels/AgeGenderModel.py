import tensorflow as tf
from tensorflow import keras
import dlib
import cv2
import os
import numpy as np
import json
import io
from PIL import Image
from flask import Flask, request, jsonify
app = Flask(__name__)
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route('/data',methods=["POST"])
def data():
    r = request.data
    image = Image.open(io.BytesIO(r))
    array_image = np.array(image)
    dets = detector(array_image)
    if dets:
        if len(dets)>1:
            return jsonify({'status':'toomany','data':{}})
        else:
            shape = sp(array_image, dets[0])
            descriptor = list(facerec.compute_face_descriptor(array_image, shape))
            height,width,_=array_image.shape
            x = dets[0].left()
            y = dets[0].top()
            w = dets[0].right()
            h = dets[0].bottom()

            cropped = array_image[max(0, y-75): min(h+25, height),
                            max(0, x-50): min(w+50, width)].copy()

            img = Image.fromarray(cropped)
            img = img.resize((224,224))
            img = np.expand_dims(np.array(img)/255,axis=0)

            ages,genders,_ = model.predict(img)
            return jsonify({'status':'success','data':{'descriptor':descriptor,'ages':ages.tolist(),'genders':genders.tolist()}})
    else:
        #print('no dets')
        return jsonify({'status':'none','data':{}})

if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
    facerec = dlib.face_recognition_model_v1( 'models/dlib_face_recognition_resnet_model_v1.dat') #
    model = keras.models.load_model('models/model.h5')
    app.run(debug=True,port=5050)
