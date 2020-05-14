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
            x=np.array(img,dtype=np.float32)
            x = x[..., ::-1]
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.68
            img = np.expand_dims(x,axis=0)

            #img = np.expand_dims(np.array(img),axis=0)

            ages,genders,_ = predict(tf.convert_to_tensor(img,dtype=tf.float32))
            return jsonify({'status':'success','data':{'descriptor':descriptor,'ages':ages.numpy()[0].tolist(),'genders':genders.numpy()[0][0].tolist()}})

            #return jsonify({'status':'success','data':{'descriptor':descriptor,'ages':ages.tolist(),'genders':genders.tolist()}})
    else:
        #print('no dets')
        return jsonify({'status':'none','data':{}})


def get_graph():
    graph_def=None
    with tf.io.gfile.GFile('models/agegender.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=['input_1:0'],
                                    outputs=['age_pred/Softmax:0','gender_pred/Sigmoid:0','global_pooling/Mean:0'],
                                    print_graph=False)
    return frozen_func

def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph


    layers = [op.name for op in import_graph.get_operations()]
    if print_graph == True:
        for layer in layers:
            print(layer)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1( 'models/dlib_face_recognition_resnet_model_v1.dat') #
#model = keras.models.load_model('models/model.h5')
#    app.run(debug=True,port=5050)
predict = get_graph()
