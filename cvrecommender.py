from flask import Flask, render_template,request,jsonify,Response,after_this_request
import psycopg2 as pg
from flask import redirect
import json
app = Flask(__name__)
import pandas as pd
import numpy as np
import turicreate as tc
connection = None
import cv2
descriptors = []
gender = None
age = None
loggedIn = False
video_stream = None

@app.route('/',methods=["GET", "POST"])
def index():
    global usr
    global connection
    global loggedIn
    global descriptors
    global age
    global gender

    print(connection.closed)

    try:
        cur = connection.cursor()
        cur.execute('SELECT 1')

    except Exception as e:
        connection = connect()

    if request.method == 'GET':
        descriptors = []
        genders = []
        ages = []
        loggedIn = False
        return render_template('camera.html')
    elif request.method == 'POST':
        usr = getUser()
        if usr is not None and not loggedIn:
            r = getOrders()
            items = getItems()
            rec = getRecommendations()
            loggedIn = True
            #video_stream.stop()
            return render_template('orderingapp.html',prev_orders=r,food_items=items,recs=rec)
        else:
            descriptors = []
            genders = []
            ages = []
            loggedIn = False
            return render_template('camera.html')

def getUser():
    global descriptors
    if len(descriptors)>0:
        cursor = connection.cursor()
        cursor.execute(f"SELECT * FROM (SELECT user_id, vector, (cube_distance(users.vector, cube(array{list(descriptors[0])}))) as dist FROM users) as t WHERE dist < 0.45 ORDER BY dist LIMIT 1;")
        # если таких совпадений по порогу несколько (что практически невозможно), то берем самого похожего
        result = cursor.fetchone()
        if result:
            print(f'you are in the base: id {result[0]}')
            # если пользователь есть в базе, то обновляем его векторное представление новой фотографией (усредняем, чтобы лучше распознавалось потом)
            #res = np.fromstring(result[1][1:-1], dtype=float, sep=', ')
            #upd = list(np.mean([res,descriptor],axis=0))
            #cursor.execute(f"UPDATE users SET vector = cube(array{upd}) WHERE user_id = {result[0]}")#update
            #connection.commit()
            return result[0]
        else:
            print('you are not in the base yet')
            # если нет в базе, то просто добавляем его
            age_list = [(0,17),(18,24),(25,34),(35,44),(45,54),(55,64),(65,100)]
            for j,a in enumerate(age_list):
                if a[0] <= age <= a[1]:
                    n_age = j+1
                    break
            cursor.execute(f"INSERT INTO users(vector,gender,age_group_id) VALUES(cube(array{list(descriptors[0])}),{bool(gender)},{int(n_age)}) RETURNING user_id")
            u = cursor.fetchone()[0]
            connection.commit()

            return u

    else: return None


def gen(camera):
    while camera.active:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    video_stream.start()
    print('started')
    return Response(gen(video_stream), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop',methods=["GET"])
def stop():
    print('stopped')
    video_stream.stop()
    return '',200

#@app.route('/start',methods=["GET"])
#def start():

#    return '',200

def getOrders():

        cursor= connection.cursor()
        cursor.execute(f"SELECT o.order_id, array_agg(i.item_name), array_agg(i.item_id)\
        FROM orders o,items i, orders_items oi WHERE o.user_id = {usr} \
        AND oi.item_id = i.item_id AND o.order_id = oi.order_id GROUP BY o.order_id")
        pr_orders = cursor.fetchall()
        pr_orders.sort(reverse=True)
        r = []
        for order in pr_orders:
            ord = {'order_id':order[0],'order_data':[]}
            for i in range(len(order[1])):
                ord['order_data'].append({'item_id':order[2][i],'item_name':order[1][i]})
            r.append(ord)
        return r

def getItems():
    cursor= connection.cursor()
    cursor.execute("SELECT * FROM items")
    items = cursor.fetchall()
    return items

@app.route('/updrecs')
def updrecs():
    rc = getRecommendations()
    return render_template('recommendations.html',recs=rc)

@app.route('/updorders')
def updorders():
    r =getOrders()
    return render_template('orders.html',prev_orders=r)


@app.route('/order',methods=['POST'])
def order():
    cursor= connection.cursor()
    data = request.json
    cursor.execute("INSERT into orders(user_id) VALUES (%s) RETURNING order_id;", (usr,))
    order_id = cursor.fetchone()[0]

    for item_id in data:
        cursor.execute("INSERT into orders_items(order_id,item_id) VALUES (%s, %s);",(order_id,item_id))
    connection.commit()

    return jsonify({"success": True})

import time
def getRecommendations():

    cursor = connection.cursor()
    cursor.execute(f"SELECT i.item_id \
    FROM orders_items oi, items i, orders o \
    WHERE oi.order_id = o.order_id AND oi.item_id = i.item_id\
    GROUP BY i.item_id ORDER BY count(i.item_id) desc\
    LIMIT 5")

    result = cursor.fetchall()
    dishes = [tpl[0] for tpl in result]

    cursor.execute("SELECT user_id FROM users")
    users = [user[0] for user in cursor.fetchall()]
    cursor.execute("SELECT item_id,item_name FROM items")
    ids_names = cursor.fetchall()
    items = [item[0] for item in ids_names]
    items_names = np.array([item[1] for item in ids_names])
    pairs = set((i,j) for i in users for j in items)
    cursor.execute("SELECT o.user_id, i.item_id, count(i.item_id) FROM users u,orders o,orders_items oi,items i \
            where u.user_id = o.user_id and o.order_id = oi.order_id and i.item_id=oi.item_id group by o.user_id,i.item_id"
            )
    cnts = cursor.fetchall()

    total_df = pd.DataFrame(cnts)
    total_df.columns = ['user_id','item_id','count']
    total_df['count'] = total_df['count']/(total_df['count'].max()-total_df['count'].min())
    total_df.head()
    cursor.execute("SELECT i.item_id,t.tag_id from items_tags it,items i,tags t where i.item_id=it.item_id and t.tag_id=it.tag_id")
    ita = pd.DataFrame(cursor.fetchall())
    ita.columns = ['item_id','tag_id']
    ita['l']=1
    pivoted = ita.pivot_table(index='item_id',columns='tag_id', values='l').fillna(0).reset_index()
    p = tc.SFrame(pivoted)
    user_id = 'user_id'
    item_id = 'item_id'
    target = 'count'

    n_rec = 3 # number of items to recommend
    n_display = 30 # to display the first few rows in an output dataset
    data = tc.SFrame(total_df)
    t4=time.time()
    #долго!
    icr = tc.item_content_recommender.create(p,item_id = item_id,observation_data=data,user_id = user_id,target = target,verbose=False)
    t5=time.time()
    rfr = tc.ranking_factorization_recommender.create(data,
                       user_id=user_id,
                                item_id=item_id, target=target,verbose=False)

    t6=time.time()
    y=icr.recommend([usr],5)
    z=rfr.recommend([usr],5)

    cursor.execute(f"SELECT vector,age_group_id,gender from users WHERE user_id={usr}")
    curr_user = cursor.fetchone()
    vec = list(np.fromstring(curr_user[0][1:-1], dtype=float, sep=', '))

    cursor.execute(f"SELECT user_id,dist FROM (SELECT user_id, (cube_distance(users.vector, cube(array{vec}))) as dist FROM users) as t ORDER BY dist LIMIT 6;")
    closest_users = cursor.fetchall()
    print(closest_users)
    looks_foods=[]
    for cl_u in closest_users[1:]:
        cursor.execute(f"SELECT i.item_id\
        FROM orders o,items i, orders_items oi WHERE o.user_id = {cl_u[0]} \
        AND oi.item_id = i.item_id AND o.order_id = oi.order_id")
        pr_orders = cursor.fetchall()
        looks_foods= looks_foods+pr_orders


    from collections import Counter

    cursor.execute(f"SELECT user_id,dist FROM (SELECT user_id, (cube_distance(users.vector, cube(array{vec}))) as dist FROM users where gender={curr_user[2]} and age_group_id={curr_user[1]}) as t ORDER BY dist LIMIT 6;")
    closest_users = cursor.fetchall()
    print(closest_users)
    dem =[]
    for cl_u in closest_users[1:]:
        cursor.execute(f"SELECT i.item_id\
        FROM orders o,items i, orders_items oi WHERE o.user_id = {cl_u[0]} \
        AND oi.item_id = i.item_id AND o.order_id = oi.order_id")
        pr_orders = cursor.fetchall()
        dem= dem+pr_orders

    #age_gender_foods = cursor.fetchall()
    rec_looks = [r[0] for r in looks_foods]
    c_looks = Counter(rec_looks).most_common(5)
    rec_looks = [r[0] for r in c_looks]

    rec_dem = [r[0] for r in dem]
    c_dem = Counter(rec_dem).most_common(5)
    rec_dem = [r[0] for r in c_dem]

    hm = 5
    rc=[np.array(dishes[:hm],dtype=int),np.array(rec_looks,dtype=int),np.array(rec_dem,dtype=int),np.array(y['item_id'][:hm],dtype=int),np.array(z['item_id'][:hm],dtype=int)]
    titles = ['Most Popular Dishes:','People Who Look Like You, Love:','People Of Your Age And Gender, Love:','Similar By Content:',"Similar By Interactions:"]
    cursor.execute(f"SELECT count(order_id) from orders WHERE user_id={usr}")

    num_orders = cursor.fetchone()[0]
    if num_orders<1:
        rc=rc[:3]
        titles=titles[:3]


    recs = []
    for t,r in zip(titles,rc):
        recs.append({'title':t, 'recs':[{'item_id':k, 'item_name':v} for k, v in zip(r, items_names[r-1])]})
    print(t5-t4,t6-t5)
    return recs

import dlib
import tensorflow as tf
from tensorflow import keras
import tensorflow as tf

class VideoCamera(object):
    def __init__(self):
        self.active=False
        '''
        genderProto = "gender_deploy.prototxt"
        genderModel = "gender_net.caffemodel"
        self.genderNet = cv2.dnn.readNet(genderModel, genderProto)

        self.genderList = ['Male','Female']
        self.ageList = np.arange(0, 101).reshape(101, 1)

        ageProto = "age_deploy.prototxt"
        ageModel = "age_net.caffemodel"
        self.ageNet = cv2.dnn.readNet(ageModel, ageProto)

        self.ageList = ['(<15)', '(15 - 20)', '(20 - 30)', '(30 - 40)', '(40 - 50)', '(50 - 60)', '(>60)']
        '''

        self.video = None
        self.detector = dlib.get_frontal_face_detector() # модель для поиска лица
        self.sp = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat') # модель для поиска глаз и носа на фото
        self.facerec = dlib.face_recognition_model_v1( 'dlib_face_recognition_resnet_model_v1.dat') #



        config = tf.ConfigProto()
        self.session = tf.Session(config=config)
        keras.backend.set_session(self.session)
        self.model=keras.models.load_model('model.h5')
        #self.model._make_predict_function()


        #self.graph = tf.get_default_graph()
        #self.model._make_predict_function()
        self.gender_list = ['Female','Male']

    def start(self):
        self.active=True
        self.video = cv2.VideoCapture(0)

    def stop(self):
        self.active=False
        self.video.release()
        self.video = None

    def get_frame(self):
        ret, frame = self.video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def process(self,rgb_image):
     global descriptors
     global gender
     global age
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
                descriptors.append(descriptor)
                descriptors = [np.mean(descriptors,axis=0)]
                img = cv2.resize(cropped, (224, 224))
                img = np.expand_dims(img/255,axis=0)

                with self.session.as_default():
                    with self.session.graph.as_default():
                        ages,genders,_ = self.model.predict(img)
                n_age = np.dot(ages[0], list(range(101)))
                n_gender = np.argmax(genders[0])

                if age is not None:
                    age = 0.95 * age + 0.05 * n_age
                else:
                    age = n_age

                if gender is not None:
                    gender = 0.8 * gender + 0.2 * n_gender
                else:
                    gender = n_gender

                cv2.putText(rgb_image, f"Age : {int(np.round(age))}", (10,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 100, 255),5)
                cv2.putText(rgb_image, f"Gender : {self.gender_list[n_gender]}", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 100, 255),5)


     return rgb_image


def connect():
    with open('credentials.txt') as creds:
        db,user,host,port = creds.read().splitlines()
    return pg.connect(dbname=db, user=user, host=host, port=port)


if __name__ == '__main__':

    connection = connect()
    video_stream = VideoCamera()
    app.run(debug=True)
