from flask import Flask, render_template,request,jsonify,Response
import psycopg2 as pg
from flask import redirect
import json
app = Flask(__name__)
import pandas as pd
import numpy as np
import turicreate as tc
connection = None
import cv2
descriptor = None

@app.route('/',methods=["GET", "POST"])
def index():
    global usr
    global connection

    print(connection.closed)

    try:
        cur = connection.cursor()
        cur.execute('SELECT 1')

    except Exception as e:
        connection = connect()

    if request.method == 'GET':

        return render_template('camera.html')
    elif request.method == 'POST':

        usr = getUser()
        if usr is not None:
            r = getOrders()
            items = getItems()
            rec = getRecommendations()

            return render_template('orderingapp.html',prev_orders=r,food_items=items,recs=rec)
        else:
            return render_template('camera.html')

def getUser():
    global descriptor
    if descriptor is not None:
        cursor = connection.cursor()
        cursor.execute(f"SELECT * FROM (SELECT user_id, vector, (cube_distance(users.vector, cube(array{descriptor}))) as dist FROM users) as t WHERE dist < 0.6 ORDER BY dist LIMIT 1;")
        # если таких совпадений по порогу несколько (что практически невозможно), то берем самого похожего
        result = cursor.fetchone()
        if result:
            print(f'you are in the base: id {result[0]}')
            # если пользователь есть в базе, то обновляем его векторное представление новой фотографией (усредняем, чтобы лучше распознавалось потом)
            res = np.fromstring(result[1][1:-1], dtype=float, sep=', ')
            upd = list(np.mean([res,descriptor],axis=0))
            cursor.execute(f"UPDATE users SET vector = cube(array{upd}) WHERE user_id = {result[0]}")#update
            connection.commit()
            return result[0]
        else:
            print('you are not in the base yet')
            # если нет в базе, то просто добавляем его
            cursor.execute(f"INSERT INTO users(vector) VALUES(cube(array{descriptor})) RETURNING user_id")
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
    video_stream = VideoCamera()
    return Response(gen(video_stream), mimetype='multipart/x-mixed-replace; boundary=frame')





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

    cursor.execute(f"SELECT vector from users WHERE user_id={usr}")
    vec = cursor.fetchone()[0]
    vec = list(np.fromstring(vec[1:-1], dtype=float, sep=', '))

    cursor.execute(f"SELECT user_id FROM (SELECT user_id, (cube_distance(users.vector, cube(array{vec}))) as dist FROM users) as t ORDER BY dist LIMIT 6;")
    closest_users = cursor.fetchall()
    foods=[]
    for cl_u in closest_users[1:]:
        cursor.execute(f"SELECT i.item_id\
        FROM orders o,items i, orders_items oi WHERE o.user_id = {cl_u[0]} \
        AND oi.item_id = i.item_id AND o.order_id = oi.order_id")
        pr_orders = cursor.fetchall()
        foods= foods+pr_orders

    from collections import Counter

    sss=Counter(foods)
    p=sss.most_common(5)
    re = [f[0][0] for f in p]
    print(re)
    hm = 5
    rc=[np.array(dishes[:hm],dtype=int),np.array(re,dtype=int),np.array(y['item_id'][:hm],dtype=int),np.array(z['item_id'][:hm],dtype=int)]
    titles = ['Most Popular Dishes:','People Who Look Like You, Love:','Content','Interactions']
    recs = []
    for t,r in zip(titles,rc):
        recs.append({'title':t, 'recs':[{'item_id':k, 'item_name':v} for k, v in zip(r, items_names[r-1])]})
    print(t5-t4,t6-t5)
    return recs

import dlib


class VideoCamera(object):
    def __init__(self):
        self.active=True
        genderProto = "gender_deploy.prototxt"
        genderModel = "gender_net.caffemodel"
        self.genderNet = cv2.dnn.readNet(genderModel, genderProto)

        self.genderList = ['Male','Female']
        self.ageList = np.arange(0, 101).reshape(101, 1)

        ageProto = "age_deploy.prototxt"
        ageModel = "age_net.caffemodel"
        self.ageNet = cv2.dnn.readNet(ageModel, ageProto)

        self.ageList = ['(0 - 2)', '(4 - 6)', '(8 - 12)', '(15 - 20)', '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 100)']

        cam = cv2.VideoCapture(0)
        self.video = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector() # модель для поиска лица
        self.sp = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat') # модель для поиска глаз и носа на фото
        self.facerec = dlib.face_recognition_model_v1( 'dlib_face_recognition_resnet_model_v1.dat') #

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()

        frame = self.process(frame)# DO WHAT YOU WANT WITH TENSORFLOW / KERAS AND OPENCV

        ret, jpeg = cv2.imencode('.jpg', frame)

        return jpeg.tobytes()

    def process(self,rgb_image):
     global descriptor
     dets = self.detector(rgb_image)
     if dets:
         if len(dets)>1:
            cv2.putText(rgb_image, "Too many people in view", (10,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

         else:
            shape = self.sp(rgb_image, dets[0])
            descriptor = list(self.facerec.compute_face_descriptor(rgb_image, shape)) # по особым точкам лица составляем вектор


            for i in range(5):
                c = shape.part(i)
                cv2.circle(rgb_image, (c.x,c.y), 2, (0, 0, 255), -1)

            blob = cv2.dnn.blobFromImage(rgb_image, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            self.genderNet.setInput(blob)
            genderPreds = self.genderNet.forward()
            gender = self.genderList[genderPreds[0].argmax()]

            self.ageNet.setInput(blob)
            agePreds = self.ageNet.forward()
            age = self.ageList[agePreds[0].argmax()]


            cv2.putText(rgb_image, f"Age : {age}", (10,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
            cv2.putText(rgb_image, f"Gender : {gender}", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)


     return rgb_image


def connect():
    with open('credentials.txt') as creds:
        db,user,host,port = creds.read().splitlines()
    return pg.connect(dbname=db, user=user, host=host, port=port)


if __name__ == '__main__':

    connection = connect()

    app.run(debug=True)
