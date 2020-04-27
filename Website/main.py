from sys import stdout
from video_processing import VideoProcessing
import logging
from flask import Flask, render_template, Response,request,stream_with_context,session,jsonify,redirect,url_for
from flask_socketio import SocketIO
from camera import Camera
from utils import base64_to_pil_image, pil_image_to_base64
from CVRecommender import CVRecommender
import redis
import uuid
import json
from datetime import datetime, timedelta
app = Flask(__name__)
DEBUG=True
# Check Configuration section for more details
app.config.from_object(__name__)

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True
socketio = SocketIO(app)
camera = Camera(VideoProcessing())


@socketio.on('input image', namespace='/cvr')
def get_input(input):
    input = input.split(",")[1]
    camera.enqueue_input(input)

@app.route('/',methods=["GET", "POST"])
def index():
    if request.method =='GET':
        cvr = CVRecommender()
        cvr.testConnection()
        session['sid']=str(uuid.uuid4())
        r = redis.StrictRedis(host='localhost', port='6379', password='', decode_responses=True)
        r.hmset(session.get('sid'), {'age':0,'gender':0,'descriptors':0})
        ttl = timedelta(hours=1)
        r.expire(name=session.get('sid'), time=ttl)
        session['loggedIn'] = False
        return render_template('camera.html')

    elif request.method == 'POST':
        if session['loggedIn']:
            session['LoggedIn'] = False
            return redirect('/')
        else:
            cvr = CVRecommender()
            cvr.testConnection()
            r = redis.StrictRedis(host='localhost', port='6379', password='', decode_responses=True)
            usr = r.hgetall(session.get('sid'))
            id = cvr.getUserID(usr)
            orders = cvr.getOrders(id)
            items = cvr.getItems()
            recs = cvr.getRecommendations(id, 5)
            r.hmset(session.get('sid'), {'id':id})
            session['loggedIn'] = True
            return render_template('orderingapp.html',prev_orders=orders,food_items=items,recs=recs)



def gen(id):
    while True:
        tx = camera.get_data(id)
        text = tx if tx is not None else ''
        yield f'data:{text}\n\n'

@app.route('/overlay_feed',methods=["GET", "HEAD"])
def video_feed():
    if request.method == 'GET':
        return Response(stream_with_context(gen(session.get('sid'))), mimetype='text/event-stream')
    elif request.method == 'HEAD':
        resp = Response()
        r = redis.StrictRedis(host='localhost', port='6379', password='', decode_responses=True)
        k = r.hget(session.get('sid'),'age')
        resp.headers['face_found'] = 1 if int(json.loads(k)) > 0 else 0
        return resp


@app.route('/updrecs')
def updrecs():
        cvr = CVRecommender()
        cvr.testConnection()
        r = redis.StrictRedis(host='localhost', port='6379', password='', decode_responses=True)
        usr = r.hget(session.get('sid'),'id')
        recs = cvr.getRecommendations(int(usr),5)
        return render_template('recommendations.html',recs=recs)

@app.route('/updorders')
def updorders():
        cvr = CVRecommender()
        cvr.testConnection()
        r = redis.StrictRedis(host='localhost', port='6379', password='', decode_responses=True)
        usr = r.hget(session.get('sid'),'id')
        orders = cvr.getOrders(int(usr))
        return render_template('orders.html',prev_orders=orders)

@app.route('/order',methods=['POST'])
def order():
        cvr = CVRecommender()
        cvr.testConnection()
        data = request.json
        r = redis.StrictRedis(host='localhost', port='6379', password='', decode_responses=True)
        usr = r.hget(session.get('sid'),'id')
        cvr.makeOrder(int(usr),data)
        return jsonify({"success": True})


if __name__ == '__main__':
    socketio.run(app)
