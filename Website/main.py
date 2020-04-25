from flask import Flask, render_template, request, jsonify, Response, redirect
from CVRecommender import CVRecommender

app = Flask(__name__)
cvr = None

@app.route('/',methods=["GET", "POST"])
def index():
    cvr.testConnection()
    if request.method == 'GET':
        cvr.user.descriptors = []
        cvr.user.ID = None
        cvr.user.gender = None
        cvr.user.age = None
        cvr.user.loggedIn = False
        return render_template('camera.html')

    elif request.method == 'POST':
        cvr.getUserID()
        if cvr.user.ID is not None and not cvr.user.loggedIn:
            orders = cvr.getOrders()
            items = cvr.getItems()
            recs = cvr.getRecommendations(5)
            cvr.user.loggedIn = True
            return render_template('orderingapp.html',prev_orders=orders,food_items=items,recs=recs)
        else:
            cvr.user.descriptors = []
            cvr.user.gender = None
            cvr.user.ID = None
            cvr.user.age = None
            cvr.user.loggedIn = False
            return render_template('camera.html')

@app.route('/updrecs')
def updrecs():
    cvr.testConnection()
    recs = cvr.getRecommendations(5)
    return render_template('recommendations.html',recs=recs)

@app.route('/updorders')
def updorders():
    cvr.testConnection()
    orders = cvr.getOrders()
    return render_template('orders.html',prev_orders=orders)

@app.route('/order',methods=['POST'])
def order():
    cvr.testConnection()
    data = request.json
    cvr.makeOrder(data)
    return jsonify({"success": True})

@app.route('/video_feed',methods=['GET','HEAD'])
def video_feed():
    if request.method == 'GET':
        cvr.testConnection()
        cvr.vp.start()
        print('started')
        resp = Response(cvr.yieldFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        #resp.headers['face_found'] = True if len(cvr.user.descriptors) > 0 else False

        return resp
    elif request.method == 'HEAD':
        resp = Response()
        resp.headers['face_found'] = 1 if len(cvr.user.descriptors) > 0 else 0

        return resp

@app.route('/stop',methods=["POST"])
def stop():
    print('stopped')
    cvr.vp.stop()
    return '',200


if __name__ == '__main__':
    cvr = CVRecommender()
    app.run(debug=True)
