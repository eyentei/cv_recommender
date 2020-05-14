from PIL import Image
import requests
import io
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np
import redis
import json
class VideoProcessing(object):
    def __init__(self):
        self.url = 'http://0.0.0.0:5050/data'
        #self.descriptors = []
        #self.age = None
        #self.gender = None
        self.gender_list = ['Female','Male']

    def process(self, img, id):
        content_type = 'image/jpeg'
        headers = {'content-type': content_type}
        imgByteArr = io.BytesIO()
        img.save(imgByteArr, format='jpeg')
        imgByteArr = imgByteArr.getvalue()

        response = requests.post(self.url, data=imgByteArr, headers=headers)
        r = response.json()
        if r['status'] == 'success':
            #font = ImageFont.truetype('static/fonts/JosefinSans-Regular.ttf', 25)
            #draw = ImageDraw.Draw(img)
            rds = redis.StrictRedis(host='localhost', port='6379', password='', decode_responses=True)
            data = r['data']
            user = rds.hgetall(id)
            descriptors = user['descriptors']
            if descriptors == '0':
                descriptors = [data['descriptor']]
            else:
                descriptors = [np.fromstring(user['descriptors'][1:-1], dtype=float, sep=', ')]
                descriptors.append(data['descriptor'])
                descriptors = [np.mean(descriptors,axis=0)]
            
            n_age = np.dot(data['ages'], list(range(100)))
            n_gender = data['genders']
            age = int(user['age'])
            gender = int(user['gender'])
            if age == -1:
                age = n_age
            else:
                age = 0.8 * age + 0.2 * n_age

            if gender == -1:
                gender = n_gender
            else:
                gender = 0.8 * gender + 0.2 * n_gender
        
            rds.hmset(id, {'age':int(np.round(age)),'gender':int(np.round(gender)),'descriptors':json.dumps(list(descriptors[0]))})
            
         #   return (f'Age: {str(int(np.round(self.age)))}|Gender: {self.gender_list[int(np.round(self.gender))]}',id)
