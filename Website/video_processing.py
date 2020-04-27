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
        self.descriptors = []
        self.age = None
        self.gender = None
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
            font = ImageFont.truetype('static/fonts/JosefinSans-Regular.ttf', 25)
            draw = ImageDraw.Draw(img)
            data = r['data']

            self.descriptors.append(data['descriptor'])
            self.descriptors = [np.mean(self.descriptors,axis=0)]
            n_age = np.dot(data['ages'][0], list(range(101)))
            n_gender = np.argmax(data['genders'][0])
            print(data['genders'])
            if self.age is not None:
                self.age = 0.95 * self.age + 0.05 * n_age
            else:
                self.age = n_age

            if self.gender is not None:
                self.gender = 0.8 * self.gender + 0.2 * n_gender
            else:
                self.gender = n_gender

            draw.text((10, 10), 'Age: '+str(int(np.round(self.age))), (0, 255, 255), font=font)
            draw.text((10, 30), 'Gender: '+self.gender_list[int(np.round(self.gender))], (0, 255, 255), font=font)
            r = redis.StrictRedis(host='localhost', port='6379', password='', decode_responses=True)
            r.hmset(id, {'age':int(np.round(self.age)),'gender':int(np.round(self.gender)),'descriptors':json.dumps(list(self.descriptors[0]))})

        return img
