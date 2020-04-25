from User import User
from VideoProcessing import VideoProcessing
import psycopg2 as pg
import pandas as pd
import numpy as np
import turicreate as tc
import time
from collections import Counter

class CVRecommender(object):
    def __init__(self):
        self.user = User()
        self.vp = VideoProcessing()
        self.connection = self.connectToDB()

    def connectToDB(self):
        with open('credentials.txt') as creds:
            db, user, host, port = creds.read().splitlines()
        return pg.connect(dbname=db, user=user, host=host, port=port)

    def testConnection(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute('SELECT 1')
            cursor.close()
        except Exception as e:
            self.connection = self.connectToDB()

    def yieldFrames(self):
        while self.vp.isActive:
            frame, self.user = self.vp.get_frame(self.user)
            yield (b'--frame\r\n'+ b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    def makeOrder(self, data):
        cursor = self.connection.cursor()
        cursor.execute("INSERT into orders(user_id) VALUES (%s) RETURNING order_id;", (self.user.ID,))
        order_id = cursor.fetchone()[0]
        for item_id in data:
            cursor.execute("INSERT into orders_items(order_id,item_id) VALUES (%s, %s);",(order_id,item_id))
        self.connection.commit()
        cursor.close()

    def getUserID(self):

        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM (SELECT user_id, vector, \
                        (cube_distance(users.vector, cube(%s))) AS dist FROM users) AS t \
                        WHERE dist < 0.45 ORDER BY dist LIMIT 1;",(list(self.user.descriptors[0]),))
        # если таких совпадений по порогу несколько (что практически невозможно), то берем самого похожего

        id = cursor.fetchone()
        if id:
            print(f'you are in the base: id {id[0]}')
            # если пользователь есть в базе, то обновляем его векторное представление новой фотографией (усредняем, чтобы лучше распознавалось потом)
            #res = np.fromstring(result[1][1:-1], dtype=float, sep=', ')
            #upd = list(np.mean([res,descriptor],axis=0))
            #cursor.execute(f"UPDATE users SET vector = cube(array{upd}) WHERE user_id = {result[0]}")#update
            #connection.commit()
            cursor.close()
            self.user.ID = id[0]
        else:
            print('you are not in the base yet')
            # если нет в базе, то просто добавляем его
            age_list = [(0,17),(18,24),(25,34),(35,44),(45,54),(55,64),(65,100)]
            for i, a in enumerate(age_list):
                if a[0] <= self.user.age <= a[1]:
                    n_age = i+1
                    break
            cursor.execute("INSERT INTO users(vector,gender,age_group_id) \
                            VALUES(cube(%s),%s,%s) RETURNING user_id",
                            (list(self.user.descriptors[0]),bool(self.user.gender),int(n_age)))

            id = cursor.fetchone()[0]
            self.connection.commit()
            cursor.close()
            self.user.ID = id


    def getOrders(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT o.order_id, array_agg(i.item_name), array_agg(i.item_id)\
                        FROM orders o,items i, orders_items oi \
                        WHERE o.user_id = %s AND oi.item_id = i.item_id AND o.order_id = oi.order_id \
                        GROUP BY o.order_id",
                        (self.user.ID,))
        pr_orders = cursor.fetchall()
        pr_orders.sort(reverse=True)
        orders = []

        for order in pr_orders:
            ord = {'order_id':order[0],'order_data':[]}
            for i in range(len(order[1])):
                ord['order_data'].append({'item_id':order[2][i],'item_name':order[1][i]})
            orders.append(ord)

        cursor.close()
        return orders

    def getItems(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM items")
        items = cursor.fetchall()
        cursor.close()
        return items

    def getRecommendations(self, n_recs):
        cursor = self.connection.cursor()
        cursor.execute("SELECT i.item_id \
                        FROM items i, orders_items oi \
                        WHERE i.item_id = oi.item_id \
                        GROUP BY i.item_id \
                        ORDER BY count(i.item_id) DESC \
                        LIMIT %s",(n_recs,))
        result = cursor.fetchall()
        most_popular_dishes = [tpl[0] for tpl in result]

        cursor.execute("SELECT o.user_id, oi.item_id, count(oi.item_id) \
                        FROM orders_items oi, orders o \
                        WHERE oi.order_id = o.order_id \
                        GROUP BY o.user_id, oi.item_id")
        user_item_counts = pd.DataFrame(cursor.fetchall())
        user_item_counts.columns = ['user_id','item_id','count']
        user_item_counts['count'] = user_item_counts['count']/(user_item_counts['count'].max()-user_item_counts['count'].min())


        cursor.execute("SELECT item_id,item_name FROM items")
        ids_names = cursor.fetchall()
        items = [item[0] for item in ids_names]
        items_names = np.array([item[1] for item in ids_names])

        cursor.execute("SELECT item_id, tag_id FROM items_tags")
        items_tags = pd.DataFrame(cursor.fetchall())
        items_tags.columns = ['item_id','tag_id']
        items_tags['v']=1
        items_tags = items_tags.pivot_table(index='item_id',columns='tag_id', values='v').fillna(0).reset_index()
        items_tags = tc.SFrame(items_tags)

        user_id = 'user_id'
        item_id = 'item_id'
        target = 'count'

        user_item_counts = tc.SFrame(user_item_counts)
        t1=time.time()
        #долго!
        icr = tc.item_content_recommender.create(items_tags, item_id = item_id,
                                                observation_data = user_item_counts,
                                                user_id = user_id, target = target,
                                                verbose = False)
        t2=time.time()
        rfr = tc.ranking_factorization_recommender.create(user_item_counts,
                                                        user_id = user_id, item_id = item_id,
                                                        target = target,verbose = False)

        t3=time.time()

        item_based=icr.recommend([self.user.ID], n_recs)
        interaction_based=rfr.recommend([self.user.ID], n_recs)

        cursor.execute("SELECT vector, age_group_id, gender FROM users \
                        WHERE user_id = %s",(self.user.ID,))
        curr_user = cursor.fetchone()
        vec = list(np.fromstring(curr_user[0][1:-1], dtype=float, sep=', '))

        cursor.execute("SELECT user_id, dist FROM (SELECT user_id, \
                        (cube_distance(users.vector, cube(%s))) as dist \
                        FROM users) as t ORDER BY dist LIMIT %s;",(vec,n_recs+1))
        closest_users = cursor.fetchall()
        print(closest_users)

        looks_foods=[]
        for cl_u in closest_users[1:]:
            cursor.execute("SELECT i.item_id\
                            FROM orders o,items i, orders_items oi \
                            WHERE o.user_id = %s AND oi.item_id = i.item_id \
                            AND o.order_id = oi.order_id",(cl_u[0],))
            pr_orders = cursor.fetchall()
            looks_foods= looks_foods+pr_orders


        cursor.execute("SELECT user_id, dist FROM (SELECT user_id, \
                        (cube_distance(users.vector, cube(%s))) as dist \
                        FROM users where gender=%s and age_group_id=%s) as t \
                        ORDER BY dist LIMIT %s;",(vec,curr_user[2],curr_user[1],n_recs+1))
        closest_users = cursor.fetchall()
        print(closest_users)

        demographic =[]
        for cl_u in closest_users[1:]:
            cursor.execute("SELECT i.item_id \
                            FROM orders o,items i, orders_items oi \
                            WHERE o.user_id = %s AND oi.item_id = i.item_id \
                            AND o.order_id = oi.order_id",(cl_u[0],))
            pr_orders = cursor.fetchall()
            demographic = demographic+pr_orders

        rec_looks = [r[0] for r in looks_foods]
        c_looks = Counter(rec_looks).most_common(n_recs)
        rec_looks = [r[0] for r in c_looks]

        rec_dem = [r[0] for r in demographic]
        c_dem = Counter(rec_dem).most_common(n_recs)
        rec_dem = [r[0] for r in c_dem]

        recs=[np.array(most_popular_dishes[:n_recs],dtype=int),
            np.array(rec_looks,dtype=int),
            np.array(rec_dem,dtype=int),
            np.array(item_based['item_id'][:n_recs],dtype=int),
            np.array(interaction_based['item_id'][:n_recs],dtype=int)]

        titles = ['Most Popular Dishes:',
                    'People Who Look Like You, Love:',
                    'People Of Your Age And Gender, Love:',
                    'Similar By Content:',
                    "Similar By Interactions:"]

        cursor.execute("SELECT count(order_id) FROM orders WHERE user_id=%s",(self.user.ID,))
        num_orders = cursor.fetchone()[0]

        if num_orders<1:
            recs=recs[:3]
            titles=titles[:3]

        recommendations = []
        for t,r in zip(titles,recs):
            recommendations.append({'title':t, 'recs':[{'item_id':k, 'item_name':v} for k, v in zip(r, items_names[r-1])]})
        print(t2-t1,t3-t2)
        cursor.close()
        return recommendations
