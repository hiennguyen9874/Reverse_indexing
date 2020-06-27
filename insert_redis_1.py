import time
import redis
from tqdm import tqdm
from datetime import datetime

r = redis.StrictRedis(host='localhost', port=6379, db=0)
pipe = r.pipeline()
for key in r.scan_iter("*"):
    pipe.delete(key)
pipe.execute()

datasets = 'PA-100K'

if datasets == 'Market-1501':
    from data import Market1501_Attribute
    datasource = Market1501_Attribute(root_dir='/home/hien/Documents/datasets', download=False, extract=False, re_label_on_train=False)
    all_data = datasource.get_data('train') + datasource.get_data('query') + datasource.get_data('gallery')
    all_attribute = datasource.get_attribute('train')[0]
    all_attribute.update(datasource.get_attribute('query')[0])
    attribute_label = datasource.get_attribute('train')[1]

    start_time = time.time()
    attribute_label = datasource.get_attribute('train')[1]
    start_index = int(datetime.now().strftime(r'%m%d%H%M%S'))
    for index in range(len(all_data)):
        if all_data[index][1] not in all_attribute.keys():
            continue
        save_dict = {
            'path': all_data[index][0],
            'person_id': all_data[index][1],
            'camera_id': all_data[index][2]}

        for index_attribute in range(len(attribute_label)):
            save_dict.update({attribute_label[index_attribute]: int(all_attribute[all_data[index][1]][index_attribute])})
        pipe.hmset(start_index+index, save_dict)
        for key, value in save_dict.items():
            if key != 'path' and key != 'person_id' and key != 'camera_id':
                if value == 1:
                    pipe.lpush('1_' + key, start_index+index)
                elif value == 0:
                    pipe.lpush('0_' + key, start_index+index)
    pipe.execute()
    print("time excute: %s seconds" % (time.time() - start_time))

elif datasets == 'PA-100K':
    from data import PA_100K

    datasource = PA_100K(root_dir='/home/hien/Documents/datasets', download=False, extract=False)
    all_data = datasource.get_data('train')[0] + datasource.get_data('val')[0] + datasource.get_data('test')[0]
    attribute_label = datasource.get_data('train')[1]

    start_time = time.time()
    start_index = int(datetime.now().strftime(r'%m%d%H%M%S'))
    with tqdm(total=len(all_data)) as pbar:
        for index in range(len(all_data)):
            save_dict = {
                'path': all_data[index][0]
            }
            for index_attribute in range(len(attribute_label)):
                save_dict.update({attribute_label[index_attribute]: int(all_data[index][1][index_attribute])})
            pipe.hmset(start_index+index, save_dict)

            for key, value in save_dict.items():
                if key != 'path':
                    if value == 1:
                        pipe.lpush('1_' + key, start_index+index)
                    elif value == 0:
                        pipe.lpush('0_' + key, start_index+index)
            pbar.update(1)
    pipe.execute()
    print("time excute: %s seconds" % (time.time() - start_time))

# time excute: 74.87690019607544 seconds