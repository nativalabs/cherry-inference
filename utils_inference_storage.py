from roboflow import Roboflow
from google.cloud import storage
import json
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

cherry_dict = {'c-light': 'lightcoral',
               'c-dark': 'mediumvioletred',
               'c-v-dark': 'darkorchid',
               'c-negro' : 'dimgray',
               'c-f-color' : 'bisque'}

ROBOFLOW_API_KEY = "dewRu1upu5sImkWrjMbE"
ROBOFLOW_MODEL_ENDPOINT = "cherry-colors"
ROBOFLOW_VERSION = "4"

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project(ROBOFLOW_MODEL_ENDPOINT)
model = project.version(ROBOFLOW_VERSION).model
model.confidence = 20
model.overlap = 30

bucket_name = 'test-bucket-nativa'

def create_lot(bucket_name='test-bucket-nativa', lot_name='default_lot'):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    lot_folder = f'{lot_name}'
    blob = bucket.blob(lot_folder)
    blob.upload_from_string('')
    return lot_folder

def save_data_to_lot(bucket_name, lot_name, name, data, data_type):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    lot_folder = create_lot(bucket_name, lot_name)
    object_name = f'{lot_folder}{name}'

    if data_type == 'image':
        blob = bucket.blob(object_name)
        blob.upload_from_filename(data)
    elif data_type == 'json':
        blob = bucket.blob(object_name)
        blob.upload_from_string(json.dumps(data))
    elif data_type == 'text':
        blob = bucket.blob(object_name)
        blob.upload_from_string(data)

    return object_name

def is_centroid_inside_object(centroid, obj):
    return (obj['x'] - 0.5*obj['width'] <= centroid['x'] <= obj['x'] + 0.5*obj['width']) and (obj['y'] - 0.5*obj['height'] <= centroid['y'] <= obj['y'] + 0.5*obj['height'])

def parse_single_json(json_data):
    data = json.loads(json_data)

    dup = []
    for i in range(len(data['predictions'])):
        centroid = {'x': data['predictions'][i]['x'],'y': data['predictions'][i]['y']}
        classes = []
        classes.append({'x': data['predictions'][i]['x'],'y': data['predictions'][i]['y']
                   ,'width': data['predictions'][i]['width']
                   ,'height': data['predictions'][i]['height']
                   ,'class': data['predictions'][i]['class']
                   ,'confidence': data['predictions'][i]['confidence']})
        for j in range(len(data['predictions'])):
            obj = {'x': data['predictions'][j]['x'],'y': data['predictions'][j]['y']
                   ,'width': data['predictions'][j]['width']
                   ,'height': data['predictions'][j]['height']
                   ,'class': data['predictions'][j]['class']
                   ,'confidence': data['predictions'][j]['confidence']}
            if i!=j and is_centroid_inside_object(centroid,obj):
                classes.append(obj)
        if classes:
            dup.append(max(classes, key=lambda x:x['confidence']))
                
    seen = set()
    results = []
    for d in dup:
        t = tuple(d.items())
        if t not in seen:
            seen.add(t)
            results.append(d)
            
    return results

def count_n_best_confidence(json_data,n=100):
    temp = parse_single_json(json_data)
    if len(temp) < n:
        n = len(temp)
    order = sorted(temp, key=lambda d: d['confidence'],reverse=True)[0:n]
    counts = Counter(tok['class'] for tok in order)
    return dict(counts)

def plot_patches(image, json_data, _dict, counts):
    img = Image.open(image)
    predictions = parse_single_json(json_data)

    plt.ioff()
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    used_classes = set()
    
    for i in range(len(predictions)):
        color = _dict[predictions[i]['class']]
        used_classes.add(predictions[i]['class']) 
        rect = patches.Rectangle((predictions[i]['x'] - 0.5 * predictions[i]['width'],
                                  predictions[i]['y'] - 0.5 * predictions[i]['height']),
                                  predictions[i]['width'], predictions[i]['height'],
                                  linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    
    markers = [plt.Line2D([0, 0], [0, 0], color=_dict[class_name], marker='o', linestyle='') for class_name in used_classes]
    labels = [class_name+' '+str(counts[class_name]) for class_name in used_classes]
    
    plt.legend(markers, labels, numpoints=1, ncol=2, prop={'size': 7})
    plt.xlabel('x (pix)')
    plt.ylabel('y (pix)')

    return fig