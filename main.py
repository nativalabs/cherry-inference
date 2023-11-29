from flask import Flask, jsonify
import requests
import json
import numpy as np
import smtplib
import cv2
import base64
from email.mime.multipart import MIMEMultipart
from utils_inference_storage import *
from utils_email import *
from utils_pedicel import *

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload(request):
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image_file = request.files["image"]
    image_filename = image_file.filename
    image_content = image_file.read()

    filepath = f'/tmp/{image_filename}'
    with open(filepath, 'wb') as f:
        f.write(image_content)

    image_code = image_filename.replace('.jpg','')
    lot_name = f'cherry/{image_code}/'

    save_data_to_lot(bucket_name, lot_name, f'{image_filename}', filepath, 'image')

    #----------------------------------COLOR-----------------------

    model.confidence = 20
    model.overlap = 30
    predictions = model.predict(filepath)
    predictions.save(f'/tmp/predictions_{image_filename}')

    save_data_to_lot(bucket_name, lot_name, f'predictions_{image_filename}', f'/tmp/predictions_{image_filename}', 'image')

    results = predictions.json()

    class_count = count_n_best_confidence(json.dumps(results))
    body = f'Cherry color predictions for {image_filename}:\n'
    for idx in class_count.keys():
        body += f'{idx}: {class_count[idx]}\n'
    body += '\n'

    save_data_to_lot(bucket_name, lot_name, f'counts_{image_code}.json', body, 'json')

    plot_predictions = plot_patches(image_file,json.dumps(results), cherry_dict, class_count)
    plt.savefig(f'/tmp/predictions_{image_code}.jpg')
    save_data_to_lot(bucket_name, lot_name, f'predictions_{image_code}.jpg', f'/tmp/predictions_{image_code}.jpg', 'image')

    #-------------------------PEDICEL-------------------
    compressed_image_content = compress_image(image_content, target_size_bytes=1500000)
    base64_img_data = base64.b64encode(compressed_image_content)

    pedicel_predictions = requests.post(pedicel_prediction_url, data=base64_img_data, headers={
        "Content-Type": "application/x-www-form-urlencoded"})

    #with open('/tmp/compressed_image.jpg', 'wb') as f:
    #    f.write(compressed_image_content)

    #pedicel_predictions = pedicel_model.predict('/tmp/compressed_image.jpg')
    #pedicel_predictions.save(f'/tmp/pedicel_predictions_{image_filename}')

    #save_data_to_lot(bucket_name, lot_name, f'pedicel_predictions_{image_filename}', f'/tmp/pedicel_predictions_{image_filename}', 'image')

    pedicel_results = pedicel_predictions.json()

    save_data_to_lot(bucket_name, lot_name, f'json_pedicel_test.json', json.dumps(pedicel_results), 'json')

    ped_count = count_n_best_confidence(json.dumps(pedicel_results))
    body += f'Cherry pedicel predictions for {image_filename}:\n'
    for idx in ped_count.keys():
        body += f'{idx}: {ped_count[idx]}\n'

    plot_pedicel = plot_patches(image_file,json.dumps(pedicel_results), pedicel_dict, ped_count)
    plt.savefig(f'/tmp/pedicel_predictions_{image_code}.jpg')
    save_data_to_lot(bucket_name, lot_name, f'pedicel_predictions_{image_code}.jpg', f'/tmp/pedicel_predictions_{image_code}.jpg', 'image')

    #-------------------------------MAIL---------------------------

    msg = MIMEMultipart()
    msg["From"] = SMTP_USERNAME
    msg["To"] = recipient_list
    msg["Subject"] = f"Resultados de muestra {image_filename}"

    with open(f'/tmp/predictions_{image_filename}', 'rb') as mail_image:
        mail_image_data = mail_image.read()
    
    with open(f'/tmp/predictions_{image_code}.jpg', 'rb') as plot_image:
        plot_image_data = plot_image.read()

    with open(f'/tmp/pedicel_predictions_{image_filename}', 'rb') as pedicel_image:
        pedicel_image_data = pedicel_image.read()
    
    add_to_mail(msg, body, 'text',f'counts_{image_code}.json')
    add_to_mail(msg, json.dumps(results),'json',f'predictions_{image_code}.json')
    add_to_mail(msg, mail_image_data,'image',f'predictions_{image_filename}')
    add_to_mail(msg, image_content, 'image', image_filename)
    add_to_mail(msg, plot_image_data, 'image', f'predictions_{image_code}.jpg')
    add_to_mail(msg, pedicel_image_data, 'image', f'pedicel_predictions_{image_code}.jpg')

    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(SMTP_USERNAME, smtp_password)
    server.send_message(msg)
    server.quit()

    return f'Inference done on: {image_filename}\n'

    
    

    

