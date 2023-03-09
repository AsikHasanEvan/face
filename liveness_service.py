import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import load_model
import numpy as np
import pathlib
import pickle
import cv2
from .face_service import base64_to_numpy, crop_img,crop_roi,recognise

basedir = os.path.abspath(os.path.dirname(__file__))

model_path = pathlib.Path(os.path.join(basedir, '../artifacts/liveliness/livmob_5e.h5'))
mobilenetv2 = load_model(model_path)

label_path = pathlib.Path(os.path.join(basedir, '../artifacts/liveliness/label.pickle'))
labels = pickle.loads(open(label_path, "rb").read())


def detect_liveness(img_base64):
    input_arr = base64_to_numpy(img_base64)
    input_arr = crop_img(input_arr)
    input_arr = cv2.resize(input_arr, (224, 224))
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = mobilenetv2.predict(input_arr)
    # Constructing result using higher probability
    result = "SPOOF" if labels["fake"] == np.argmax(predictions) else "LIVE"
    return result

def detect_liveness_compare(base64_img1, base64_img2):
    try:
        img1 = base64_to_numpy(base64_img1)
        img2 = base64_to_numpy(base64_img2)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        #img3 = crop_img(img1)

        input_arr = cv2.resize(img1, (224, 224))
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        predictions = mobilenetv2.predict(input_arr)
        # Constructing result using higher probability
        live = "SPOOF" if labels["fake"] == np.argmax(predictions) else "LIVE"


        
        #input_arr = img1
        img1 = cv2.resize(img1, (250, 250))
        img2 = cv2.resize(img2, (250, 250))

        
        result, distance = recognise(img1, img2)

        return result[0], distance[0], live
    except Exception as e:
        print(e)
        raise FaceNotFoundError(
            "Face recognition error please check your input", e)

