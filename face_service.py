import face_recognition
from PIL import Image
from io import BytesIO
import numpy as np
import base64
import cv2
from scipy.spatial import distance as dist
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from mtcnn import MTCNN

face_detector = MTCNN()

TOLERANCE_THRESH = 0.475



class FaceNotFoundError(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors


def recognise(img1, img2):
    img1_encoding = face_recognition.face_encodings(img1, num_jitters=10)[0]
    img2_encoding = face_recognition.face_encodings(img2, num_jitters=10)[0]
    distance = face_recognition.face_distance([img1_encoding], img2_encoding)
    results = face_recognition.compare_faces(
        [img1_encoding], img2_encoding, tolerance=TOLERANCE_THRESH)
    return results, distance


def base64_to_numpy(base64_image):
    decoded = base64.b64decode(base64_image)
    img = np.array(Image.open(BytesIO(decoded)))
    return img

def numpy_to_base64(numpy_img):
    pil_img = Image.fromarray(numpy_img)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    base64_img = base64.b64encode(buff.getvalue()).decode("utf-8")
    return base64_img 


def crop_roi(img):
    face_locations = face_recognition.face_locations(img)
    top, right, bottom, left = face_locations[0]
    cropped = img[top:bottom, left:right, ::]
    return cropped

def crop_img(img):  
    face = face_detector.detect_faces(img)
    box = face[0]['box']
    startX = box[0]
    startY = box[1]
    endX = startX + box[2]
    endY = startY + box[3]
    img = cv2.rectangle(img,(startX,startY),(endX,endY),(0,0,255),2)
    roi_color = img[startY:endY, startX:endX]
    return roi_color

def crop_roi_extend(img, extend_height, extend_width):
    face_locations = face_recognition.face_locations(img)
    top, right, bottom, left = face_locations[0]
    top -= extend_height
    bottom += extend_height
    left -= extend_width
    right += extend_width
    cropped = img[top:bottom, left:right, ::]
    return cropped


def compare_face(base64_img1, base64_img2):
    try:
        img1 = base64_to_numpy(base64_img1)
        img2 = base64_to_numpy(base64_img2)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        #img1 = crop_roi(img1)
        #img2 = crop_roi(img2)

        img1 = cv2.resize(img1, (250, 250))
        img2 = cv2.resize(img2, (250, 250))

        result, distance = recognise(img1, img2)
        return result[0], distance[0]
    except Exception as e:
        print(e)
        raise FaceNotFoundError(
            "Face recognition error please check your input", e)


def get_face_encodings(base64_img):
    img = base64_to_numpy(base64_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_roi(img)
    img = cv2.resize(img, (250, 250))
    encoding = face_recognition.face_encodings(img, num_jitters=10)[0]
    return encoding

def get_face_locations(base64_img):
    img = base64_to_numpy(base64_img)
    locations = face_recognition.face_locations(img)
    return locations


# blink detection functions

EYE_AR_THRESH = 0.25


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


def avg_eye_aspect_ratio(eye1, eye2):
    avg = np.mean([eye1, eye2])
    return avg


def is_eye_open(base64_image):
    # pre processing
    image = base64_to_numpy(base64_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_roi(image)
    image = cv2.resize(image, (250, 250))

    # landmark detection
    face_landmarks_list = face_recognition.face_landmarks(image)
    rEye = eye_aspect_ratio(face_landmarks_list[0]['right_eye'])
    lEye = eye_aspect_ratio(face_landmarks_list[0]['left_eye'])
    # aspect ratio of eye calc
    avg_aspect_ratio = avg_eye_aspect_ratio(rEye, lEye)
    if avg_aspect_ratio > EYE_AR_THRESH:
        return True
    else:
        return False

