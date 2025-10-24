import os
import numpy as np
import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.svm import SVC
import joblib
from PIL import Image
from warnings import simplefilter
import boto3
import io
from sklearn.preprocessing import LabelEncoder
import requests

simplefilter(action='ignore', category=FutureWarning)

S3_BUCKET_NAME = "elasticbeanstalk-eu-north-1-395451633256"
MODEL_FILE_KEY = "models/svm_model.pkl" 
LOCAL_MODEL_PATH = "/tmp/svm_model.pkl"

svm_model = None 
label_encoder = None

detector = MTCNN()
facenet = FaceNet().model 
s3_client = boto3.client('s3')

def load_model_to_memory():
    global svm_model, label_encoder
    try:
        s3_client.download_file(S3_BUCKET_NAME, MODEL_FILE_KEY, LOCAL_MODEL_PATH)
        (svm_model, label_encoder) = joblib.load(LOCAL_MODEL_PATH)
        return True
    except Exception as e:
        svm_model = None
        label_encoder = None
        return False

def extract_face_and_preprocess(image_data):
    try:
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_array = np.array(img)
        faces = detector.detect_faces(img_array)
        if not faces: return None

        x, y, w, h = faces[0]['box']
        face = img_array[y:y+h, x:x+w]
        face_img = Image.fromarray(face).resize((160, 160))
        face_array = np.array(face_img).astype('float32') / 255.0
        
        return np.expand_dims(face_array, axis=0)
    except Exception:
        return None

def get_embedding(face_array):
    return facenet.predict(face_array)[0]

def handle_prediction(image_name):
    
    if svm_model is None or label_encoder is None:
        return {"error": "Model is not loaded. Please check S3 connection and logs."}, 503

    try:
        s3_object = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=image_name)
        image_data = s3_object['Body'].read()
    except Exception:
        return {"error": f"Image file '{image_name}' not found on S3."}, 404

    face_array = extract_face_and_preprocess(image_data)
    if face_array is None:
        return {"error": "No face detected in the image."}, 400

    embedding = get_embedding(face_array)

    try:
        embedding = embedding.reshape(1, -1)
        prediction_index = svm_model.predict(embedding)[0]
        prediction_name = label_encoder.inverse_transform([prediction_index])[0]

        return {"prediction": str(prediction_name)}, 200

    except Exception as e:
        return {"error": f"Failed to predict. Model error: {e}"}, 500