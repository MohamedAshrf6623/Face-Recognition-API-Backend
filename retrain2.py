import boto3
import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
from sklearn.svm import SVC
import joblib
from sklearn.preprocessing import LabelEncoder
import requests
from PIL import Image
import pickle # ### هنستخدمه لحفظ الكاش

# --- 1. إعدادات Firebase (لم نعد بحاجتها) ---
# ... (محذوف) ...

# --- 2. إعدادات AWS S3 ---
S3_BUCKET_NAME = "elasticbeanstalk-eu-north-1-395451633256"
TRAINING_DIR_PREFIX = 'training_data/'
MODEL_FILE_KEY = "models/svm_model.pkl"
# ### ملف جديد لحفظ الكاش
CACHE_FILE_KEY = "models/embeddings_cache.pkl"

# --- 3. إعدادات سيرفر الإنتاج (Elastic Beanstalk) ---
RELOAD_URL = "http://51.21.255.1:5000/reload_eng_mo"

# --- 4. تهيئة الأدوات ---
print("Initializing models (FaceNet, MTCNN)...")
facenet = FaceNet().model
detector = MTCNN()

print("Initializing AWS S3 client...")
s3_client = boto3.client('s3')

def get_face_embedding(image_bytes):
    """(هذه الدالة لم تتغير)"""
    try:
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        faces = detector.detect_faces(img)
        if len(faces) == 0:
            return None

        x, y, w, h = faces[0]['box']
        face = img[y:y+h, x:x+w]
        face_img = Image.fromarray(face).resize((160, 160))
        face_array = np.array(face_img).astype('float32') / 255.0
        face_array = np.expand_dims(face_array, axis=0)

        embedding = facenet.predict(face_array)[0]
        return embedding
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def load_cache():
    """يحاول تحميل الكاش القديم من S3"""
    print(f"Loading embeddings cache from S3 ({CACHE_FILE_KEY})...")
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=CACHE_FILE_KEY)
        cache_data = pickle.loads(response['Body'].read())
        print(f"Cache loaded. Found {len(cache_data)} cached items.")
        return cache_data
    except s3_client.exceptions.NoSuchKey:
        print("No cache file found. Starting fresh.")
        return {} # لا يوجد كاش، نبدأ من الصفر
    except Exception as e:
        print(f"Error loading cache: {e}. Starting fresh.")
        return {}

def save_cache(cache_data):
    """يحفظ الكاش الجديد إلى S3"""
    print(f"Saving new cache ({len(cache_data)} items) to S3 ({CACHE_FILE_KEY})...")
    try:
        with open("cache.pkl", "wb") as f:
            pickle.dump(cache_data, f)
        s3_client.upload_file("cache.pkl", S3_BUCKET_NAME, CACHE_FILE_KEY)
        os.remove("cache.pkl")
        print("Cache saved successfully.")
    except Exception as e:
        print(f"Error saving cache: {e}")

def run_retraining():
    print("--- Starting Retraining Process ---")

    # --- أ. تحميل البيانات (باستخدام الكاش) ---
    print(f"Fetching training data from S3 Bucket ({S3_BUCKET_NAME}/{TRAINING_DIR_PREFIX})...")

    # 1. تحميل الكاش القديم
    cached_data = load_cache()

    # 2. إعداد القوائم الجديدة
    current_embeddings = []
    current_labels = []
    new_cache_data = {} # الكاش الجديد الذي سنبنيه

    # 3. جلب كل الملفات الحالية من S3
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=TRAINING_DIR_PREFIX)

    s3_objects = []
    for page in pages:
        s3_objects.extend(page.get('Contents', []))

    processed_count = 0
    cache_hit_count = 0

    for obj in s3_objects:
        s3_key = obj['Key']
        s3_size = obj['Size']
        s3_etag = obj['ETag'] # "بصمة" الملف

        if not s3_key.endswith('/') and s3_size > 0:
            try:
                label = s3_key.split('/')[-2]
                embedding = None

                # 4. التحقق من الكاش
                if s3_key in cached_data and cached_data[s3_key]['etag'] == s3_etag:
                    # Cache Hit! الملف موجود ولم يتغير
                    embedding = cached_data[s3_key]['embedding']
                    label = cached_data[s3_key]['label'] # تأكيداً
                    cache_hit_count += 1
                else:
                    # Cache Miss! ملف جديد أو تم تعديله
                    print(f"Cache miss for {s3_key}. Processing...")
                    response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                    image_data = response['Body'].read()
                    embedding = get_face_embedding(image_data)
                    processed_count += 1

                # 5. إضافة البيانات (لو صالحة)
                if embedding is not None:
                    current_embeddings.append(embedding)
                    current_labels.append(label)
                    # إضافة/تحديث بيانات الملف في الكاش الجديد
                    new_cache_data[s3_key] = {
                        'etag': s3_etag,
                        'embedding': embedding,
                        'label': label
                    }
                else:
                    if s3_key not in cached_data: # نطبع التحذير فقط لو ملف جديد
                         print(f"Warning: No face found in {s3_key}. Skipping.")

            except Exception as e:
                print(f"Error processing {s3_key}: {e}")

    if not current_labels:
        print("No training data found or processed. Exiting.")
        return

    print("--- Data Fetching Summary ---")
    print(f"Total images processed: {len(current_labels)}")
    print(f"Loaded from cache (Cache Hits): {cache_hit_count}")
    print(f"Newly processed (Cache Misses): {processed_count}")
    print(f"Unique people: {len(set(current_labels))}")

    # --- ب. تدريب الموديل ---
    print("Training new SVM model...")

    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(current_labels)

    new_svm_model = SVC(kernel='linear', probability=True)
    new_svm_model.fit(current_embeddings, labels_encoded)

    local_model_path = "svm_model_new.pkl"
    joblib.dump((new_svm_model, encoder), local_model_path)
    print(f"New model and encoder saved locally as {local_model_path}.")

    # --- ج. رفع الموديل والكاش إلى S3 ---
    print(f"Uploading new model to S3 ({MODEL_FILE_KEY})...")
    try:
        s3_client.upload_file(local_model_path, S3_BUCKET_NAME, MODEL_FILE_KEY)
        print("✅ Model successfully uploaded to S3.")

        # ### خطوة جديدة: حفظ الكاش الجديد
        save_cache(new_cache_data)

    except Exception as e:
        print(f"❌ Error uploading to S3: {e}")
        return

    # --- د. إخبار السيرفر بإعادة التحميل ---
    print(f"Sending reload request to server: {RELOAD_URL}...")
    try:
        response = requests.post(RELOAD_URL)
        if response.status_code == 200:
            print("✅ Server responded: Model reloaded successfully!")
        else:
            print(f"⚠ Server responded with error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Failed to send reload request: {e}")

    print("--- Retraining Process Finished ---")

if _name_ == "_main_":
    run_retraining()