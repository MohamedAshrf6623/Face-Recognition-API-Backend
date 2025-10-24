from flask import Blueprint, request, jsonify
from services import load_model_from_s3, handle_prediction, svm_model, s3_client
import os

api_bp = Blueprint('api_bp', __name__)

@api_bp.route("/reload_eng_mo", methods=["POST"])
def reload_model_controller():
    success = load_model_from_s3()
    if success:
        return jsonify({"status": "success", "message": "Model reloaded successfully."}), 200
    return jsonify({"status": "error", "message": "Failed to reload model."}), 500

@api_bp.route("/")
def home():
    if not svm_model:
        return "<h1>Error: Model is not loaded. Check server logs and S3 connection.</h1>", 500
    return "<h1>Face Recognition Server is running and model is loaded.</h1>"

@api_bp.route("/get-latest-image")
def get_latest_image_controller():
    FILE_NAME_TO_CHECK = "engmo.jpg"
    try:
        s3_object_metadata = s3_client.head_object(
            Bucket=S3_BUCKET_NAME, Key=FILE_NAME_TO_CHECK
        )
        last_modified_timestamp = s3_object_metadata['LastModified'].isoformat()
        return jsonify({
            "image_name": FILE_NAME_TO_CHECK,
            "last_modified": last_modified_timestamp
        })
    except s3_client.exceptions.NoSuchKey:
        return jsonify({"error": "File not found yet."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route("/predict", methods=["POST"])
def predict_route():
    if not svm_model:
        return jsonify({"error": "Model is not loaded"}), 503

    data = request.get_json()
    if not data or "image_name" not in data:
        return jsonify({"error": "Please provide the 'image_name'"}), 400

    image_name = data["image_name"]
    
    response_data, status_code = handle_prediction(image_name)

    return jsonify(response_data), status_code