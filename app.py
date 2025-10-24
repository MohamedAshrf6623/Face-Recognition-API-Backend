from flask import Flask
from controllers import api_bp
from services import load_model_to_memory

app = Flask(__name__)

load_model_to_memory()

app.register_blueprint(api_bp)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)