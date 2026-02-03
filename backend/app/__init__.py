from .utils import preprocess_canvas_image, get_preprocessed_pil
from .predict import predict

from flask import Flask

def create_app():
    app = Flask(__name__)
    from .routes import main
    app.register_blueprint(main)
    
    return app