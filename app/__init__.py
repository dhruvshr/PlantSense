"""
Flask Initialization
"""
import os
from dotenv import load_dotenv
from flask import Flask
from torch.serialization import add_safe_globals
from app.main.config import Config
from src.model.plantsense_resnet import PlantSenseResNetBase
from src.datasets.plant_village import PlantVillage
from src.utils.device import get_device

from src.utils.model_loader import load_model

load_dotenv()

def create_app():
    # flask app init
    app = Flask(__name__)
    # secret key
    app.secret_key = os.getenv("PLANTSENSE_SECRET_KEY")

    # load config
    app.config.from_object(Config)

    # configure database
    from app.db.models import db
    app.config['SQLALCHEMY_DATABASE_URI'] = Config.PLANTSENSE_IMAGES_DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = Config.SQLALCHEMY_TRACK_MODIFICATIONS
    db.init_app(app)

    # establish app context
    with app.app_context():
        db.create_all()

        # load model as config
        app.config['MODEL'] = load_model()
        app.config['DEVICE'] = get_device()

    # load main blueprint
    from app.main import main as main_bp
    app.register_blueprint(main_bp)

    return app

if __name__ == "__main__":
    app = create_app().run(debug=True)


