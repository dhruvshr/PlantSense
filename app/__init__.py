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
import torch

load_dotenv()

# # add safe globals
# add_safe_globals([PlantSenseResNetBase])

# define model path
MODEL_PATH = "saved_models/modelv1_1.pth"

def load_model(app, model_path=None):
    """
    Load the model
    """

    model = PlantSenseResNetBase(
        num_classes=PlantVillage().NUM_CLASSES
    ).to(get_device())

    # load the state dict and modify the keys to match model structure
    state_dict = torch.load(MODEL_PATH, weights_only=True)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[f"base_model.{key}"] = value
    
    # load the modified state dict
    model.load_state_dict(new_state_dict)
    model.eval()

    return model


def create_app():
    # flask app init
    app = Flask(__name__)
    # secret key
    app.secret_key = os.getenv("PLANTSENSE_SECRET_KEY")
    
    # load config
    app.config.from_object(Config)

    # load model into a global variable for easy access to later inference
    global model
    model = load_model(app)

    # TODO load database

    # load main blueprint
    from app.main import main as main_bp
    app.register_blueprint(main_bp)

    return app

if __name__ == "__main__":
    app = create_app().run(debug=True)


