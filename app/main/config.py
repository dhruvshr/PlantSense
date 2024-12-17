"""
Web App Config
"""

import os
import dotenv
from cryptography.fernet import Fernet
class Config():

    # secret key
    SECRET_KEY = os.getenv("PLANTSENSE_SECRET_KEY")

    # OpenAI API Key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Model
    PLANTSENSE_MODEL_PATH = os.getenv("MODEL_PATH")
    PLANTSENSE_MODEL_VERSION = os.getenv("MODEL_VERSION")

    # Device
    DEVICE = os.getenv("DEVICE")

    # Checkpoint
    PLANTSENSE_CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")

    # Database
    PLANTSENSE_IMAGES_DATABASE_URI = os.getenv("PLANTSENSE_IMAGES_DATABASE_URI")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Encryption
    FERNET_KEY = Fernet.generate_key()


if __name__ == "__main__":
    config = Config()
    print(config.OPENAI_API_KEY)
    print(config.PLANTSENSE_MODEL_PATH)
    print(config.PLANTSENSE_MODEL_VERSION)