"""
Flask Initialization
"""

from dotenv import load_dotenv
from flask import Flask
from flask_migrate import Migrate
from app.main.config import Config
from src.utils.device import get_device

from src.utils.model_loader import load_model

load_dotenv()


def create_app():
    # flask app init
    app = Flask(__name__)

    # load config
    app.config.from_object(Config)

    # secret key
    app.secret_key = Config.SECRET_KEY

    # print(f"App secret key set: {bool(app.secret_key)}")

    # configure database
    from app.db.models import db
    app.config['SQLALCHEMY_DATABASE_URI'] = Config.PLANTSENSE_IMAGES_DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = Config.SQLALCHEMY_TRACK_MODIFICATIONS
    db.init_app(app)
    migrate = Migrate(app, db)

    # establish app context
    with app.app_context():
        db.create_all()

        # load model as config
        app.config['MODEL'] = load_model()
        app.config['DEVICE'] = get_device()

    # load main blueprint
    from app.main import main as main_bp
    app.register_blueprint(main_bp)

    # load utils
    from app.utils import encryption
    app.jinja_env.globals['encrypt_id'] = encryption.encrypt_id
    app.jinja_env.globals['decrypt_id'] = encryption.decrypt_id

    return app

if __name__ == "__main__":
    app = create_app().run(debug=True)


