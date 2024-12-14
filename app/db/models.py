"""
Flask Database Models
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone

db = SQLAlchemy()

class UploadedImage(db.Model):
    """
    Model for uploaded images
    """
    id = db.Column(
        db.Integer,
        primary_key=True
    )

    filename = db.Column(
        db.String(255),
        nullable=False
    )

    file_path = db.Column(
        db.String(255),
        nullable=False
    )

    uploaded_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.now(timezone.utc),
        index=True
    )

    predicted_class = db.Column(
        db.String(255),
        nullable=True
    )

    confidence = db.Column(
        db.Float,
        nullable=True
    )

    def __repr__(self):
        return (
            f"<UploadedImage id={self.id}, filename={self.filename}, "
            f"predicted_class={self.predicted_class}, confidence={self.confidence}>"
        )


