from app import create_app
from app.db.models import db, UploadedImage, Message
import os

def clear_images():
    app = create_app()
    with app.app_context():
        # Check for existing images
        image_count = UploadedImage.query.count()
        message_count = Message.query.count()
        
        if image_count == 0 and message_count == 0:
            print("No images or messages found in the database!")
            return
            
        print(f"Found {image_count} images and {message_count} messages in the database.")
        
        # Delete all messages first (due to foreign key constraints)
        Message.query.delete()
        
        # Delete all image records from database
        UploadedImage.query.delete()
        
        # Commit the changes
        db.session.commit()
        
        # Clear the actual image files from the uploads directory
        upload_dir = 'app/static/uploaded_images'
        file_count = 0
        for filename in os.listdir(upload_dir):
            if filename != '.gitkeep':  # Preserve .gitkeep file if it exists
                file_path = os.path.join(upload_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        file_count += 1
                except Exception as e:
                    print(f'Error: {e}')
        
        print(f"Cleared {file_count} image files from {upload_dir}")
        print("All images and related records have been cleared!")

if __name__ == "__main__":
    clear_images()