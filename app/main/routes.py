"""
Flask App Routes
"""

import os
from markdown import markdown
from flask import session, current_app, flash, request, render_template, redirect, url_for
from PIL import Image
from app.main import main
from app.db.models import UploadedImage, Message, db
from src.utils.inference import infer_image
from src.llm.insights_engine import InsightsEngine
from app.utils.encryption import encrypt_id, decrypt_id


UPLOADED_IMAGES_DIR = 'app/static/uploaded_images'

# landing page
@main.route('/', methods=['GET', 'POST'])
def index():
    model = current_app.config['MODEL']
    device = current_app.config['DEVICE']

    # Get previous images with their messages
    previous_images = UploadedImage.query.order_by(UploadedImage.uploaded_at.desc()).all()

    # clear session variables
    session.clear()

    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            image_path = os.path.join(UPLOADED_IMAGES_DIR, image.filename)
            image.save(image_path)

            # run inference on image
            predicted_class, confidence = infer_image(image_path, model, device) 
            print(f"Image Path: {image_path}")
            print(f"Predicted Class: {predicted_class}, Confidence: {confidence}")

            uploaded_image = UploadedImage(
                filename=image.filename,
                file_path=image_path,
                predicted_class=predicted_class,
                confidence=confidence
            )
            db.session.add(uploaded_image)
            db.session.commit()

            session['image_id'] = uploaded_image.id
            encrypted_image_id = encrypt_id(uploaded_image.id)

            # redirect to chat page with encrypted image id
            return redirect(url_for('main.chat', encrypted_image_id=encrypted_image_id))
        
        elif 'image_id' in request.args:
            # Set the image_id in session when clicking a previous image
            image_id = request.args.get('image_id')
            session['image_id'] = int(image_id)
            encrypted_image_id = encrypt_id(image_id)
            return redirect(url_for('main.chat', encrypted_image_id=encrypted_image_id))

        return redirect(url_for('main.index'))
    
    return render_template('index.html', previous_images=previous_images)

@main.route('/chat/<string:encrypted_image_id>', methods=['GET', 'POST'])
def chat(encrypted_image_id):
    image_id = decrypt_id(encrypted_image_id)
    uploaded_image = UploadedImage.query.get_or_404(image_id)
    insights_engine = InsightsEngine()

    if uploaded_image:
        session['image_path'] = uploaded_image.file_path
        session['image_id'] = image_id
        encrypted_image_id = encrypt_id(image_id)

        # Get all messages for this image
        messages = uploaded_image.messages

        if not messages:
            # generate initial insights and store in db
            initial_insights = markdown(
                insights_engine.generate_insights(
                    uploaded_image.predicted_class,
                    uploaded_image.confidence
                )
            )

            new_ps_message = Message(
                image_id=uploaded_image.id,
                sender='PlantSense',
                message=initial_insights
            )
            db.session.add(new_ps_message)
            db.session.commit()

            # append initial message to messages list
            messages = [new_ps_message]

        if request.method == 'POST':
            user_feedback = request.form['user_feedback']

            if user_feedback:
                new_user_message = Message(
                    image_id=uploaded_image.id,
                    sender='User',
                    message=user_feedback
                )
                db.session.add(new_user_message)
                db.session.commit()

                # generate follow-up insights
                follow_up_insights = markdown(
                    insights_engine.generate_insights(
                        uploaded_image.predicted_class,
                        uploaded_image.confidence,
                        user_feedback
                    )
                )

                new_ps_message = Message(
                    image_id=uploaded_image.id,
                    sender='PlantSense',
                    message=follow_up_insights
                )
                db.session.add(new_ps_message)
                db.session.commit()

                return redirect(url_for('main.chat', encrypted_image_id=encrypted_image_id))

    # Get previous images for the grid
    previous_images = UploadedImage.query.order_by(UploadedImage.uploaded_at.desc()).all()
    return render_template(
        'chat.html', 
        uploaded_image=uploaded_image, 
        messages=messages, 
        previous_images=previous_images
    )

