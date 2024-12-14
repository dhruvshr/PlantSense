"""
Flask App Routes
"""

import os
import numpy as np
from markdown import markdown
from flask import session, current_app, flash, request, render_template, redirect, url_for
from PIL import Image
from app.main import main
from app.db.models import UploadedImage, InferenceConversation, db
from src.utils.inference import infer_image
from src.llm.insights_engine import InsightsEngine

@main.route('/', methods=['GET', 'POST'])
def index():
    # Retrieve the uploaded image and conversation history from the database
    uploaded_image = None
    conversation = []

    if session.get('image_id'):
        # Fetch the uploaded image and its related conversation
        uploaded_image = UploadedImage.query.get(session['image_id'])
        conversation = InferenceConversation.query.filter_by(
            image_id=uploaded_image.id
        ).order_by(
            InferenceConversation.timestamp
        ).all()

        # conversation = Conversation.query.filter_by(image_id=uploaded_image.id).order_by(Conversation.timestamp).paginate(page=1, per_page=10).items

    model = current_app.config['MODEL']
    insights_engine = InsightsEngine()

    if request.method == 'POST':
        # check if form includes an image
        if 'image' in request.files:
            file = request.files['image']

            # check if file was selected
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            
            if file:
                try:
                    # save uploaded file
                    upload_dir = 'app/static/uploaded_images'
                    os.makedirs(upload_dir, exist_ok=True)
                    uploaded_image_path = os.path.join(upload_dir, file.filename)
                    file.save(uploaded_image_path)
                    
                    # perform image inference
                    # image = Image.open(file)
                    predicted_class, confidence = infer_image(
                        uploaded_image_path,
                        model,
                        current_app.config['DEVICE']
                    )

                    # save metadata and llm response to the database
                    new_image = UploadedImage(
                        filename=file.filename,
                        file_path=uploaded_image_path,
                        predicted_class=predicted_class,
                        confidence=confidence
                    )
                    db.session.add(new_image)
                    db.session.commit()

                    print(f"\nnew image id: {new_image.id}\n")

                    # save image id to session
                    if new_image.id and new_image.id:
                        try:
                            session['image_id'] = str(new_image.id)
                            print(f"\nimage id saved to session: {session['image_id']}\n")
                        except Exception as e:
                            print(f"Error saving image id to session: {e}")
                            flash(f"Error saving image id to session: {e}")
                            return redirect(url_for('main.index'))

                    # generating conversational insights with llm
                    initial_insights = markdown(
                        insights_engine.generate_insights(
                            predicted_class, 
                            confidence
                        )
                    )

                    bot_message = InferenceConversation(
                        image_id=uploaded_image.id,
                        sender="Bot",
                        message=initial_insights
                    )
                    db.session.add(bot_message)
                    db.session.commit()
                    

                except Exception as e:
                    flash(f'An error occurred: {e}')
                    return redirect(request.url)
                
        elif 'user_feedback' in request.form:
            # Handle user feedback for follow-up questions
            user_feedback = request.form['user_feedback']
            if session.get('image_id'):
                # Save the user's message to the conversation
                user_message = InferenceConversation(
                    image_id=session['image_id'],
                    sender="User",
                    message=user_feedback
                )
                db.session.add(user_message)
                db.session.commit()

                # Generate a response based on the feedback
                uploaded_image = UploadedImage.query.get(session['image_id'])
                follow_up_response = insights_engine.generate_insights(
                    uploaded_image.predicted_class,
                    uploaded_image.confidence,
                    user_feedback=user_feedback
                )
                html_response = markdown(follow_up_response)

                # Save the bot's response to the conversation
                bot_response = InferenceConversation(
                    image_id=session['image_id'],
                    sender="Bot",
                    message=html_response
                )
                db.session.add(bot_response)
                db.session.commit()


    if uploaded_image:
        conversation = InferenceConversation.query.filter_by(
            image_id=uploaded_image.id
        ).order_by(
            InferenceConversation.timestamp
        ).all()

    return render_template(
        'index.html',
        # predicted_class=predicted_class,
        # confidence=confidence,
        uploaded_image_path=uploaded_image.file_path if uploaded_image else None,
        conversation=conversation
    )







