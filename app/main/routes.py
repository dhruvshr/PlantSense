"""
Flask App Routes
"""

import os
import numpy as np
from flask import Flask, current_app, flash, request, render_template, redirect, url_for
from PIL import Image
from app.main import main
from app.db.models import UploadedImage, db
from src.utils.inference import infer_image
from src.llm.insights_engine import InsightsEngine

@main.route('/', methods=['GET', 'POST'])
def index():
    disease = None
    confidence = None
    uploaded_image_path = None
    insights = None
    model = current_app.config['MODEL']
    insights_engine = InsightsEngine()

    if request.method == 'POST':
        # check if form includes an image
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
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

                # generating conversational insights with llm
                insights = insights_engine.generate_insights(disease, confidence)

                # save metadata and llm response to the database
                new_image = UploadedImage(
                    filename=file.filename,
                    file_path=uploaded_image_path,
                    predicted_class=predicted_class,
                    confidence=confidence
                )
                db.session.add(new_image)
                db.session.commit()

            except Exception as e:
                flash(f'An error occurred: {e}')
                return redirect(request.url)

    return render_template(
        'index.html',
        disease=disease,
        confidence=confidence,
        uploaded_image_path=uploaded_image_path,
        insights=insights
    )







