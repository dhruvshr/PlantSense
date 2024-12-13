"""
Flask App Routes
"""

import os
import numpy as np
from flask import Flask, current_app, flash, request, render_template, redirect, url_for
from PIL import Image
from src.model.plantsense_resnet import PlantSenseResNetBase
from src.datasets.plant_village import PlantVillage
from app.main import main

@main.route('/', methods=['GET', 'POST'])
def index():
    disease = None
    confidence = None
    uploaded_image_path = None
    insights = None

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file:
            try:
                # save uploaded file
                upload_path = os.path.join('web_app/static/images', file.filename)
                file.save(upload_path)
                uploaded_image_path = f'static/uploaded_images/{file.filename}'
                
                # Perform ML inference
                # image = Image.open(file)
                # processed_image = preprocess_image(image)
                # predictions = model.predict(processed_image)
                # predicted_class = np.argmax(predictions, axis=1)[0]
                # confidence = np.max(predictions, axis=1)[0]
                # disease = CLASS_NAMES[predicted_class]
                
                # Generate insights using the LLM layer
                # insights = generate_insights(disease, confidence)

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







