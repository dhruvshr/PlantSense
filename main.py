import os

from flask import (
    Flask,
    render_template,
    request
)

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    disease = None
    insights = None

    # get request
    if request.method == "POST":
        file = request.files.get('image')
        if file and file.filename:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)

            # TODO: link these to model inference and insights
            disease = run_inference(img_path)
            insights = generate_insights(disease)

    return render_template('index.html', disease=disease, insights=insights)

def main():
    app.run(
        port=int(os.environ.get('PORT', 80)), 
        debug=True
    )

if __name__ == "__main__":
    main()
