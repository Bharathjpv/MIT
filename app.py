from flask import Flask, render_template, request
import os
import sys

from MIT.exceptions import CustomException

from MIT.pipeline.training_pipeline import TrainingPipeline
from MIT.pipeline.prediction_pipeline import SinglePrediction

app = Flask(__name__)

IMG_PATH = os.path.join(os.getcwd(), 'static','image.jpg')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods = ['GET', 'POST'])
def train():

    training_pipeline = TrainingPipeline()

    training_pipeline.run_pipeline()

    return "Training Completed"

@app.route('/predict', methods = ['GET', 'POST'])
def predict():

    try:
        source_img = request.files['img']

        source_img.save(IMG_PATH)

        prediciton = SinglePrediction()

        result = prediciton.predict(IMG_PATH)

        return render_template('result.html', result=result)

    except Exception as e:
        raise CustomException(e, sys)
    
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)