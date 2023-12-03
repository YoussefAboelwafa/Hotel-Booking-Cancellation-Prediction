import numpy as np
from flask import Flask, render_template, request
import pickle

# Create the application object
app = Flask(__name__)

# Load the model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading the model: {e}")


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    features = [int(x) for x in request.form.values()]
    features_array = [np.array(features)]
    pred = model.predict(features_array)
    prediction = pred[0]

    if prediction == 0:
        prediction = "Not Canceled"
    else:
        prediction = "Canceled"
    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
