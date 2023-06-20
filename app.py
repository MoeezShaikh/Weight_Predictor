import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [eval(x) for x in request.form.values()] #It uses a list comprehension to iterate over the values and evaluates each one using the eval function. This is necessary to convert the string representations of the form values into their respective Python types.
    final_features = [np.array(int_features)] #converted into a NumPy array which done to format the data in a way that can be used as input for the model prediction
    prediction = model.predict(final_features)
    

    output = np.round(prediction[0], 2)

    return render_template('index.html', prediction_text="Weight prediction : {} kg's".format(output))

if __name__ == "__main__":
    app.run(debug=True)