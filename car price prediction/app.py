from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pandas as pd


# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image



# Flask utils
from flask import Flask, redirect, url_for, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from keras.utils import load_img, img_to_array
#from gevent.pywsgi import WSGIServer
import joblib
import statsmodels.api as sm


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='C:/Users/HP/Desktop/carprice and brand detection/car_brand_resnet50.h5'

# Load your trained model
model = load_model(MODEL_PATH)
lm=joblib.load('car_model.joblib')
scaler=joblib.load('pricescaler.joblib')


# Ensure the uploads directory exists
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def model_predict(img_path, model):
    img = load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
    x= preprocess_input(x)

    preds= model.predict (x)
    preds=np.argmax(preds, axis=1)

    
    if preds == 0:
        return "The Car IS Audi"
    elif preds == 1:
        return "The Car is Lamborghini"
    else:
        return "The Car Is Mercedes"

    
    
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    
   
    if request.method == 'POST':
            
        f = request.files['file']

        # Save the file to ./uploads
       
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(file_path)


        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        image_filename = f.filename 
    
        return render_template('result brand.html', result=result, image=image_filename)
    # For GET request, render the upload form
    return render_template('result brand.html', result="", image=None)
    

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename) 


@app.route('/price', methods=['GET','POST'])
def price():

    rounded_price = None 

    if request.method == 'POST':     
        # Get the input values
        horsepower = float(request.form['horsepower'])
        carwidth = float(request.form['carwidth'])
        carbody=request.form['carbody'] 
        cylinder = request.form['cylinder']
        car_range = request.form['car_range']


        # Encode values for prediction
        hatchback = 1 if carbody == 'hatchback' else 0
        four = 1 if cylinder == 'four' else 0
        Highend = 1 if car_range == 'highend' else 0

        # Prepare new data for prediction
        new_data = {
            'horsepower': [horsepower],  
            'carwidth': [carwidth],     
            'hatchback': [hatchback],     
            'four': [four],  # Only using the 'four' encoding
            'Highend': [Highend]  
        }

        # Step 2: Create a DataFrame
        new_df = pd.DataFrame(new_data)
        print(new_df)

        # Step 3: Scale the new data using the same scaler
        new_df_scaled = scaler.fit_transform(new_df)

        # Step 3: Convert scaled data back to DataFrame
        new_df_scaled = pd.DataFrame(new_df_scaled,columns=new_df.columns)

        # Step 5: Predict the price using your trained model
        new_df_scaled.insert(0, 'const', 1) 
       
        predicted_price = lm.predict(new_df_scaled)
        # Round the predicted price to 2 decimal places
        rounded_price = round(predicted_price[0], 2)

        # Output the predicted price
    return render_template('price.html', predicted_price=rounded_price)
    
if __name__ == '__main__':
    app.run(debug=True)
