from flask import Flask, render_template, request
from tensorflow import keras
model = keras.models.load_model("\\dog breeds\\my_modedl\\")
import cv2 as cv
import numpy as np
import pandas as pd

df=pd.read_csv("code_conversion.csv")
breed=df["breed"]
breed_code=df["breed_code"]
app = Flask(__name__,template_folder="template")

@app.route('/')
def index():
	return render_template("index.html", data="hey")


@app.route("/prediction", methods=["POST"])
def prediction():
    img = request.files['img']

	
    img.save("img.jpg")

	
    image = cv.imread("img.jpg")

	
    image = cv.resize(image, (224,224))

    image=np.array(image/255)

	
    pred = model.predict(image[np.newaxis, ...])

	
    pred = np.argmax(pred)

	
    
    
    for i in range(0,120):
            if breed_code[i]==pred:
                # index = breed_code.index(x)
                pred=breed[i]

	
    return render_template("prediction.html", data=pred)

@app.route('/contact')
def contact():
    render_template("contact.html")
if __name__ == "__main__":
	app.run(debug=True)