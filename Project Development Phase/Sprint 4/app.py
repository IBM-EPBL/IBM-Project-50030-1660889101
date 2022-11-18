import os
import numpy as np 
from flask import Flask,request,render_template 
import pickle
import requests

app= Flask(__name__)
model_path = os.path.join(os.path.dirname(__file__), "model/ibm.pkl")
model = pickle.load(open(model_path, 'rb')) # loading the trained model

@app.route("/") 
def about():
    return render_template("about.html")

@app.route("/about")
def home():
    return render_template("about.html")

@app.route("/info") 
def information():
    return render_template("info.html")


@app.route("/predict",methods=["GET","POST"]) 
def upload():
    if request.method=='POST':
        init_features = [float(x) for x in request.form.values()]

        pred = model.predict([init_features])

        print(pred)

        output = pred[0]
        return render_template("result.html",prediction='The WQI predicted is {:.2f}'.format(output))

    else:
        return render_template("predict.html")


if __name__=="__main__":
    app.run(debug=True,port=5500)
            
            