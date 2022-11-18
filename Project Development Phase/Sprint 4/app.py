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

        payload_scoring = {"input_data": 
			[{"field": [["f0", "f1","f2","f3","f4","f5"]], 
                "values": [init_features]}]}
        pred = model.predict(init_features)


        output = pred[0]['values'][0][0]
        return render_template("result.html",prediction='The WQI predicted is {:.2f}'.format(output))

    else:
        return render_template("predict.html")


if __name__=="__main__":
    app.run(debug=False,port=5500)
            
            