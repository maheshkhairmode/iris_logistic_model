from distutils.command.config import config
from flask import Flask,jsonify,render_template,request
from models.utils import IrisPrediction
import config

app=Flask(__name__)

@app.route("/")
def hello_flask():
    return render_template("index.html")

@app.route("/predict_flower",methods=["POST","GET"])
def pred_flower():
    if request.method =="POST":

        Id=float(request.form.get("Id"))
        SepalLengthCm=float(request.form.get("SepalLengthCm"))
        SepalWidthCm=float(request.form.get("SepalWidthCm"))
        PetalLengthCm=float(request.form.get("PetalLengthCm"))
        PetalWidthCm=float(request.form.get("PetalWidthCm"))
        
        
        
        pred=IrisPrediction(Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm)
        flower_prediction=pred.Get_prediction()


        print("type of flower is",flower_prediction)
        return render_template("index.html",prediction=flower_prediction)
        


if __name__=="__main__":
    app.run(host="0.0.0.0",port=config.PORT_NUMBER,debug=True)
