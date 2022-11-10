import pandas as pd 
import numpy as np 
import pickle
import json
import config

class IrisPrediction():

    def __init__(self,Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm):
        self.Id=Id
        self.SepalLengthCm=SepalLengthCm
        self.SepalWidthCm=SepalWidthCm
        self.PetalLengthCm=PetalLengthCm
        self.PetalWidthCm=PetalWidthCm

    def Load_model(self):
        with open("models\logistic_model_iris.pkl","rb")as f:
            self.logistic_model=pickle.load(f)

        with open("models\project_data_iris.json","r")as f:
            self.project_data=json.load(f)

    def Get_prediction(self):
        self.Load_model()

        array=np.zeros(len(self.project_data["columns"]))
        array[0]=self.Id
        array[1]=self.SepalLengthCm
        array[2]=self.SepalWidthCm
        array[3]=self.PetalLengthCm
        array[4]=self.PetalWidthCm

        prediction=self.logistic_model.predict([array])[0]
        print("flower type is",prediction)
        return prediction

if __name__=="__main__":
    Id=1.0
    SepalLengthCm=5.1
    SepalWidthCm=3.5
    PetalLengthCm=1.4
    PetalWidthCm=0.2

    pred=IrisPrediction(Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm)
    flower_prediction=pred.Get_prediction()
    print("type of flower is",flower_prediction)


        
        
