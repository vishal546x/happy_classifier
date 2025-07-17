import pickle
import numpy as np
from PIL import Image
from flask import Flask,request,render_template
app=Flask(__name__)
with open('model.pkl','rb') as file:
    model=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)
@app.route("/")
def home():
    return render_template("home.html")
@app.route("/pred", methods=["POST"])
def pred():
    try:
        img=request.files["image_file"]
        img=Image.open(img)
        img=img.resize((64,64))
        img=img.convert("L")
        pixels=np.array(img)
        pix=scaler.transform(pixels.reshape(1,-1))
        ypred=model.predict(pix)
        ypred="Happy😁" if ypred[0]==1 else "Not Happy😒"
        return render_template("home.html",prediction=ypred)
    except Exception as e:
        return render_template("home.html",error=e)
if __name__=="__main__":
    app.run(debug=True)