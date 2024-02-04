from flask import Flask,render_template,request,redirect,url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
app =Flask(__name__)
#loading of the pre trained model
model = load_model('tollywood_model.h5')
#preprocessing of images
def preprocess_image(img_path):
    img =image.load_img(img_path,target_size =(224,224))
    img_array =image.img_to_array(img)
    img_array =np.expand_dims(img_array,axis=0)
    img_array/=255.0
    return img_array
#prediction function
def predict_class(img_path):
    img_array =preprocess_image(img_path)
    predictions =model.predict(img_array)
    predicted_class =np.argmax(predictions,axis=1)
    return predicted_class[0]
@app.route('/',methods=['GET','POST'])
def main():
    if request.method=='POST':
        file = request.files['file']
        file_path ='C:\Code\image_dataram\celeb.jpg'
        file.save(file_path)
        #predicting of image
        predicted_class =predict_class(file_path)
        #redirect to the result page with the predicted class and image path
        return redirect(url_for('result',predicted_class=predicted_class,image_path =file_path))
    return render_template('main.html')
class_names = ['AlluArjun','MaheshBabu','NagaChaithanya','Nani','NTR','Prabhas','RamCharan','Vijaydevarakonda']

#creaing a dictionary mapping class indices to class names
class_index_to_name ={i: name for i,name in enumerate(class_names)}
@app.route('/result/<int:predicted_class>/<path:image_path>')
def result(predicted_class,image_path):
    #get the class name corresponding to the predicted class
    predicted_class_name =class_index_to_name.get(predicted_class,'Unknown')
    return render_template('result.html',predicted_class =predicted_class,predicted_class_name =predicted_class_name,image_path = image_path)
if __name__ =='__main__':
    app.run(debug=True)
