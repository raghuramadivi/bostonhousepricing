import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__) ##This is the basic flask app.Here (__name__) is the starting point of my application from where it will run 
##Load the Model
regmodel=pickle.load(open('../bostonhousepricing/regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/') ##app.route is the 1st route u can jst say tht like the localhost and URL and Slash basically say I shd definetly go to my home page
def home():#Here im creating home page as my definition and this will probably return html pqage 
    return render_template('home.html')#home.html have not yet defined jst consider that this will be my html page where i can probably say welcome everyone 
                           #So by default once i hit this flask app it is jst gng to redirect to the home.html and tht is how we go w.r.t flask

@app.route('/predict_api',methods=['POST']) #we r gng to make sure tht we create a predict api so for creating predict api I'm jst gng to create an api
def predict_api():                #where i can using post man or any other tool u know we can send a request to our app and then we get the output 
   data=request.json['data']      #This is POST request becoz frm my side im gng to give some input and tht will capture the input then our model will give the output.               
   print(data)                  #Whenever i hit this predict_api() the input tht im gng to give i'm gng to make sure tht i give it in the json format which will capatured inside the 'data' key.
   print(np.array(list(data.values())).reshape(1,-1))           #Frm here as soon as i hit this api('/predict') as a POST request with this information('data') watever info inside this 'data' 
   new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))   #We are gng to capture it using request.json and then this will get stored in data variable.  
   output=regmodel.predict(new_data)   #This is in two dimensional array
   print(output[0])      #Im gng to take first value                                 
   return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()] #This is list format and watever values we r filling in tht form we'll be able to capture it becoz all the info present in request object 
    final_input=scalar.transform(np.array(data).reshape(1,-1))  #and i want convert all these values into float becoz all these values needs to be given as float w.r.t the model
    print(final_input)                                               # im creatng for loop for every values inside this form convert tht into float and finally get in the form of list format 
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House Price Prediction is {}".format(output)) #Here im gng to render 'home.html' and it is gng to replace this placeholder(prediction_text).there will some kind of placeholder in the html page and we r gng to house price predictionis some output  

if __name__=="__main__":
    app.run(debug=True)
