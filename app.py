#!/usr/bin/env python
# # -*- coding: utf-8 -*-
""" Flask API for predicting probability of survival """
# importing the packages
import json
import pickle
import sys
# import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for, session, logging
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


try:
    model = joblib.load('random_forest.mdl')
    merk = joblib.load('merk.pkl')
    seri = joblib.load('seri.pkl')
    ram = joblib.load('RAM.pkl')
    memori = joblib.load('Memori.pkl')
    
except:
    print("Error loading application. Please run `python create_random_forest.py` first!")
    sys.exit(0)
    
app = Flask(__name__)

@app.route('/')
def main():
    """ Main page of the API """
    return render_template('home.html')
    
@app.route('/next')
def next():
    """ Main page of the API """
    return render_template('index.html')
    
@app.route('/flask')
def flask():
    """ Main page of the API """
    return render_template('flask1.html')

@app.route('/RF')
def RF():
    """ Main page of the API """
    return render_template('RF1.html')

@app.route('/iphone')
def iphone():
    """ Main page of the API """
    return render_template('iphone.html')
    
@app.route('/huawei')
def huawei():
    """ Main page of the API """
    return render_template('huawei.html')
@app.route('/advan')
def advan():
    """ Main page of the API """
    return render_template('advan.html')

@app.route('/asus')
def asus():
    """ Main page of the API """
    return render_template('asus.html')
@app.route('/nokia')
def nokia():
    """ Main page of the API """
    return render_template('nokia.html')
@app.route('/sony')
def sony():
    """ Main page of the API """
    return render_template('sony.html')
@app.route('/xiaomi')
def xiaomi():
    """ Main page of the API """
    return render_template('xiaomi.html')
@app.route('/oppo')
def oppo():
    """ Main page of the API """
    return render_template('oppo.html')
@app.route('/samsung')
def samsung():
    """ Main page of the API """
    return render_template('samsung.html')
@app.route('/realme')
def realme():
    """ Main page of the API """
    return render_template('realme.html')
@app.route('/vivo')
def vivo():
    """ Main page of the API """
    return render_template('vivo.html')
    
@app.route('/1GB')
def ramq():
    """ Main page of the API """
    return render_template('1GB.html') 
    
@app.route('/15GB')
def ramw():
    """ Main page of the API """
    return render_template('1.5GB.html')   
@app.route('/2GB')
def rame():
    """ Main page of the API """
    return render_template('2GB.html')     
@app.route('/3GB')
def ramr():
    """ Main page of the API """
    return render_template('3GB.html')    
@app.route('/4GB')
def ramt():
    """ Main page of the API """
    return render_template('4GB.html')    
@app.route('/6GB')
def ramy():
    """ Main page of the API """
    return render_template('6GB.html')     
@app.route('/8GB')
def ramu():
    """ Main page of the API """
    return render_template('8GB.html')    
@app.route('/10GB')
def rami():
    """ Main page of the API """
    return render_template('10GB.html')    
@app.route('/12GB')
def ramo():
    """ Main page of the API """
    return render_template('12GB.html')     
@app.route('/32MB')
def ramp():
    """ Main page of the API """
    return render_template('32MB.html')     
@app.route('/128MB')
def rama():
    """ Main page of the API """
    return render_template('128MB.html')   
@app.route('/512MB')
def rams():
    """ Main page of the API """
    return render_template('512MB.html') 
@app.route('/278MB')
def ramd():
    """ Main page of the API """
    return render_template('278MB.html') 
@app.route('/AN')
def AN():
    """ Main page of the API """
    return render_template('AN.html') 
 
@app.route('/predict', methods=['GET'])
def predict():
    args = request.args
    required_args = ['Merk','Seri','RAM','Memori']
    # Simple error handling for the arguments
    diff = set(required_args).difference(set(args.keys()))
    if len(diff) < 0:
        return "Error: wrong arguments. Missing arguments {}".format(str(diff))
    person_features = np.array([merk[args['Merk']],
                                seri[args['Seri']],
                                ram[args['RAM']],
                                memori[args['Memori']]
                               ]).reshape (1, -1)                                
    probability = model.predict(person_features)
    return render_template('index.html',
                            Merk=merk[args['Merk']],
                            Seri=seri[args['Seri']],
                            Memori=memori[args['Memori']],
                            RAM=ram[args['RAM']],
                            output=probability[0])
    
@app.route('/predict_api',methods=['GET'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)
if __name__ == "__main__":
    app.run(debug=True)