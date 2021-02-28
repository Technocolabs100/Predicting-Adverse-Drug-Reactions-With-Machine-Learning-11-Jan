import numpy as np
import flask
from flask import Flask, request, jsonify, render_template
from flask import Flask,render_template,session,url_for,redirect
import pandas as pd
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem import rdMolDescriptors
import re, requests
import pickle
import warnings
from flask_wtf import FlaskForm 
from wtforms import TextField,SubmitField 
warnings.filterwarnings('ignore')

app = Flask(__name__)


model1 = pickle.load(open('.\Models\\Hepatobiliary disorders.pkl', 'rb'))
model2 = pickle.load(open('.\Models\\Metabolism and nutrition disorders.pkl', 'rb'))
model3 = pickle.load(open('.\Models\\Eye disorders.pkl', 'rb'))
model4 = pickle.load(open('.\Models\\Musculoskeletal and connective tissue disorders.pkl', 'rb'))
model5 = pickle.load(open('.\Models\\Gastrointestinal disorders.pkl', 'rb'))
model6 = pickle.load(open('.\Models\\Immune system disorders.pkl', 'rb'))
model7 = pickle.load(open('.\Models\\Reproductive system and breast disorders.pkl', 'rb'))
model8 = pickle.load(open('.\Models\\Neoplasms benign, malignant and unspecified (incl cysts and polyps).pkl', 'rb'))
model9 = pickle.load(open('.\Models\\General disorders and administration site conditions.pkl', 'rb'))
model10 = pickle.load(open('.\Models\\Endocrine disorders.pkl', 'rb'))
model11= pickle.load(open('.\Models\\Surgical and medical procedures.pkl', 'rb'))
model12= pickle.load(open('.\Models\\Vascular disorders.pkl', 'rb'))
model13 = pickle.load(open('.\Models\\Blood and lymphatic system disorders.pkl', 'rb'))
model14= pickle.load(open('.\Models\\Skin and subcutaneous tissue disorders.pkl', 'rb'))
model15 = pickle.load(open('.\Models\\Congenital, familial and genetic disorders.pkl', 'rb'))
model16 = pickle.load(open('.\Models\\Infections and infestations.pkl', 'rb'))
model17 = pickle.load(open('.\Models\\Respiratory, thoracic and mediastinal disorders.pkl', 'rb'))
model18 = pickle.load(open('.\Models\\Psychiatric disorders.pkl', 'rb'))
model19 = pickle.load(open('.\Models\\Renal and urinary disorders.pkl', 'rb'))
model20 = pickle.load(open('.\Models\\Pregnancy, puerperium and perinatal conditions.pkl', 'rb'))
model21 = pickle.load(open('.\Models\\Ear and labyrinth disorders.pkl', 'rb'))
model22 = pickle.load(open('.\Models\\Cardiac disorders.pkl', 'rb'))
model23 = pickle.load(open('.\Models\\Nervous system disorders.pkl', 'rb'))
model24 = pickle.load(open('.\Models\\Injury, poisoning and procedural complications.pkl', 'rb'))


def get_data(smile):
    mol = Chem.MolFromSmiles(smile)
    desc = get_descriptors(mol)
    return mol,desc


def get_descriptors(mol, write=False):
    # Make a copy of the molecule dataframe
    desc = [Lipinski.NumAromaticHeterocycles(mol),
            Lipinski.NumAromaticRings(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.RingCount(mol),
            Lipinski.NHOHCount(mol),
            Lipinski.NumHeteroatoms(mol),
            Lipinski.NumAliphaticCarbocycles(mol),
            Lipinski.NumSaturatedCarbocycles(mol),
            Lipinski.NumAliphaticHeterocycles(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumSaturatedHeterocycles(mol),
            Lipinski.NumAliphaticRings(mol),
            Descriptors.NumRadicalElectrons(mol),
            Descriptors.MaxPartialCharge(mol),
            Descriptors.NumValenceElectrons(mol),
            Lipinski.FractionCSP3(mol),
            Descriptors.MaxAbsPartialCharge(mol),
            Lipinski.NumAromaticCarbocycles(mol),
            Lipinski.NumSaturatedRings(mol),
            Lipinski.NumRotatableBonds(mol)
           ]
        
    desc = [0 if i!=i else i for i in desc]
    return desc


app.config['SECRET_KEY'] = 'mysecretkey'

class DrugForm(FlaskForm):
  drug = TextField("Smiles")
  submit = SubmitField("Predict")

@app.route("/",methods = ['GET','POST'])

def index():

  form =  DrugForm()

  if form.validate_on_submit():

    session['drug'] = form.drug.data

    return redirect(url_for("prediction"))
  return render_template('home.html',form=form) 


@app.route('/prediction')

def prediction():

    mol = session['drug']
    mol,desc = get_data(mol)
    try:
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1125)
    except Exception as e:
        fp = np.nan
        
    test = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, test)
    test = pd.Series(test)
    test = test.append(pd.Series(desc))
    test = [0 if i!=i else i for i in test]
    
    result = pd.Series(index = ['Hepatobiliary disorders', 'Metabolism and nutrition disorders',
       'Eye disorders', 'Musculoskeletal and connective tissue disorders',
       'Gastrointestinal disorders', 'Immune system disorders',
       'Reproductive system and breast disorders',
       'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
       'General disorders and administration site conditions',
       'Endocrine disorders', 'Surgical and medical procedures',
       'Vascular disorders', 'Blood and lymphatic system disorders',
       'Skin and subcutaneous tissue disorders',
       'Congenital, familial and genetic disorders',
       'Infections and infestations',
       'Respiratory, thoracic and mediastinal disorders',
       'Psychiatric disorders', 'Renal and urinary disorders',
       'Pregnancy, puerperium and perinatal conditions',
       'Ear and labyrinth disorders', 'Cardiac disorders',
       'Nervous system disorders',
       'Injury, poisoning and procedural complications'])
    
    model = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11,
             model12, model13, model14, model15, model16, model17, model18, model19, model20, model21,
             model22, model23, model24]

    for i,model in enumerate(model):
        result[i] = (model.predict([test])[0])
    
    
    return render_template('prediction.html', results = 'The Drug ADR are :\n {}'.format([result.index[i] for i,v in enumerate(result) if v==1]))

if __name__ == "__main__":
    app.run(debug = True)
         