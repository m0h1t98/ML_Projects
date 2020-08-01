from flask import Flask, render_template, url_for, request
import numpy as np
import pandas as pd
import joblib

filename = 'model.pkl'
pipe = joblib.load(filename)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    arr = []
    
    if request.method == 'POST':
        age = int(request.form['Age'])
        marital_status = request.form['Marital_Status']
        jobrole = request.form['Job_Role']
        monthly_salary = int(request.form['Monthly_Salary'])//75.20
        num_companies_worked = int(request.form['NumCompaniesWorked'])
        years_at_company = int(request.form['YearsAtCompany'])
        job_satisfaction = int(request.form['Job_Satisfaction'])
        business_travel = request.form['Business_Travel']
        overtime = request.form['Overtime']
        
        arr = [[age, business_travel, jobrole, job_satisfaction, marital_status, 
                monthly_salary, num_companies_worked, overtime, years_at_company]]
        
        X_test = pd.DataFrame(arr,columns=['Age','BusinessTravel','JobRole','JobSatisfaction','MaritalStatus',
                                           'MonthlyIncome','NumCompaniesWorked','OverTime','YearsAtCompany'])
        pred = pipe.predict(X_test)
        
        if pred == 1:
            pred_text = "It's time to look for new opportunities. According to past data, your job offer is likely to be revoked."
        else:
            pred_text = "Your Job is secure. Good luck for future endeavours!"
    
    return render_template('results.html',pred_text=pred_text)

if __name__ == '__main__':
    app.run()
