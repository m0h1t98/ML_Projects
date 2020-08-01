import pandas as pd
import numpy as np
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import joblib


df = pd.read_csv('train.csv')

df.drop(['Id','Behaviour','DistanceFromHome','EmployeeNumber','Gender','JobInvolvement','PercentSalaryHike',
         'CommunicationSkill','TrainingTimesLastYear','PerformanceRating','YearsSinceLastPromotion',
         'Department','StockOptionLevel','EducationField','YearsWithCurrManager','TotalWorkingYears',
         'YearsInCurrentRole','EnvironmentSatisfaction','Education'],axis=1,inplace=True)

df.drop_duplicates(inplace=True)

X = df.drop('Attrition', inplace=False, axis=1)
y = df.Attrition

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

ct = make_column_transformer((ohe, make_column_selector(dtype_include='object')),
                            remainder='passthrough')

xg = xgb.XGBClassifier(subsample=0.6,colsample_bytree=1.0,min_child_weight=7,max_depth=5,
                       learning_rate=0.03571428571428572,gamma=0.07755102040816328,
                       scale_pos_weight=5,random_state=42)

rnd = RandomForestClassifier(n_estimators=500,min_samples_split=80,min_samples_leaf=2,
                             max_features='log2',max_depth=8,random_state=42)

lgbm = LGBMClassifier(random_state=42)

vc = VotingClassifier(estimators=[('rnd', rnd), ('xg', xg), ('lgbm', lgbm)],voting='soft', n_jobs=-1)

pipe = make_pipeline(ct,vc)

pipe.fit(X, y)

filename = 'model.pkl'
joblib.dump(pipe, filename)
