import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler

from scipy.stats import chi2_contingency

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import xgboost as xgb

raw_df = pd.read_csv("cardio_train.csv", sep=";")
print(raw_df.duplicated().sum())
raw_df.drop_duplicates(inplace=True)

def feature_outlier_removal(df, feature, min_q, max_q):
    feature_min_outlier_mask = df[feature] > df[feature].quantile(min_q)
    feature_max_outlier_mask = df[feature] < df[feature].quantile(max_q)
    df = df[(feature_min_outlier_mask) & (feature_max_outlier_mask)]
    print(feature, "min: ", df[feature].quantile(min_q))
    print(feature, "max: ", df[feature].quantile(max_q))
    return df

raw_df = feature_outlier_removal(raw_df, "height", 0.005, 0.999)
raw_df = feature_outlier_removal(raw_df, "weight", 0.001, 0.999)
raw_df['age'] = round(raw_df['age']/365.25).apply(lambda x: int(x))
raw_df['gender']= raw_df['gender'].apply(lambda x: 0 if x==2 else 1)
raw_df = raw_df[raw_df['ap_hi'] > raw_df['ap_lo']].reset_index(drop=True)
def find_bmi(data):
    bmi = data['weight']/((data['height']/100)**2)
    return bmi

raw_df['bmi'] = raw_df.apply(find_bmi, axis=1)

def bp_level(data):
    if (data['ap_hi'] <= 120) and (data['ap_lo'] <= 80):
        return 'normal'
    if (data['ap_hi'] >= 120 and data['ap_hi'] < 129) and (data['ap_lo'] < 80):
        return 'above_normal'
    if (data['ap_hi'] >= 129 and data['ap_hi'] < 139) | (data['ap_lo'] >= 80 and data['ap_lo'] < 89):
        return 'high'
    if (data['ap_hi'] >= 139) | (data['ap_lo'] >= 89):
        return 'very_high'
    if (data['ap_hi'] >= 180) | (data['ap_lo'] >= 120):
        return 'extreme_high'

    
raw_df['bp_level'] = raw_df.apply(bp_level, axis=1)
def age_level(data):
    if data["age"] < 40:
        return '1'
    if data['age'] >= 40 and data['age'] < 45:
        return '2'
    if data['age'] >= 45 and data['age'] < 50:
        return '3'
    if data['age'] >= 50 and data['age'] < 55:
        return '4'
    if data['age'] >= 55 and data['age'] < 60:
        return '5'
    if data['age'] >= 60:
        return '6'

    
raw_df['age_level'] = raw_df.apply(age_level, axis=1)
def bmi_level(data):
    if data['bmi'] <= 18.5:
        return 'underweight'
    if data['bmi'] > 18.5 and data['bmi'] <= 24.9:
        return 'normal'
    if data['bmi'] > 24.9 and data['bmi'] <= 29.9:
        return 'overweight'
    if data['bmi'] >= 29.9:
        return 'obese'
    
raw_df['bmi_level'] = raw_df.apply(bmi_level, axis=1)
def hypothesis_testing(feature, target):
    g = pd.crosstab(feature, target, margins=True)
    chi2_result = chi2_contingency(g)
    return chi2_result[1]

df_cols = raw_df.columns
drop_list = []

for i in tqdm(range(len(df_cols)), position=0, leave=True):
    p = hypothesis_testing(raw_df[df_cols[i]], raw_df['cardio'])
    
    if p >= 0.05:
        drop_list.append(df_cols[i])
        
print(drop_list)
raw_df.drop(drop_list, axis=1, inplace=True)
joblib.dump(drop_list, './pickles/drop_columns.pkl')


f, ax = plt.subplots(figsize=(12, 10))

clean_data = raw_df.copy()
clean_data = pd.get_dummies(clean_data,drop_first=False)
clean_data.to_csv('clean_df.csv', index = False)

clean_df = pd.read_csv("clean_df.csv")

df_X = clean_df.drop(["cardio"], axis=1)
df_y = clean_df.loc[:, "cardio"]
df_X = df_X.reindex(sorted(df_X.columns), axis=1)

print(df_X.info())
joblib.dump(df_X.columns, './pickles/data_columns.pkl')
scaler = MinMaxScaler()
df_X = scaler.fit_transform(df_X)
print(df_X)
joblib.dump(scaler, './pickles/std_scaler.pkl') 
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)
def model_evaluation(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm, end="\n\n")

    TN = cm[0, 0]
    FN = cm[0, 1]
    FP = cm[1, 0]
    TP = cm[1, 1]

    P = FN+TP
    N = TN+FP

    TPR = TP/P
    TNR = TN/N
    FPR = FP/N
    FNR = FN/P

    accuracy = (TN+TP)/(P+N)
    print("Test Accuracy: "+str(accuracy), end="\n\n")
    print("All 4 parameters: ",TN, FN, FP, TP, end="\n\n")
    print("TPR: {}".format(TPR))
    print("TNR: {}".format(TNR))
    print("FPR: {}".format(FPR))
    print("FNR: {}".format(FNR))
    print()
    
    print(classification_report(y_test, y_pred))
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
print("Training Accuracy: "+str(xgb_model.score(X_train, y_train)), end="\n\n")
xgb_pred = xgb_model.predict(X_test)
model_evaluation(y_test, xgb_pred)
joblib.dump(xgb_model, './pickles/classifier_model.pkl') 
