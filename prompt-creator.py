import numpy as np
import pandas as pd
  
# fetch dataset 
df = pd.read_csv("diabetes_012.csv")
  
# data (as pandas dataframes) 
X = df.drop(['Diabetes_012'], axis=1) 
y = df['Diabetes_012'] 
  
# Convert each row into a single string
prompts = X.head(5000).apply(lambda row: (
    f"Given the following patient data, please answer with ONLY one of the following labels: diabetic, prediabetic, or not diabetic.\nPatient Data: "
    f"High Blood Pressure: {row['HighBP']},"
    f" High Cholesterol: {row['HighChol']},"
    f" Cholesterol Check: {row['CholCheck']},"
    f" Body Mass Index: {row['BMI']},"
    f" Is A Smoker: {row['Smoker']},"
    f" Had A Stroke: {row['Stroke']},"
    f" Had Heart Disease or Heart Attack: {row['HeartDiseaseorAttack']},"
    f" Physical Activity Level: {row['PhysActivity']},"
    f" Eats Fruits: {row['Fruits']},"
    f" Eats Vegetables: {row['Veggies']},"
    f" Heavy Alcohol Consumption: {row['HvyAlcoholConsump']},"
    f" Has Had Health Insurance: {row['AnyHealthcare']},"
    f" Did Not Go To Doctor Because of Cost: {row['NoDocbcCost']},"
    f" General Health Level: {row['GenHlth']},"
    f" Mental Health Level: {row['MentHlth']},"
    f" Has Difficulty Walking: {row['DiffWalk']},"
    f" Sex (0 = female, 1 = male): {row['Sex']},"
    f" Age Level (1-13): {row['Age']},"
    f" Education Level (1-6): {row['Education']},"
    f" Income Level (1-8): {row['Income']}, label: "
), axis=1)
prompts.to_csv('prompts.csv', index=False, header=False)



def createTarget(row):
    if row == 0.0:
        return 'not diabetic'
    if row == 1.0:
        return 'prediabetic'
    if row == 2.0:
        return 'diabetic'
    
    return 'undefined'

targets = y.head(200).apply(createTarget)

# targets.to_csv('targets.csv', index=False, header=False)
