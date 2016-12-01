import pandas as pd
import numpy as np
from sklearn import preprocessing
from imblearn.combine import SMOTETomek


df = pd.read_csv('../data/diabetes.csv')

tmp_blood_pressure = df.BloodPressure
tmp_blood_pressure = tmp_blood_pressure[tmp_blood_pressure > 0]
tmp_blood_pressure_mean = np.mean(tmp_blood_pressure)
tmp_blood_pressure_std = np.std(tmp_blood_pressure)
l = len(df.loc[df.BloodPressure <= 0])

df.loc[df.BloodPressure <= 0, 'BloodPressure'] = np.random.randint(tmp_blood_pressure_mean - tmp_blood_pressure_std,
                                                                   tmp_blood_pressure_mean + tmp_blood_pressure_std,
                                                                   l)
tmp_skin_thickness = df.SkinThickness
tmp_skin_thickness = tmp_skin_thickness[tmp_skin_thickness > 0]
skin_thickness_mean = np.mean(tmp_skin_thickness)
skin_thickness_std = np.std(tmp_skin_thickness)
l = len(df.loc[df.SkinThickness <= 0])

df.loc[df.SkinThickness <= 0, 'SkinThickness'] = np.random.randint(skin_thickness_mean - skin_thickness_std,
                                                                   skin_thickness_mean + skin_thickness_std,
                                                                   l)

tmp_insulin = df.Insulin
tmp_insulin = tmp_insulin[tmp_insulin > 0]
insulin_mean = np.mean(tmp_insulin)
insulin_std = np.std(tmp_insulin)
l = len(df.loc[df.Insulin <= 0])

df.loc[df.Insulin <= 0, 'Insulin'] = np.random.randint(insulin_mean - insulin_std,
                                                       insulin_mean + insulin_std,
                                                       l)

tmp_bmi = df.BMI
tmp_bmi = tmp_bmi[tmp_bmi > 0]
bmi_mean = np.mean(tmp_bmi)
bmi_std = np.std(tmp_bmi)
l = len(df.loc[df.BMI <= 0])

df.loc[df.BMI <= 0, 'BMI'] = bmi_mean - bmi_std + 2 * bmi_std * np.random.rand(l)

tmp_glu = df.Glucose
tmp_glu = tmp_glu[tmp_glu > 0]
tmp_glu_mean = np.mean(tmp_glu)
tmp_glu_std = np.std(tmp_glu)
l = len(df.loc[df.Glucose <= 0])

df.loc[df.Glucose <= 0, 'Glucose'] = np.random.randint(tmp_glu_mean - tmp_glu_std, tmp_glu_mean + tmp_glu_std, l)

label = np.array(df['Outcome'])

df = preprocessing.scale(df)
df[:, -1] = label

X = df[:, 0:-1]
Y = df[:, -1]

print('current proportion of y = 1: ', np.sum(Y)/len(Y))

sm = SMOTETomek(ratio=0.95, k=7)
X_resampled, Y_resampled = sm.fit_sample(X, Y)

print('after balancing, sample size = %d, feature size = %d' % X_resampled.shape)
print('after balancing, proportion of y = 1: ',np.sum(Y_resampled)/len(Y_resampled))

X_resampled = preprocessing.scale(X_resampled)
df = np.hstack((X_resampled, Y_resampled.reshape((len(Y_resampled), 1))))
pd.DataFrame(df).to_csv('../data/diabetes_normalized_balanced.csv')
pd.DataFrame(df).to_pickle('../data/diabetes_normalized_balanced.pickle')

