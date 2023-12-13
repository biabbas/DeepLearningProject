import pandas as pd
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

filename='Training.csv'
data=read_csv(filename)

df_x=data[['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)','depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze'
]]
df_y=data[['prognosis']]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb = gnb.fit(df_x, np.ravel(df_y))
from sklearn.metrics import accuracy_score
y_pred=gnb.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred,normalize=False))

prediction=gnb.predict(x_test)
print(prediction[0:10])
import joblib as joblib
joblib.dump(gnb,'healthcare/model/naive_bayes.pkl')
nb=joblib.load('healthcare/model/naive_bayes.pkl')

prediction=nb.predict(x_test)
print(prediction[0:10])
list_a=['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)','depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']
print(len(list_a))

list_c=[]
for x in range(0,len(list_a)):
    list_c.append(0)

list_b=['shivering','headache','vomiting','muscle_pain','diarrhoea']

for z in range(0,len(list_a)):
    for k in list_b:
        if(k==list_a[z]):
            list_c[z]=1
print(list_c)

test=np.array(list_c)
test=np.array(test).reshape(1,-1)
print(test.shape)

prediction=nb.predict(test)
print(prediction)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf=rf.fit(df_x,np.ravel(df_y))

y_pred=rf.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred,normalize=False))

rf.score(x_test,y_test)

prediction=rf.predict(x_test)
print(prediction[0:10])
joblib.dump(rf,'healthcare/model/random_forest.pkl')
rand_forest=joblib.load('healthcare/model/random_forest.pkl')

prediction=rand_forest.predict(x_test)
print(prediction[0:10])

prediction=rand_forest.predict(test)
print(prediction[0])

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(df_x, df_y)

logreg.score(x_test, y_test)

import joblib as joblib
joblib.dump(logreg, 'healthcare/model/lopistic_regression.pkl')
dt = joblib.load('healthcare/model/lopistic_regression.pkl')

prediction = dt.predict(test)
print(prediction[0])

from sklearn import tree

clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
clf3 = clf3.fit(df_x,df_y)

from sklearn.metrics import accuracy_score
y_pred=clf3.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred,normalize=False))

import joblib as joblib
joblib.dump(clf3, 'healthcare/model/decision_tree.pkl')


dt = joblib.load('healthcare/model/decision_tree.pkl')

prediction = dt.predict(test)
print(prediction[0])

print("Completed")