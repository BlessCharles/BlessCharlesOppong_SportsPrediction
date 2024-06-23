import streamlit as st 
from streamlit import runtime
import pandas as pd 
import numpy as np 
import pickle 
from sklearn.preprocessing import StandardScaler

st.title('Sports Prediction Assignment')

file=r"C:\\Users\\user\\Downloads\\C__Users_user_Downloads"
files=pickle.load(open(file,'rb'))

potential = st.number_input('potential')
age = st.number_input('age')
international_reputation = st.number_input('international_reputation')
passing = st.number_input('passing')
dribbling = st.number_input('dribbling')
physic = st.number_input('physic')
attacking_short_passing = st.number_input('attacking_short_passing')
skill_curve = st.number_input('skill_curve')
skill_long_passing = st.number_input('skill_long_passing')
skill_ball_control = st.number_input('skill_ball_control')
movement_reactions = st.number_input('movement_reactions')
power_shot_power = st.number_input('power_shot_power')
power_long_shots = st.number_input('power_long_shots')
mentality_vision = st.number_input('mentality_vision')
mentality_composure = st.number_input('mentality_composure')

data = np.array([[potential, age, international_reputation, passing, dribbling, physic,
                        attacking_short_passing, skill_curve, skill_long_passing,
                        skill_ball_control, movement_reactions, power_shot_power,
                        power_long_shots, mentality_vision, mentality_composure]])
df=pd.DataFrame(data)
#scale the data
sc = StandardScaler()
scaled=sc.fit_transform(df)


if st.button('Predict'):
    predictions = files.predict(scaled)
    st.write("The overall rate for you player is: ", predictions[0])



