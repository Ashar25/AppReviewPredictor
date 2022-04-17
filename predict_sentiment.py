import numpy
import numpy as np
import pickle

import pandas as pd
import streamlit as st
import os

direc = os.getcwd()
path = os.path.join(direc)
loaded_model = pickle.load(open('/Users/fatima.m_rr/PyProjects/app-rating-predictor/trained_model.sav', 'rb'))


def prediction(review):
    predic = loaded_model.predict(review)
    return predic

def main():
    st.title('Sentiment Prediction Web App')
    data_file = st.file_uploader("Upload CSV", type=["csv"])
    if data_file is not None:
        df = pd.read_csv(data_file)
        df = df.drop(np.argmax(df["Text"].isna()))
        st.dataframe(df)
        star = df['Star'].values
        review = list(df['Text'])
        predic_csv = prediction(review)
        answer = np.logical_or(np.logical_and(predic_csv > 3, star >= 3), np.logical_and(predic_csv <= 3, star < 3))
        d = pd.DataFrame(
            {'answer': answer,
             'review': review,
             'star': star
             })
        st.title('Incorrect review-ratings')
        st.dataframe(d[d['answer'] == 0])

if __name__ == '__main__':
    main()
