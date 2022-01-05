import streamlit as st
import pickle
import re
import requests
import warnings
warnings.filterwarnings('ignore')
from underthesea import word_tokenize
import pandas as pd

# #load data from file and get stopsword
# df=pd.read_csv("resources/vn_news_223_tdlfr.csv")
with open("resources/vietnamese-stopwords.txt", encoding ="utf8", errors ='replace') as f_st:
    stop_word=f_st.read().split("\n")



#Preprocessing
def preprocess_line(a):
    a=a.lower()
    a=re.sub(r'[^\w\s]','', a)
    a=a.replace('\n',' ')
    for i in stop_word:
        temp=' '+i+' '
        if temp in a:
            a=a.replace(temp,' ')
    a=word_tokenize(a)
    return a

# def preprocess(t):
#     for i in range(len(t)):
#         t[i]=preprocess_line(t[i])
#     return t

# preprocess(df.text)
# fea = []
# for i in df.text:
#     for j in i:
#         if j not in fea:
#             fea.append(j)
fea=pickle.load(open('models/fea.pkl','rb'))
def numerics(list_text,fea):
    matrix = []
    for i in list_text:
        tmp = []
        for j in fea:
            if j in i:
                tmp.append(1)
            else:
                tmp.append(0)
        matrix.append(tmp)
    df1 = pd.DataFrame(matrix, columns= fea)
    return df1


@st.cache(allow_output_mutation=True)
def load_session():
    return requests.Session()

def main():
    st.set_page_config(
        page_title="Vietnamese fake news detector",
        page_icon=":shark:",
        layout="wide"
    )


    st.header(":newspaper: Fake news detection for Vietnamese")
    sess = load_session()
    #load models
    model_1="models/PassiveAgressive.pkl"
    model_pac= pickle.load(open(model_1, 'rb'))

    model_2="models/LogicalRegressive.pkl"
    model_lrc= pickle.load(open(model_2, 'rb'))



    # set the layout for app
    col1, col2 = st.columns([6, 4])

    model_list=['Passive Agressive Classifier','Logistic Regression Classifier']
    with col1:
        model_name=st.selectbox("Choose a model", index=0,options=model_list)
        input_news=st.text_area("Insert news that you want to check here")
        entered_items=st.empty()

    pred_button=st.button("Predict Real or Fake news")
    if pred_button:
        with st.spinner("Predicting..."):
            if not len(input_news):
                st.markdown("Input at least a piece of news")
            else:
                res=-1
                if model_name==model_list[0]:
                    # res = model_pac.predict(numerics([preprocess_line(input_news)], fea))

                    vectorized=numerics([preprocess_line(input_news)], fea)
                    print(vectorized)
                    res = model_pac.predict(vectorized)
                else:
                    vectorized = numerics([preprocess_line(input_news)], fea)
                    res = model_lrc.predict(vectorized)

                if res[0] == 1:
                    st.markdown("Fake news")
                else:
                    st.markdown("Real news!!!")

if __name__ == "__main__":
    main()





