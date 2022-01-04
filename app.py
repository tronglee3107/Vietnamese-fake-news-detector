import streamlit as st
import pickle
import re

# get Vietnamese stop word
stop_word = []

#khoi tao tfidf_vectorizer
with open("vietnamese-stopwords.txt", encoding="utf8") as f:
    for i in f:
        i = i.replace('\n', '')
        stop_word.append(i)

#Tien xu ly string
def preprocess(a):

    a=a.lower()
    a=re.sub(r'[^\w\s]','', a)
    a=a.replace('\n',' ')
    for i in stop_word:
        temp=' '+i+' '
        if i in a:
            a=a.replace(temp,' ')
    return a

def main():
    st.set_page_config(
        page_title="Vietnamese fake news detector",
        page_icon=":shark:",
        layout="wide"
    )


    st.header(":newspaper: Fake news detection for Vietnamese")

    #load models
    model_1="models/PassiveAgressive.pkl"
    model_pac= pickle.load(open(model_1, 'rb'))

    model_2="models/LogicalRegressive.pkl"
    model_lrc= pickle.load(open(model_2, 'rb'))

    model_3="models/tfidf_vectorizer.pkl"
    tfidf_vectorizer=pickle.load(open(model_3,'rb'))

    # set the layout for app
    col1, col2 = st.columns([6, 4])

    model_list=['Passive Agressive Classifier','Logical Regression Classifier']
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
                    input_news=[preprocess(input_news)]
                    input_tfidf = tfidf_vectorizer.transform(input_news)
                    res = model_pac.predict(input_tfidf)
                else:
                    input_news = [preprocess(input_news)]
                    input_tfidf = tfidf_vectorizer.transform(input_news)
                    res = model_lrc.predict(input_tfidf)

                if res[0] == 1:
                    st.markdown("Fake news")
                else:
                    st.markdown("Real news!!!")

if __name__ == "__main__":
    main()





