__author__ = 'Prudhvi Badri'
from datetime import datetime
start_time = datetime.now()
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re

def Review_Score(reviews_data):
    return reviews_data['RATINGS'].values

def remove_stopWords(review):
    word_tokens = []
    stopWords = []                                       # created empty stopwords list
    filtered_words = []                                  # created empty list to store the reviews filtered from stopwords
    stop_words_file =  open("stopWords.txt","r")         # reading stopwords file

    for each_stop_word in stop_words_file:
        stopWords.append(str(each_stop_word.strip("\n").decode('utf-8',errors='ignore')))     # adding stopwords to stopwords list

    clean_review = str(review).decode('utf-8',errors='ignore')
    processed_review = re.findall(r"\b[a-zA-Z']+\b", clean_review)
    for each_word in processed_review:
        word_tokens.append(str(each_word))
    upper_Case_stopwords =  map(str.upper,stopWords)

    for word in word_tokens:
        if (word.lower() not in stopWords and len(word)>1 and word not in upper_Case_stopwords): #checks condition if word length > 1 and not present in stopwords list
            filtered_words.append(word)
    return filtered_words

def LSA_using_TF_IDF(reviews_data):
    review_docs = []
    text_reviews = reviews_data['REVIEW_TEXT']
    for each_review in text_reviews:
        cleaned_review = re.findall(r"\b[a-z']+\b", str(each_review).lower())
        processed_review = ' '.join(word for word in cleaned_review if len(word)>1)
        review_docs.append(str(processed_review).decode('utf-8',errors='ignore'))
    stopset = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words= stopset , use_idf= True)
    X = vectorizer.fit_transform(review_docs)
    print "TD-IDF DONE!!!"

    number_of_components = 100
    LSA = TruncatedSVD(n_components= number_of_components, algorithm= "randomized", n_iter= 100)
    Reviews_LSA = LSA.fit_transform(X)
    print "Computing LATENT SEMATIC ANALYSIS"
    columns_names_list = []
    for i in range(1,number_of_components+1):
        columns_names_list.append("component_"+str(i))
    df1 = pd.DataFrame(Reviews_LSA, columns = columns_names_list )
    df1.to_csv("Electronics_Output/LSA_Results.csv")
    return df1

def text_features(reviews_data):
    text_reviews = reviews_data['REVIEW_TEXT']
    Review_Lengths_List = []
    Avg_Word_Lengths_List = []
    Avg_Sentence_Lengths_List = []
    Capital_Words_Ratio_List = []
    Question_Exclamation_Ratio_List = []

    for each_review in text_reviews:
        word_lengths = []
        word_list = remove_stopWords(each_review)
        capital_words_count = 0
        question_exclamation_count = 0

        Review_Lengths_List.append(len(str(each_review).lower()))
        for character in str(each_review):
            if("?" == character or "!" == character):
                question_exclamation_count+=1

        for each_word in word_list:
            word_lengths.append(len(each_word))
            if(str(each_word).isupper() == True):
                capital_words_count+=1

        sentence_lengths = []
        sentences = sent_tokenize(str(each_review),language='english')
        for each_sentence in sentences:
            sentence_lengths.append(len(each_sentence))

        try:
            Avg_Word_Lengths_List.append(sum(word_lengths)/float(len(word_lengths)))
            Avg_Sentence_Lengths_List.append(sum(sentence_lengths)/float(len(sentence_lengths)))
            Capital_Words_Ratio_List.append(capital_words_count/float(len(word_list)))
            Question_Exclamation_Ratio_List.append(question_exclamation_count/float(len(str(each_review))))
        except ZeroDivisionError:
            Avg_Word_Lengths_List.append(0)
            Avg_Sentence_Lengths_List.append(0)
            Capital_Words_Ratio_List.append(0)
            Question_Exclamation_Ratio_List.append(0)

    return Review_Lengths_List , Avg_Word_Lengths_List , Avg_Sentence_Lengths_List , Capital_Words_Ratio_List, Question_Exclamation_Ratio_List

def Create_Helpfulness_Class(reviews_data):
    Binary_Class = []
    for helpful in reviews_data['HELPFUL']:
        helpful_votes = helpful.split(',')
        Helpfulness_Ratio = float(helpful_votes[0][1:])/float(helpful_votes[1][:-1])
        if(Helpfulness_Ratio >= 0.6):
            Binary_Class.append(1)
        else:
            Binary_Class.append(0)
    return Binary_Class

def Extract_Features(review_dataset):
    Review_Rating = Review_Score(review_dataset)
    Review_Length , Avg_Word_Length, Avg_Sentence_Length, Capital_Words_Ratio, Question_Exlamation_Ratio = text_features(review_dataset)
    Class = Create_Helpfulness_Class(review_dataset)
    return Review_Rating, Review_Length, Avg_Word_Length, Avg_Sentence_Length, Capital_Words_Ratio, Question_Exlamation_Ratio , Class

data_frame = pd.read_csv('Electronics_reviews_data.csv')
# print data_frame.head()
Feature_1, Feature_2, Feature_3, Feature_4, Feature_5, Feature_6, Class = Extract_Features(data_frame)
Reviews = data_frame['REVIEW_TEXT']

Feature_1_Series = pd.Series(Feature_1)
Feature_2_Series = pd.Series(Feature_2)
Feature_3_Series = pd.Series(Feature_3)
Feature_4_Series = pd.Series(Feature_4)
Feature_5_Series = pd.Series(Feature_5)
Feature_6_Series = pd.Series(Feature_6)
Class_Series     = pd.Series(Class)

df = pd.DataFrame([Feature_1_Series, Feature_2_Series, Feature_3_Series, Feature_4_Series, Feature_5_Series, Feature_6_Series, Class_Series])
df2 = df.unstack().unstack()
df2.rename(columns={0:'RW_SCORE',1:'RW_LENGTH',2:'WORD_LENGTH',3:'SENTENCE_LENGTH',4:'CAPS_RATIO',5:'QUES_EXCLAIM_RATIO', 6:'CLASS'}, inplace=True)
df2[['RW_SCORE','RW_LENGTH','WORD_LENGTH','SENTENCE_LENGTH','CAPS_RATIO','QUES_EXCLAIM_RATIO','CLASS']] = \
        df2[['RW_SCORE','RW_LENGTH','WORD_LENGTH','SENTENCE_LENGTH','CAPS_RATIO','QUES_EXCLAIM_RATIO','CLASS']].convert_objects(convert_numeric=True)
df2.to_csv("Electronics_Output/Raw_Features.csv")

df3 = pd.concat([Reviews,df2],axis=1)
print "##################################"
DATA_FOR_LSA = df3.sort('CLASS',ascending = True)[0:24000]
DATA_FOR_LSA.reset_index(drop=True,inplace=True)
DATA_FOR_LSA.to_csv('Electronics_Output/Basic_Classifier_Features.csv')
print "************************************"

LSA_RESULT = LSA_using_TF_IDF(DATA_FOR_LSA)
df4 = pd.concat([DATA_FOR_LSA, LSA_RESULT],axis=1)
df4.set_index('RW_SCORE',inplace = True)
df4.to_csv('Electronics_Output/Text_Features.csv')
print df4.head()
end_time = datetime.now()
print('Program execution Time: {}'.format(end_time - start_time))