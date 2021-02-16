# import required packages

import os
import re
import nltk
import string
import numpy as np
import pandas as pd
from  sklearn.utils import shuffle
from nltk.corpus import stopwords
nltk.download('stopwords')
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



# YOUR IMPLEMENTATION

def create_dataset():
    #Creating a list of testing data and testing labels
    neg_dir = "data/aclImdb/test/neg/"
    pos_dir = "data/aclImdb/test/pos/"
    neg_file = os.listdir(neg_dir)
    pos_file = os.listdir(pos_dir)
    
    neg_file = [neg_dir+filename for filename in neg_file]
    pos_file = [pos_dir+filename for filename in pos_file]
    testing_files = neg_file+pos_file
    
    #english stopwords
    stpwords = set(stopwords.words('english'))
    item = str.maketrans("", "", string.punctuation)
    
    testing_data = []
    testing_labels = []

    count = 0
    for filename in testing_files:
        with open(filename,'r',encoding='utf-8') as f:
            #Cleaning the data
            single_review = f.read()
            single_review = re.sub("<br />","", single_review)
            single_review = single_review.translate(item)
            
            words = []
            for i in single_review.split():
                if i.lower() not in stpwords:
                    words.append(i.lower())
            single_review = " ".join(words)
            
            testing_data.append(single_review)
            #Creating labels for test data
            count = count+1
            if (count>len(neg_file)):
                testing_labels.append(1)
            else:
                testing_labels.append(0)
    #Final testing data achieved               
    combined_testdata = pd.DataFrame(np.c_[testing_data,testing_labels], columns = ["Review", "Opinion"])
    return combined_testdata

if __name__ == "__main__":
    
    # 1. Load your saved model
    
    nlp_model=load_model("models/20853543_NLP_model")
    
    # 2. Load your testing data
    
    combined_testdata = create_dataset()
    combined_testdata.to_csv('data/test_data_NLP.csv', index=False)
    combined_testdata = shuffle(combined_testdata,random_state=42)
    print(combined_testdata.head())
    
    testing_data = combined_testdata["Review"]
    testing_labels = combined_testdata["Opinion"]
    #Tokenization
    tokens = Tokenizer(num_words=4000, oov_token=True)
    tokens.fit_on_texts(testing_data)
    #Padding
    x_test = tokens.texts_to_sequences(testing_data)
    x_test = pad_sequences(x_test,maxlen=200, padding="post")
    
	# 3. Run prediction on the test data and print the test accuracy
    
    test_loss, test_acc = nlp_model.evaluate(x_test, testing_labels, verbose=1)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)
