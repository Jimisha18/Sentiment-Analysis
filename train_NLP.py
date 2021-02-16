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
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# YOUR IMPLEMENTATION

def create_dataset():
    #Creating a list of training data and training labels
    neg_dir = "data/aclImdb/train/neg/"
    pos_dir = "data/aclImdb/train/pos/"
    neg_file = os.listdir(neg_dir)
    pos_file = os.listdir(pos_dir)
    
    neg_file = [neg_dir+filename for filename in neg_file]
    pos_file = [pos_dir+filename for filename in pos_file]
    training_files = neg_file+pos_file
    
    #english stopwords
    stpwords = set(stopwords.words('english'))
    item = str.maketrans("", "", string.punctuation)
    
    training_data = []
    training_labels = []
    
    count = 0
    for filename in training_files:
        #Cleaning the data
        with open(filename,'r',encoding='utf-8') as f:
            single_review = f.read()
            single_review = re.sub("<br />","", single_review)
            single_review = single_review.translate(item)
            
            words = []
            for i in single_review.split():
                if i.lower() not in stpwords:
                    words.append(i.lower())
            single_review = " ".join(words)
            
            training_data.append(single_review)
            #Creating the training labels
            count = count+1
            if (count>len(neg_file)):
                training_labels.append(1)
            else:
                training_labels.append(0)
    #Final training data achieved            
    combined_data = pd.DataFrame(np.c_[training_data,training_labels], columns = ["Review", "Opinion"])
    return combined_data


if __name__ == "__main__": 
    
	# 1. load your training data
    combined_data = create_dataset()
    combined_data.to_csv('data/train_data_NLP.csv', index=False)
    combined_data = shuffle(combined_data,random_state=42)
    print(combined_data.head())
    
    training_data = combined_data["Review"]
    training_labels = combined_data["Opinion"]
    #Tokenization
    tokens = Tokenizer(num_words=4000, oov_token=True)
    tokens.fit_on_texts(training_data)
    #Padding the sequence
    x = tokens.texts_to_sequences(training_data)
    x = pad_sequences(x,maxlen=200, padding="post")
    
    #Running the LSTM model
    nlp_model = Sequential()
    nlp_model.add(Embedding(4000,200, input_length=200))
    nlp_model.add(LSTM(250,dropout=0.2))
    nlp_model.add(Dense(2,activation="softmax"))
    
	# 2. Train your network
    nlp_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    nlp_model.fit(x, training_labels, epochs = 10, batch_size=32, verbose=1)
    

	# 3. Save your model
    nlp_model.summary()
    nlp_model.save("models/20853543_NLP_model")
