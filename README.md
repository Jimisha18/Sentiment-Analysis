# Sentiment-Analysis

The IMDB large movie review dataset contains 50,000 reviews split evenly into 25k train and 25k test sets.
The overall distribution of labels is balanced (25k positive reviews and 25k negative reviews).

That is we have 12.5k positive and negative labels under training data and same applies for the test data.
In the labeled train/test sets, retreiving the positive and negative reviews as: 
1) negative review has a score <= 4 out of 10
2) positive review has a score >= 7 out of 10.
3) neutral ratings are not included.

**I. LOAD AND VISUALIZE THE DATA:**
Download the dataset here (http://ai.stanford.edu/~amaas/data/sentiment/).

**II. Data pre-processing:**

##### 1. Removal of stop words
Removal of commonly used words unlikely to be useful for learning like 'and', 'if', 'the', etc.

##### 2. Remove punctuation marks
All the punctuation symbols can be filtered out cleaning our data with just the words to preprocess.

##### 3. Regex: 
We make use of Regular expression for text processing.

##### 4. Lower Case: 
The bunch of training and testing words are further transformed to lower case.
The labels are provided to all the data of the training and testing with the labels 0 and 1. Hence, labelled data is created as per our requirement and loaded it to a CSV file for both the training and testing data.

##### 5. Tokenization: 
The process of breaking sentences and paragraphs into individual words/tokens is called tokenization which is a very essential step for text analysis. 
Using tokenizer, we can label each word and provide a dictionary of the words being used in the sentences. We create an instance of tokenizer and assign a hyperparameter num_words to 4000. This essentially takes the most common 4000 words and tokenize them. Further, the fit_on_texts() method is used to encode the sentences.
Passing set of sentences to the ‘text_to_sequences()’ method converts the sentences to their labelled equivalent based on the corpus of words passed to it. If the corpus has a word missing that is present in the sentence, the word while being encoded to the label equivalent is omitted and the rest of the words are encoded and printed. 
To overcome such a problem, we can either use a huge corpus of words or use a hyperparameter ‘oov_token’ and assign it to a certain value which will be used to encode words previously unseen in the corpus. ‘oov_token’ can be assigned to anything but one should assign a unique value to it so that it isn’t confused with an original word.

##### 6. Padding:
Padding is done after the sentence, the hyperparameter padding is set to ‘post’. Padding is generally done with reference to the longest sentence, however the hyperparameter maxlen can be provided to override it and define the maximum length of the sentence.

Using the training and testing data for network, performed data preprocessing to prepare the data for NLP. 
Then, created a network to classify the review as positive or negative.

**III. Creating LSTM Model:**
Embedding Layer that converts our word tokens (integers) into embedding of specific size. LSTM Layer has hidden layer of 250 neurons then a dropout of 0.2 is applies and finally there is the softmax activation layer. Training the model on 10 epochs and batch size 32.

Training the network using sparse_categorical_crossentropy function and Adam optimizer. Hence, 97% training accuracy is achieved for the model and this model is saved as 20853543_NLP_model.
Further, the testing data is loaded and preprocessed the similar way and tested on the trained model providing an accuracy of 80.6% and test loss is 1.83.
