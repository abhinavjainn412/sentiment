#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


# In[15]:


congress_test = pd.read_csv('congress_test.csv')
bjp_test = pd.read_csv('bjp_test.csv')


# In[16]:


congress_test =congress_test[:2000]
bjp_test = bjp_test[0:2000]


# In[17]:


congress_test[0:5]


# In[18]:


bjp_test[0:5]


# In[19]:


def tweet_to_words( raw_review ):
    # Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", str(raw_review))
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', str(letters_only)) # remove URLs
    tweet = re.sub('RT', ' ', str(tweet)) 

    #Convert to lower case, split into individual words
    tweet = letters_only.lower().split()                             
    return( " ".join(tweet))


# In[20]:


# Get the number of Tweets based on the dataframe column size
num_tweets = 2000

# Initialize an empty list to hold the clean reviews


# Loop over each tweet; create an index i that goes from 0 to the length
# of the tweet list
def clean_test(dataframe):
    clean_train_tweets = []
    for i in range( 0, num_tweets ):
        # Call function for each one, and add the result to the list of
        clean_train_tweets.append( tweet_to_words(dataframe[i]))
    return clean_train_tweets


# In[21]:


congress_inputs = clean_test(congress_test['clean_text'])
bjp_inputs = clean_test(bjp_test['clean_text'])


# In[22]:


def tokenze_data(data_inputs):
        tokenizer = Tokenizer(nb_words=2000)
        tokenizer.fit_on_texts(data_inputs)
        sequences = tokenizer.texts_to_sequences(data_inputs)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        max_len = 200
        data = pad_sequences(sequences, max_len)
        print('Shape of data tensor:', data.shape)
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        return data


# In[23]:


congress_inputs = tokenze_data(congress_inputs)
bjp_inputs = tokenze_data(bjp_inputs)


# In[49]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])


# In[50]:


congress_prediction = model.predict(congress_inputs)
bjp_prediction = model.predict(bjp_inputs)


# In[54]:


congress_prediction


# In[51]:


congress_pred = (congress_prediction>0.5)
bjp_pred = (bjp_prediction>0.5)


# In[55]:


congress_pred


# In[58]:


len(congress_pred)


# In[59]:


congress_pred.shape


# In[63]:


def get_predictions(party_pred):
    x = 0
    for i in party_pred:
        if(i[0]==True):
            x+=1
    return x


# In[64]:


congress_numbers = get_predictions(congress_pred)
bjp_numbers = get_predictions(bjp_pred)
print("Congress Positive Tweets:",congress_numbers)
print("BJP Positive Tweets:",bjp_numbers)


# In[ ]:





# In[ ]:





# In[45]:





# In[27]:





# In[28]:





# In[29]:





# In[30]:





# In[42]:





# In[ ]:




