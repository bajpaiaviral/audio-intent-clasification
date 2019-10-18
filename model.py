import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import nltk
import re
import os
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint


def load_dataset(filename):
  df = pd.read_csv(filename, encoding = "latin1", names = ["Sentence", "Intent"])
  print(df.head())
  intent = df["Intent"]
  unique_intent = list(set(intent))
  sentences = list(df["Sentence"])
  return (intent, unique_intent, sentences)


df= pd.read_excel("/Users/aviral/Desktop/tatadata78.xlsx")


#df.head()
import sklearn.utils
df = sklearn.utils.shuffle(df)
df['Convo']


sentences=df["Convo"]
intent=df["Whatabout"]

type(sentences)

unique_intent=pd.Series(list(set(intent)))
dic={'TestDrive':'Test Drive Requests','Breakdown':'Breakdown','Complaint':'Vehicle Quality','Enquiry':'New Vehicle Purchase Enquiries','Feedback':'Feedback'}



nltk.download("stopwords")
nltk.download("punkt")


#define stemmer
stemmer = LancasterStemmer()


def cleaning(sentences):
  words = []
  for s in sentences:
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
    w = word_tokenize(clean)
    #stemming
    words.append([i.lower() for i in w])
    
  return words  

cleaned_words = cleaning(sentences)
print(len(cleaned_words))
print(cleaned_words[:2])  
  
def create_tokenizer(words, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
  token = Tokenizer(filters = filters)
  token.fit_on_texts(words)
  return token

def max_length(words):
  return(len(max(words, key = len)))
  

word_tokenizer = create_tokenizer(cleaned_words)
vocab_size = len(word_tokenizer.word_index) + 1
max_length = max_length(cleaned_words)

print("Vocab Size = %d and Maximum length = %d" % (vocab_size, max_length))


def encoding_doc(token, words):
  return(token.texts_to_sequences(words))


encoded_doc = encoding_doc(word_tokenizer, cleaned_words)


def padding_doc(encoded_doc, max_length):
  return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))


padded_doc = padding_doc(encoded_doc, max_length)


print("Shape of padded docs = ",padded_doc.shape)

#tokenizer with filter changed
output_tokenizer = create_tokenizer(unique_intent, filters = '!"#$%&()*+,-/:;<=>?@[\]^`{|}~')

output_tokenizer.word_index


encoded_output = encoding_doc(output_tokenizer, intent)


encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)


encoded_output.shape


def one_hot(encode):
  o = OneHotEncoder(sparse = False)
  return(o.fit_transform(encode))


output_one_hot = one_hot(encoded_output)


output_one_hot.shape


from sklearn.model_selection import train_test_split


train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, shuffle = True, test_size = 0.1)


print("Shape of train_X = %s and train_Y = %s" % (train_X.shape, train_Y.shape))
print("Shape of val_X = %s and val_Y = %s" % (val_X.shape, val_Y.shape))


from keras import regularizers
def create_model(vocab_size, max_length):
  model = Sequential()
  model.add(Embedding(vocab_size, 64, input_length = max_length, trainable = False))
  model.add(Bidirectional(LSTM(16)))
#   model.add(LSTM(128))
  model.add(Dense(32, activation = "relu"))
  model.add(Dropout(0.5))
  model.add(Dense(5, activation = "softmax"))
  
  return model

model = create_model(vocab_size, max_length)

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.summary()


filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

hist = model.fit(train_X, train_Y, epochs = 200, batch_size = 4, validation_data = (val_X, val_Y), callbacks = [checkpoint])


model = load_model("model.h5")


def predictions(text):
  clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
  test_word = word_tokenize(clean)
  test_word = [w.lower() for w in test_word]
  test_ls = word_tokenizer.texts_to_sequences(test_word)
  print(test_word)
  #Check for unknown words
  if [] in test_ls:
    test_ls = list(filter(None, test_ls))
    
  test_ls = np.array(test_ls).reshape(1, len(test_ls))
 
  x = padding_doc(test_ls, max_length)
  
  pred = model.predict_proba(x)
  
  
  return pred

def get_final_output(pred, classes):
  predictions = pred[0]
 
  classes = np.array(classes)
  ids = np.argsort(-predictions)
  classes = classes[ids]
  predictions = -np.sort(-predictions)
 
  for i in range(pred.shape[1]):
    if(i==0):
        predictionans=classes[i]
    print("%s has confidence = %s" % (classes[i], predictions[i]))
  return   predictionans


import speech_recognition as sr 
  
AUDIO_FILE = ("/Users/aviral/Desktop/test1.wav") 
  
# use the audio file as the audio source 
  
r = sr.Recognizer() 
  
with sr.AudioFile(AUDIO_FILE) as source: 
    #reads the audio file. Here we use record instead of 
    #listen 
    audio = r.record(source)   
  
try: 
    print("The audio file contains: " + r.recognize_google(audio)) 
    text=r.recognize_google(audio)
  
except sr.UnknownValueError: 
    print("Google Speech Recognition could not understand audio") 
  
except sr.RequestError as e: 
    print("Could not request results from Google Speech Recognition service; {0}".format(e)) 


from googletrans import Translator
translator = Translator()
translations = translator.translate(text, dest='en',src='hi')
text=translations.text

pred=predictions(text)
answer=get_final_output(pred, unique_intent)
print('ANSWER....',dic[answer])
import csv
row = [AUDIO_FILE, answer]
with open('/Users/aviral/Desktop/HBTU_I-Z5BCJ.csv', 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(row)
csvFile.close()
