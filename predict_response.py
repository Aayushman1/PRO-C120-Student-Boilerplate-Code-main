#Text data preprocessing libraries
import nltk #natural language toolkit (for processing the user language)
import json
import pickle
import numpy as np #Library to work on arrays
import random

ignore_words = ['?', '!',',','.', "'s", "'m"]
 
#Model loading libraries
import tensorflow
from data_preprocessing import get_stem_words
model=tensorflow.keras.models.load_model("./chatbot_model.h5")

#Load data files
intents=json.loads(open('./intents.json').read())
words=pickle.load(open('./words.pkl', 'rb')) #read binary
classes=pickle.load(open('./classes.pkl', 'rb'))

#Funtion to preprocess the input given by the user
def preprocess_user_input(user_input):
    input_word_token_1 = nltk.word_tokenize(user_input) 
    input_word_token_2 = get_stem_words(input_word_token_1, ignore_words) 
    input_word_token_2 = sorted(list(set(input_word_token_2)))
    
    bag=[]
    bag_of_words=[]
    for word in words:
        if word in input_word_token_2:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
    bag.append(bag_of_words)
    
    return np.array(bag)

#funtion to give labels
def bot_class_prediction(user_input):
    inp=preprocess_user_input(user_input)
    prediction=model.predict(inp)
    predicted_class_label=np.argmax(prediction[0])
    
    return predicted_class_label

#funtion to return the response by the bot
def bot_response(user_input):
    predicted_class_label=bot_class_prediction(user_input)
    predicted_class=classes[predicted_class_label]
    for intent in intents['intents']:
        if intent['tag']==predicted_class:
            bot_response=random.choice(intent['responses'])
            
            return bot_response

print('Hi, I am Max, How can I help you?')

while True:
    user_input=input('Type your message here: ')
    print('User input: ', user_input)
    response=bot_response(user_input)
    print('Bot response: ', response)