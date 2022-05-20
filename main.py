import nltk
from nltk.stem.lancaster import LancasterStemmer
#nltk.download("punkt")
stemmer=LancasterStemmer()

import numpy
# import tflearn
#import tensorflow
import random
import json

with open("intents.json") as file:
    data = json.load(file)
 
words=[]
labels=[]
docs=[]

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds=nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs.append(pattern)

        if intent["tag"] not in labels:
            labels.append(intent["tag"])
