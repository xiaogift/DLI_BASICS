#!/usr/bin/python3
# Headline generator NLP demo from Nvidia DLI course
from tensorflow.keras import utils
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import os, pandas as pd, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

nyt_dir = 'articles/'

all_headlines = []
for filename in os.listdir(nyt_dir):
    if 'Articles' not in filename: continue
    headlines_df = pd.read_csv(nyt_dir + filename)
    all_headlines.extend(list(headlines_df.headline.values))

print(len(all_headlines))
all_headlines = [h for h in all_headlines if h != "Unknown"]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_headlines)
total_words = len(tokenizer.word_index) + 1
print('Total words: ', total_words)

selected = ['a','man','a','plan','a','canal','panama']
subset_dict = {key: value for key, value in tokenizer.word_index.items() if key in selected}
print(subset_dict)
tokenizer.texts_to_sequences(selected)

input_sequences = []
for line in all_headlines:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        partial_sequence = token_list[:i+1]
        input_sequences.append(partial_sequence)

print(tokenizer.sequences_to_texts(input_sequences[:5]))
input_sequences[:5]

# Determine max sequence length
# Pad all sequences with zeros at the beginning to make them all max length
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
input_sequences[0]

# Predictors are every word except the last
# Labels are the last word
predictors = input_sequences[:,:-1]
labels = input_sequences[:,-1]
labels[:5]
labels = utils.to_categorical(labels, num_classes=total_words)

# Input is max sequence length - 1, as we've removed the last word for the label
input_len = max_sequence_len - 1 
model = Sequential()
model.add(Embedding(total_words, 10, input_length=input_len))
model.add(LSTM(100))
model.add(Dropout(0.1))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(predictors, labels, epochs=10, verbose=1)

def predict_next_token(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    prediction = model.predict_classes(token_list, verbose=0)
    return prediction

prediction = predict_next_token("today in new york")
print(prediction)

tokenizer.sequences_to_texts([prediction])

def generate_headline(seed_text, next_words=1):
    for _ in range(next_words):
        prediction = predict_next_token(seed_text) # Predict next token
        next_word = tokenizer.sequences_to_texts([prediction])[0] # Convert token to word
        seed_text += " " + next_word # Add next word to the headline and loop
    return seed_text.title()

seed_texts = [ 'washington dc is',
               'today in new york',
               'the school district has',
               'crime has become' ]

for seed in seed_texts: print(generate_headline(seed, next_words=5))

# loss: 5.6038