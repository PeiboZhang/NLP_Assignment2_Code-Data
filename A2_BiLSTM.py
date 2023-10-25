
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.porter import *
from keras import backend as K
from kerastuner.tuners import RandomSearch


train_data = pd.read_csv('/Users/peibo1/Desktop/BMI 550/Assignment 2/fallreports_2023-9-21_train.csv')
train_has_na = train_data['fall_description'].isna().sum()
print(train_has_na) # 3 NA record in the fall description
train_data = train_data.dropna(subset=['fall_description'])
print(train_data['record_id'].nunique()) # 35 patients in the training dataset (26 data in the original dataset, but that patient does not have description value)


test_data = pd.read_csv('/Users/peibo1/Desktop/BMI 550/Assignment 2/fallreports_2023-9-21_test.csv')
has_na = test_data['fall_description'].isna().sum()
print(has_na) # 2 NA record in the fall description
test_data = test_data.dropna(subset=['fall_description'])
print(test_data['record_id'].nunique()) #25 patients in the test dataset (26 data in the original dataset, but that patient does not have description value)


st = stopwords.words('english')
stemmer = PorterStemmer()

def preprocess_text(raw_text):
    words = [stemmer.stem(w) for w in raw_text.lower().split() if w not in st]
    return (" ".join(words))

train_data['fall_description'] = train_data['fall_description'].apply(preprocess_text)
test_data['fall_description'] = test_data['fall_description'].apply(preprocess_text)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['fall_description'])
X_train_sequences = tokenizer.texts_to_sequences(train_data['fall_description'])
X_test_sequences = tokenizer.texts_to_sequences(test_data['fall_description'])

maxlen = max([len(s) for s in X_train_sequences])
print ('Maximum sequence length:', maxlen)

plt.hist([len(s) for s in X_train_sequences])
plt.show() # from the plot, due to the slight increase in distribution around the 70, I decided to still kept the original max length

X_train = pad_sequences(X_train_sequences)
X_test = pad_sequences(X_test_sequences, maxlen=X_train.shape[1])


y_train = train_data['fog_q_class'].values
y_test = test_data['fog_q_class'].values


def f1_score(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def build_model(input_dim, input_length):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=128, input_length=input_length))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

vocab_size = len(tokenizer.word_index) + 1
model = build_model(input_dim=vocab_size, input_length=X_train.shape[1])

X_train_part, X_val_part, y_train_part, y_val_part = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
model.fit(X_train_part, y_train_part, validation_data=(X_val_part, y_val_part), epochs=5, batch_size=32)

class BiLSTMHyperModel(HyperModel):
    def __init__(self, input_dim, input_length):
        self.input_dim = input_dim
        self.input_length = input_length

    def build(self, hp):
        model = Sequential()
        
        
        model.add(Embedding(input_dim=self.input_dim,
                            output_dim=hp.Int('embedding_output_dim', min_value=64, max_value=256, step=32),
                            input_length=self.input_length))
        
       
        model.add(Bidirectional(LSTM(units=hp.Int('lstm_units_1', min_value=32, max_value=128, step=32),
                                      return_sequences=True)))
        
        
        model.add(Bidirectional(LSTM(units=hp.Int('lstm_units_2', min_value=16, max_value=64, step=16))))
        
        model.add(Dense(1, activation='sigmoid'))
        
        
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        return model

hypermodel = BiLSTMHyperModel(input_dim=vocab_size, input_length=X_train.shape[1])


tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=10,  
    executions_per_trial=1,  
    directory='keras_tuner_dir',
    project_name='bilstm_tuning'
)


tuner.search(X_train_part, y_train_part, validation_data=(X_val_part, y_val_part), epochs=5, batch_size=32)
tuner.results_summary()

best_model = tuner.get_best_models(num_models=1)[0]

best_model.fit(X_train, y_train, epochs=5, batch_size=32)

loss, accuracy = best_model.evaluate(X_test, y_test)
print(f"BiLSTM Test accuracy: {accuracy}")
