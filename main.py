import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense

data = pd.read_csv("./zoo.csv", sep=",", encoding="utf8")
df_features = data.iloc[:, 1:-1]
df_labels = data.iloc[:, -1]
labels = pd.DataFrame(df_labels)
ordered_labels = ["shellfish", "amphibian", "reptile", "insect", "fish", "mammal", "bird"]
labels.columns = ["animal"]
labels_cat = pd.Categorical(df_labels, ordered_labels, ordered=True).codes
labels_df = pd.DataFrame(labels_cat)
labels["cat"] = labels_df
labels_codes = labels.cat

x_train,  x_test, y_train, y_test = train_test_split(df_features, labels_codes, test_size=0.2, random_state=23)

def get_my_model():
    my_model = Sequential()
    input = InputLayer(input_shape=(df_features.shape[1], ))
    my_model.add(input)
    my_model.add(Dense(16, activation='relu'))
    my_model.add(Dense(1))
    opt = Adam(learning_rate=0.01)
    my_model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    my_model.fit(tf.convert_to_tensor(x_train), tf.convert_to_tensor(y_train), epochs=50, batch_size=1)
    my_model.evaluate(x_test, y_test)
    return my_model

model = get_my_model()
example = [[1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 4, 1, 0, 1]]
prediction = model.predict(example)
model_index = round(prediction[0][0])
print(ordered_labels[model_index])