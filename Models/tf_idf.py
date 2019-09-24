import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
import sklearn.datasets as skds
from pathlib import Path
from sklearn import metrics

def plotCM(mtrx):
	import seaborn as sn
	import pandas as pd
	import matplotlib.pyplot as plt
    
	df_cm = pd.DataFrame(mtrx, range(10),
	                  range(10))
	plt.figure(figsize = (10,7))
	sn.set(font_scale=1.4)#for label size
	sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
	plt.matshow(df_cm)


train_df = pd.read_csv('../data/processed/processed_train.csv')
test_df = pd.read_csv('../data/processed/processed_valid.csv')
train_data = train_df['cleaned_query'].tolist()
train_labels = train_df['label'].tolist()
test_data = tes_df['cleaned_query'].tolist()
test_labels = test_df['label'].tolist()

# 20 news groups
num_labels = 11
vocab_size = 10000
batch_size = 100
 
# define Tokenizer with Vocab Size
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_data)
 
x_train = tokenizer.texts_to_matrix(train_data, mode='tfidf')
x_test = tokenizer.texts_to_matrix(test_data, mode='tfidf')
 
encoder = LabelBinarizer()
encoder.fit(train_labels)
y_train = encoder.transform(train_labels)
y_test = encoder.transform(test_labels)

model = Sequential()
model.add(Dense(512, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()
 
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=3,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
 
print('Test accuracy:', score[1])

model.model.save('my_model.h5')
 
# Save Tokenizer i.e. Vocabulary
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)



###########

# model = load_model('my_model.h5')
 
# # load tokenizer
# tokenizer = Tokenizer()
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

##########

preds = model.predict(x_test)

matrix = metrics.confusion_matrix(y_test.argmax(axis=1), preds.argmax(axis=1))

print(matrix)

plotCM(matrix)

