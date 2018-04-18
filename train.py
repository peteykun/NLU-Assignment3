
# coding: utf-8

# In[1]:

import numpy as np, keras
#from mr import mr
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional, TimeDistributed, Masking
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


# In[2]:

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))


# In[3]:

embedding_dim = 200 + 300


# In[4]:

embeddings_index = {}
f = open('/home1/e1-246-28/NLU-Assignment3/glove.6B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[5]:

from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('unlabeled-vector.bin', binary=True, unicode_errors='ignore') 
print('Found %s word vectors.' % len(model.index2word))


# In[6]:

X_train = np.load('train_x.npy'); y_train = np.load('train_y.npy'); l_train = np.load('train_l.npy')
X_valid = np.load('valid_x.npy'); y_valid = np.load('valid_y.npy'); l_valid = np.load('valid_l.npy')
X_test  = np.load('test_x.npy' ); y_test  = np.load('test_y.npy' ); l_test  = np.load('test_l.npy' )


# In[7]:

vocab_size = np.max(list(set.union(*[set(x) for x in X_train]))) + 1


# In[8]:

max_length = len(X_train[0])


# In[9]:

dictionary_fwd = np.load('dictionary_fwd.npy').item()


# In[10]:

embedding_matrix = np.zeros((vocab_size, embedding_dim))
found1 = 0; found2 = 0

for word, i in dictionary_fwd.iteritems():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        if i >= vocab_size:
            continue
        
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i][:300] = embedding_vector
        found1 += 1
    
    if word in model:
        if i >= vocab_size:
            continue
        
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i][300:] = model[word]
        found2 += 1

print found1, 'found in word2vec of', len(dictionary_fwd)
print found2, 'found in mesh of', len(dictionary_fwd)


# In[11]:

# create the model
model = Sequential()
model.add(Masking(mask_value=0, input_shape=(127,)))
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length,                     weights=[embedding_matrix], trainable=True))
model.add(Bidirectional(GRU(150, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
model.add(Bidirectional(GRU(150, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
model.add(Bidirectional(GRU(150, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
model.add(TimeDistributed(Dense(3, activation='softmax')))


# In[12]:

print model.summary()


# In[13]:

from keras.callbacks import ModelCheckpoint


# In[14]:

mcp = ModelCheckpoint('bidir_rnn_meshword2vec.hdf5', monitor="val_acc", save_best_only=True, save_weights_only=False)


# In[15]:

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[16]:

y_train2 = np.array([to_categorical(y, 3) for y in y_train])
y_valid2 = np.array([to_categorical(y, 3) for y in y_valid])
y_test2 = np.array([to_categorical(y, 3) for y in y_test])


# In[22]:

model.fit(X_train, y_train2, epochs=10, batch_size=128, validation_data=(X_valid, y_valid2), callbacks=[mcp])


# (total of 30)

# In[23]:

model.load_weights('bidir_rnn_meshword2vec.hdf5')


# In[24]:

result = model.evaluate(X_test, y_test2, batch_size=128)


# In[25]:

predictions = model.predict(X_test)


# In[27]:

with open('predictions.txt', 'w') as f:
    for tokens, prediction in zip(X_test, predictions):
        started = False

        for tok, pred in zip(tokens, prediction):
            if not started and tok == 1:
                started = True
            elif started and tok == 2:
                f.write('\n')
                break
            elif started:
                if np.argmax(pred) == 0:
                    f.write('O\n')
                elif np.argmax(pred) == 1:
                    f.write('T\n')
                elif np.argmax(pred) == 2:
                    f.write('D\n')

