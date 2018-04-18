
# coding: utf-8

# In[1]:


import sqlite3
import numpy as np


# In[2]:


conn = sqlite3.connect('sentence_bank.sqlite3')
cursor = conn.cursor()
cursor2 = conn.cursor()
cursor3 = conn.cursor()


# Make splits

# In[3]:


cursor.execute('SELECT id FROM sentences ORDER BY id;')
sentence_ids = []

for row in cursor:
    sentence_ids += [row[0]]


# In[4]:


np.random.seed(1337)
train = list(np.random.choice(sentence_ids, size=int(len(sentence_ids)*0.7), replace=False))
valid = np.random.choice(list(set(sentence_ids) - set(train)), size=int(len(sentence_ids)*0.1), replace=False)
test = list(set(sentence_ids) - set(train) - set(valid))


# In[5]:


print 'Splits:', len(train), len(valid), len(test)
print len(sentence_ids)


# Get vocabulary

# In[22]:


# Prepare vocabulary
vocabulary = ['@@PAD@@', '@@SOS@@', '@@EOS@@']

for word, count in cursor.execute('SELECT DISTINCT LOWER(word), COUNT(LOWER(word)) FROM words GROUP BY word ORDER BY COUNT(LOWER(word)) DESC;'):
    vocabulary.append(word)


# In[23]:


restrict_vocab_size = None


# In[28]:


if restrict_vocab_size is not None:
    vocabulary = vocabulary[:restrict_vocab_size]
    vocabulary.append('@@OOV@@')


# In[29]:


len(vocabulary)


# In[30]:


dictionary_fwd = dict(zip(vocabulary, range(len(vocabulary))))
dictionary_bwd = dict(zip(range(len(vocabulary)), vocabulary))


# In[31]:


labels = ['O', 'T', 'D']


# In[32]:


dictionary_out_fwd = dict(zip(labels, range(len(labels))))
dictionary_out_bwd = dict(zip(range(len(labels)), labels))


# Extraction

# In[39]:


train_x = []
train_y = []
train_l = []

valid_x = []
valid_y = []
valid_l = []

test_x  = []
test_y  = []
test_l  = []

for sentence_id, in cursor.execute('SELECT * FROM sentences;'):
    sentence = []
    cursor2.execute('''
    SELECT LOWER(word), tag
    FROM words
    JOIN pos_tags ON words.sentence_id = pos_tags.sentence_id AND words.word_id = pos_tags.word_id                       
    WHERE words.sentence_id = ? ORDER BY words.word_id''', (sentence_id,))
    
    x = [dictionary_fwd['@@SOS@@']]
    y = [dictionary_out_fwd['O']]
    l = 1
    
    for row in cursor2:        
        x.append(dictionary_fwd[row[0]])
        y.append(dictionary_out_fwd[row[1]])
        l += 1
        
    x.append(dictionary_fwd['@@EOS@@'])
    y.append(dictionary_out_fwd['O'])
    l += 1
    
    if sentence_id in train:
        train_x.append(x)
        train_y.append(y)
        train_l.append(l)
    if sentence_id in valid:
        valid_x.append(x)
        valid_y.append(y)
        valid_l.append(l)
    if sentence_id in test:
        test_x.append(x)
        test_y.append(y)
        test_l.append(l)


# In[40]:


maxlen = max(map(len, train_x) + map(len, valid_x) + map(len, test_x))


# In[57]:


for i in range(len(train_x)):
    #train_x[i] = [0] * (maxlen - len(train_x[i])) + train_x[i]
    train_y[i] = [0] * (maxlen - len(train_y[i])) + train_y[i]
    
for i in range(len(valid_x)):
    #valid_x[i] = [0] * (maxlen - len(valid_x[i])) + valid_x[i]
    valid_y[i] = [0] * (maxlen - len(valid_y[i])) + valid_y[i]
    
for i in range(len(test_x)):
    #test_x[i] = [0] * (maxlen - len(test_x[i])) + test_x[i]
    test_y[i] = [0] * (maxlen - len(test_y[i])) + test_y[i]


# In[64]:


train_x, train_y, valid_x, valid_y, test_x, test_y = np.array(train_x), np.array(train_y), np.array(valid_x), np.array(valid_y), np.array(test_x), np.array(test_y)


# In[69]:


np.save('train_x.npy', train_x); np.save('train_y.npy', train_y); np.save('train_l.npy', train_l)
np.save('valid_x.npy', valid_x); np.save('valid_y.npy', valid_y); np.save('valid_l.npy', valid_l)
np.save('test_x.npy', test_x); np.save('test_y.npy', test_y); np.save('test_l.npy', test_l)


# In[70]:


np.save('dictionary_fwd.npy', dictionary_fwd)
np.save('dictionary_bwd.npy', dictionary_bwd)
np.save('dictionary_out_fwd.npy', dictionary_out_fwd)
np.save('dictionary_out_bwd.npy', dictionary_out_bwd)

