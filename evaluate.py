
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


with open('features_test.txt', 'r') as f:
    actual = f.readlines()

with open('predictions.txt', 'r') as f:
    predictions = f.readlines()

predictions = [a[0] if len(a) > 1 else '' for a in predictions]
actual = [a[-2] if len(a) > 1 else '' for a in actual]


# In[3]:


tags = ['O', 'T', 'D']
counts = {}

for tag in tags:
    counts[tag] = {tag2: 0 for tag2 in tags}

for i, j in zip(actual, predictions):
    if i == '':
        assert j == ''
    else:
        counts[i][j] += 1


# In[4]:


print counts


# In[5]:


result = {}

for i in tags:
    print 'Original tag:', i
    print '----------------'
    
    precision_demominator = 0
    recall_demominator = 0
    
    for j in tags:
        recall_demominator += counts[j][i]
        precision_demominator += counts[i][j]
            
        print j, '\t', counts[i][j], '\t%.2f%%' % (float(counts[i][j])/sum(counts[i].values()) * 100)
    
    precision = counts[i][i]/float(precision_demominator)
    recall    = counts[i][i]/float(recall_demominator)
    
    print
    print 'Precision: ', precision * 100
    print 'Recall:    ', recall * 100
    print 'F1 measure:', 2*precision*recall/float(precision+recall) * 100
    print
    
    result[i] = (np.round(precision * 100, 2),
                 np.round(recall * 100, 2),
                 np.round(2*precision*recall/float(precision+recall) * 100, 2))


# In[6]:


numerator = 0; denominator = 0

for i in tags:
    numerator += counts[i][i]
    
    for j in tags:
        denominator += counts[i][j]

print 'Accuracy:  ', (numerator)/float(denominator)

