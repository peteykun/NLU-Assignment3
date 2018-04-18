
# coding: utf-8

# In[89]:


import tarfile
import xml.etree.ElementTree as ET
import tqdm
import codecs


# In[87]:


members = []

tar = tarfile.open("unlabeled.tar.gz", "r:gz")
outfile = codecs.open("unlabeled.txt", 'w', 'utf-8')

for member in tar:
    f = tar.extractfile(member)
    
    if f is None:
        continue
    else:
        print(member.name)
    
    root = ET.fromstring(f.read())
    
    result = ''

    for para in root.iter('p'):
        result = ''
        
        if para.text is not None:
            result += para.text

        for child in para:
            if child is not None:
                if child.text is not None:
                    result += child.text
                if child.tail is not None:
                    result += child.tail

        outfile.write(result + "\n")
    
    outfile.flush()

outfile.close()
tar.close()
