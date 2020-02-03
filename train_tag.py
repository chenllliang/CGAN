#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import re


# In[9]:


clean_tag = pd.read_csv("tags.csv",header=None,names=["id","description"])


# In[71]:


def get_train_tag(text):
    #only save hair and eyes description using regular expressions
    hair_descriptions = re.findall(r"[a-z]+ hair",text)
    eyes_descriptions = re.findall(r"[a-z]+ eyes",text)
    
    if len(hair_descriptions)==0 and len(eyes_descriptions)==0:
        return 0
    
    if len(hair_descriptions)>1:
        hair=""
        for x in hair_descriptions:
            q = x.split(" ")
            hair=hair+q[0]+" "
        hair=hair+"hair"
    elif len(hair_descriptions)==1:
        hair=hair_descriptions[0]
    else:
        hair = 0
        
    if len(eyes_descriptions)>1:
        eyes=""
        for x in eyes_descriptions:
            q = x.split(" ")
            eyes=eyes+q[0]+" "
        eyes=eyes+"eyes"
    elif len(eyes_descriptions)==1:
        eyes=eyes_descriptions[0]
    else:
        eyes = 0
    
    return hair,eyes


# In[73]:


train_tag={}
for index, row in clean_tag.iterrows():
    if get_train_tag(row["description"])==0:
        continue
    hair,eyes = get_train_tag(row["description"])
    
    if hair!=0 and eyes !=0:
        description = hair + " and " + eyes
    elif hair ==0:
        description=eyes
    elif eyes ==0:
        description=hair
    
    train_tag[row["id"]]=description
    


# In[76]:


len(train_tag)
#共18121/33400张有效图片


# In[75]:


f=open("train_tag_dict.txt","w")
f.write(str(train_tag))
f.close()


# In[ ]:




