import pandas as pd
import numpy as np

df=pd.read_csv("flipkartdata.csv")
df

df['discount_price'] = df['discount_price'].str.replace(r'\W', "")
df['price'] = df['price'].str.replace(r'\W', "")
df['discount'] = df['discount'].str.replace(r'\W', "")
df['discount'] = df['discount'].str.replace(r'off', "")
df['item'] = df['item'].str.replace(r'\W', " ")
# df['color'] = df['color'].str.replace(r',', " ")
df['brand'] = df['brand'].str.lower()
df['item'] = df['item'].str.lower()
df['color'] = df['color'].str.lower()
df

l=[]
for i in df.color.str.lower().str.split(','):
#     print(i)

    if i[0]=='pack of 3':
        l.append(3)
    elif i[0]=='pack of 2':
        l.append(2)
    elif i[0]=='pack of 4':
        l.append(4)
    elif i[0]=='pack of 5':
        l.append(5)
    elif i[0]=='pack of 6':
        l.append(6)
        
    else:
        l.append(0)
l2=[]
for i in df.color.str.lower().str.split(','):
    l2.append(i)
l2

df1=pd.DataFrame(l2)
df1.head(20)


for j in df1.T:

     if df1[0][j]=='pack of 2' or df1[0][j]=='pack of 3' or df1[0][j]=='pack of 4' or df1[0][j]=='pack of 5':
#         print(j)    
        df1.iloc[j]=df1.T[j].shift(-1).T

df1.head(20)

df1.fillna("0",inplace= True)
df1.head()

diCt = {0: 'A',
        1: 'B',
        2: 'C',
       3:'D',4:'E'}
df1.rename(columns=diCt,inplace=True)

df1.A.unique()

df= pd.concat([df, df1], axis=1)
df

df['price'].fillna(df['price'].median(),inplace=True)
df['discount_price'].fillna(df['discount_price'].median(),inplace=True)
df['discount'].fillna(df['discount'].median(),inplace=True)

df.isnull().sum()

df2=pd.get_dummies(df[["brand", "A", 'B','C','D','E',"category"]],drop_first=True)

df=pd.concat([df,df2],axis = 1)
df.drop(columns = ["brand", "A", 'B','C','D','E','color','discount',"category"],inplace=True)
df.head()

df.columns

from sklearn.preprocessing import MinMaxScaler


MMS=MinMaxScaler()

df['price']=MMS.fit_transform(df[['price']])
df['discount_price']=MMS.fit_transform(df[['discount_price']])
df

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import adjusted_rand_score

vectorizer = TfidfVectorizer(stop_words='english')
vect = vectorizer.fit_transform(df["item"])
print(vect)

vect.todense().shape

vect=pd.DataFrame(vect.todense())
vect

df.columns

df.columns[2:]

type(vect)

from sklearn.metrics.pairwise import cosine_similarity
from urllib.request import urlopen
from PIL import Image

#from colab

df_img=pd.read_csv('flipkartimagesinfo.csv')

data=pd.concat([df[df.columns[2:]],vect,df_img],axis=1)

data.shape

user_to_user_sim_matrix = pd.DataFrame(cosine_similarity(data))

(user_to_user_sim_matrix)

l=[]
for i in range(4160):
    a=user_to_user_sim_matrix.loc[i].sort_values(ascending = False)
    l.append(dict(a.head(9)))

l[2000]

b=[]
for i in range(user_to_user_sim_matrix.shape[0]):
    b.append(list(user_to_user_sim_matrix.loc[i].sort_values(ascending=False)[:11].index))
#     c=user_to_user_sim_matrix.loc[i].sort_values(ascending = False)
#     b.append((c.head(10)))

b[1630]

rec=pd.DataFrame(b)
rec

rec.to_csv('similar_feat.csv',index=False)

kes=[]
# for i in range(len(l)):
#     kes.append(l[i].keys())
# int(kes[1])
for i in range(len(l)):
    kes.append( [*l[i]] )
print(kes)

for i in range(len(kes)):
    print(len(kes[i]))

len(kes)

index=[]
for i in range(4160):
        index.append(i)
    

d = dict(zip(index,kes))
d

for i in range(len(d)):
    print(len(d[i]))


kes_s=pd.Series(kes)
index_s=pd.Series(index)

kes_df=pd.DataFrame(kes_s)
index_df=pd.DataFrame(index_s)
kes_df

len(l)

data=pd.DataFrame(l)
data

df_img=pd.read_csv('flipkartdata.csv')
df_img['image'][89]

for key in l[2000]:
#     print(key)
    url=df_img['image'][key]
    print(url)
#     img = Image.open(urlopen(url))
    

df.item[2155]

df_img[df_img['item'] == 'Embroidered Art Silk Semi Stitched Anarkali Gown'].index[0]

def recommendbyitem(name):
    index = df_img[df_img['item'] == name].index[0]
    # Getting similarity products in ascending order 
    distances = sorted(list(enumerate(user_to_user_sim_matrix[index])),reverse=True,key = lambda x: x[1])
    # Just picking top 5 products which have more similarity to the movie by through index
    for i in distances[1:6]:
        print(df_img.iloc[i[0]].item)

recommendbyitem('Embroidered Art Silk Semi Stitched Anarkali Gown')

import json
# convert into json
final = json.dumps(l, indent=2)

with open('finaldata.json', 'w', encoding='utf-8') as f:
    json.dump(final, f, ensure_ascii=False, indent=2)

import pickle

pickle.dump(l,open('women_wear_recommdations.pkl','wb'))

import json
# convert into json
final = json.dumps(l, indent=2)

#saving to json file 
with open('final.json', 'w', encoding='utf-8') as f:
    json.dump(final, f, ensure_ascii=False, indent=2)



