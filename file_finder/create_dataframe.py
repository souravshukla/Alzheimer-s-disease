"Alzheinet Project started on 24th Sept"

import pandas as pd
import os
from sklearn.model_selection import train_test_split

path = 'Dataset/train/'
foldernames = os.listdir('Dataset/train')
train_list = []

for foldername in foldernames:
    joined = path + '/' + foldername
    for filename in os.listdir(path + '/' + foldername ):
        train_list.append([filename, foldername, path + foldername + '/' + filename])

train_df = pd.DataFrame(train_list, columns = ['filename', 'group', 'address'])

train_df['label'] = train_df['group'].map({'ad': 0, 'mci' : 1, 'cn' : 2})
train_df.to_csv(r'Input_file/train_csv.csv', index =False, header = True)

print(train_df.head())
print(train_df['group'].value_counts())

X_train, X_val, y_train,y_test = train_test_split(train_df['address'],train_df['label'],test_size = 0.15,shuffle = True)

print(len(X_train), len(X_val))

