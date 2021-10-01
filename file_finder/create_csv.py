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
train_df.to_csv(r'train_csv.csv', index =False, header = True)