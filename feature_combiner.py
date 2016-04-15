import numpy as np
import pandas as pd
import sklearn


def convert_label_to_array(str_label):
    str_label = str_label[1:-1]

    str_label = str_label.split(',')
    return [int(x) for x in str_label if len(x) > 0]


def convert_feature_to_vector(str_feature):
    str_feature = str_feature[1:-1]
    str_feature = str_feature.split(',')
    return [float(x) for x in str_feature]


data_root = '/Users/erdicalli/dev/workspace/yelp/data/'

train_df = pd.read_csv(data_root + "train_merged.csv")

labels = np.array([convert_label_to_array(y) for y in train_df['label']])
features = np.array([convert_feature_to_vector(x) for x in train_df['feature vector']])
features = features[:, :-9]

f = sklearn.preprocessing.normalize(np.append(sklearn.preprocessing.normalize(features[:, :8192]), sklearn.preprocessing.normalize(features[:, 8192:]), axis=1))

f.shape

print "1"