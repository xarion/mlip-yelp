import time

import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MultiLabelBinarizer

data_root = '/Users/erdicalli/dev/workspace/yelp/data/'

train_photos = pd.read_csv(data_root + 'train_photo_to_biz_ids.csv')
train_photo_to_biz = pd.read_csv(data_root + 'train_photo_to_biz_ids.csv', index_col='photo_id')

train_df = pd.read_csv(data_root + "train_biz_fc67features.csv")

y_train = train_df['label'].values
X_train = train_df['feature vector'].values


def convert_label_to_array(str_label):
    str_label = str_label[1:-1]
    str_label = str_label.split(',')
    return [int(x) for x in str_label if len(x) > 0]


def convert_feature_to_vector(str_feature):
    str_feature = str_feature[1:-1]
    str_feature = str_feature.split(',')
    return [float(x) for x in str_feature]


y_train = np.array([convert_label_to_array(y) for y in train_df['label']])
X_train = np.array([convert_feature_to_vector(x) for x in train_df['feature vector']])

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)

random_state = np.random.RandomState(0)
classifier = OneVsRestClassifier(GaussianNB(), n_jobs=-2)
classifier.fit(X_train, y_train)
header = True
files = ["xaa", "xab", "xac", "xad", "xae", "xaf"]
for chunk in files:
    t = time.time()
    print "chunk: " + chunk
    test_df = pd.read_csv(data_root + chunk)
    X_test = test_df['feature vector'].values
    X_test = np.array([convert_feature_to_vector(x) for x in test_df['feature vector']])

    y_predict = classifier.predict(X_test)

    y_predict_label = mlb.inverse_transform(y_predict)

    print "Time passed: ", "{0:.1f}".format(time.time() - t), "sec"

    test_data_frame = pd.read_csv(data_root + chunk)
    df = pd.DataFrame(columns=['business_id', 'labels'])

    for i in range(len(test_data_frame)):
        biz = test_data_frame.loc[i]['business']
        label = y_predict_label[i]
        label = str(label)[1:-1].replace(",", " ")
        df.loc[i] = [str(biz), label]

    with open(data_root + "submission_fc67_test.csv", 'a') as f:
        df.to_csv(f, index=False, header=header)
        if header:
            header = False
