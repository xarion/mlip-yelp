import sys
import time

import numpy as np
import pandas as pd
from sklearn import clone, cross_validation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegressionCV, ElasticNetCV, LarsCV, LassoCV, LassoLarsCV, \
    MultiTaskElasticNetCV, \
    MultiTaskLassoCV, OrthogonalMatchingPursuitCV, RidgeClassifierCV
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data_root = '/Users/erdicalli/dev/workspace/yelp/data/'
submission_root = '/Users/erdicalli/dev/workspace/yelp/submission/'

train_df = pd.read_csv(data_root + "train_merged.csv")


def convert_label_to_array(str_label):
    str_label = str_label[1:-1]
    str_label = str_label.split(',')
    return [int(x) for x in str_label if len(x) > 0]


def convert_feature_to_vector(str_feature):
    str_feature = str_feature[1:-1]
    str_feature = str_feature.split(',')
    return [float(x) for x in str_feature]


labels = np.array([convert_label_to_array(y) for y in train_df['label']])
features = np.array([convert_feature_to_vector(x) for x in train_df['feature vector']])
features = features[:, :-9]
features = normalize(np.append(normalize(features[:, :8192]), normalize(features[:, 8192:]), axis=1))
# for f, b in zip(features, train_df["business"]):
#     if len(f) != 8446:
#         print b



names = [
    "SVC(kernel='rbf', probability=True)",
    "SVC(kernel='linear', probability=True)",
    "SVC(kernel='sigmoid', probability=True)",
    "SVC(kernel='poly', probability=True, degree=3)",
    "SVC(kernel='poly', probability=True, degree=4)",
    "SVC(kernel='poly', probability=True, degree=5)",
    "DecisionTreeClassifier()",
    "KNeighborsClassifier()",
    "GaussianNB()",
    "RandomForestClassifier()",
    "AdaBoostClassifier()",
    "QuadraticDiscriminantAnalysis()",
    "LinearDiscriminantAnalysis()",
    "ElasticNetCV()",
    "LarsCV()",
    "LassoCV()",
    "LassoLarsCV()",
    "LogisticRegressionCV()",
    "MultiTaskElasticNetCV()",
    "MultiTaskLassoCV()",
    "OrthogonalMatchingPursuitCV()",
    "RidgeClassifierCV()"
]

output_file_names = [
    "SVCRBF",
    "SVCLINEAR",
    "SVCSIGMOID",
    "SVCPOLYD3",
    "SVCPOLYD4",
    "SVCPOLYD5",
    "DecisionTreeClassifier",
    "KNeighborsClassifier",
    "GaussianNB",
    "RandomForestClassifier",
    "AdaBoostClassifier",
    "QuadraticDiscriminantAnalysis",
    "LinearDiscriminantAnalysis",
    "ElasticNetCV",
    "LarsCV",
    "LassoCV",
    "LassoLarsCV",
    "LogisticRegressionCV",
    "MultiTaskElasticNetCV",
    "MultiTaskLassoCV",
    "OrthogonalMatchingPursuitCV",
    "RidgeClassifierCV"
]

classifiers = [
    SVC(kernel="rbf", probability=True),
    SVC(kernel='linear', probability=True),
    SVC(kernel='sigmoid', probability=True),
    SVC(kernel='poly', probability=True, degree=3),
    SVC(kernel='poly', probability=True, degree=4),
    SVC(kernel='poly', probability=True, degree=5),
    DecisionTreeClassifier(),
    KNeighborsClassifier(),
    GaussianNB(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    QuadraticDiscriminantAnalysis(),
    LinearDiscriminantAnalysis(),
    ElasticNetCV(max_iter=10000),
    LarsCV(),
    LassoCV(max_iter=10000),
    LassoLarsCV(),
    LogisticRegressionCV(), #17
    MultiTaskElasticNetCV(),
    MultiTaskLassoCV(),
    OrthogonalMatchingPursuitCV(),
    RidgeClassifierCV()
]
algorithm = 17
if len(sys.argv) > 1:
    algorithm = int(sys.argv[1])

name = names[algorithm]
clf = classifiers[algorithm]
output_file_name = output_file_names[algorithm]

t = time.time()
mlb = MultiLabelBinarizer()
random_state = np.random.RandomState(0)
binarized_labels = mlb.fit_transform(labels)
print "Fitting classifier " + name
classifier = OneVsRestClassifier(clf, n_jobs=-2)
ovrlsvc = clone(classifier)
print "Fitting Classifier"
ovrlsvc.fit(features, binarized_labels)
print "Creating Model"
model = SelectFromModel(ovrlsvc, prefit=True)
print "Transforming Features"
reduced_features = model.transform(features)
print "New shape of features : " + str(reduced_features.shape)
cross_validation_classifier = clone(classifier)
print "Running Cross Validation with transformed feature set"
predictions = cross_validation.cross_val_predict(cross_validation_classifier, reduced_features, binarized_labels, cv=10)
print "Classifier: " + name
print "Time passed: ", "{0:.1f}".format(time.time() - t), "sec"
print "Micro F1 score: ", f1_score(binarized_labels, predictions, average='micro')
print "Individual Class F1 score: ", f1_score(binarized_labels, predictions, average=None)


print "Fitting new classifier with full data and reduced features"
classifier.fit(reduced_features, binarized_labels)

print "Calculating Predictions..."

files = ["xaa", "xab", "xac", "xad", "xae", "xaf"]
header = True
for chunk in files:
    t = time.time()
    print "chunk: " + chunk
    test_df = pd.read_csv(data_root + chunk)
    # test_features = test_df['feature vector'].values
    test_features = np.array([convert_feature_to_vector(x) for x in test_df['feature vector']])
    test_features = normalize(np.append(normalize(test_features[:, :8192]), normalize(test_features[:, 8192:]), axis=1))
    reduced_test_features = model.transform(test_features)

    binarized_predicted_labels = classifier.predict(reduced_test_features)

    predicted_labels = mlb.inverse_transform(binarized_predicted_labels)

    print "Calculated Predictions... Time passed: ", "{0:.1f}".format(time.time() - t), "sec"
    print "Writing predictions to output file"
    test_data_frame = pd.read_csv(data_root + chunk)
    df = pd.DataFrame(columns=['business_id', 'labels'])

    for i in range(len(test_data_frame)):
        biz = test_data_frame.loc[i]['business']
        label = predicted_labels[i]
        label = str(label)[1:-1].replace(",", " ")
        df.loc[i] = [str(biz), label]

    if header:
        with open(submission_root + "reduced_" + output_file_name + ".csv", 'w') as f:
            df.to_csv(f, index=False, header=header)
        header = False
    else:
        with open(submission_root + "reduced_" + output_file_name + ".csv", 'a') as f:
            df.to_csv(f, index=False, header=header)
