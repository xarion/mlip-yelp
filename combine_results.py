import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

output_file_names = [
    "OrthogonalMatchingPursuitCV",
    "SVCLINEAR",  # 1
    "LassoLarsCV",  # 16
    "LogisticRegressionCV",  # 17
    "RidgeClassifierCV"  # 21
]

combination = [4, 2, 2, 2, 4, 2, 2, 1, 3]

import pandas as pd

data_root = "/Users/erdicalli/dev/workspace/yelp/submission/submissions/"

mlb = MultiLabelBinarizer()
total_labels = list()
for idx, file in enumerate(output_file_names):
    f = pd.read_csv(data_root + "merged_" + output_file_names[idx] + ".csv")
    labels = np.array([list(y.replace(" ", "")) for y in f["labels"]])
    total_labels.append(mlb.fit_transform(labels))

result_labels = np.ndarray(shape=(10000, 9))

for label_id, algorithm in enumerate(combination):
    result_labels[:, label_id] = total_labels[algorithm][:, label_id]

labels = mlb.inverse_transform(result_labels)

test_data_frame = pd.read_csv(data_root + "merged_" + output_file_names[4] + ".csv")
df = pd.DataFrame(columns=['business_id', 'labels'])

for i in range(len(test_data_frame)):
    biz = test_data_frame.loc[i]['business_id']
    label = labels[i]
    label = str(label)[1:-1].replace(",", " ")
    df.loc[i] = [str(biz), label]

with open(data_root + "combined_results.csv", 'w') as f:
    df.to_csv(f, index=False)
