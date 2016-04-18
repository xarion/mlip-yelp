import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer, normalize

data_root = '/Users/erdicalli/dev/workspace/yelp/data/'
submission_root = '/Users/erdicalli/dev/workspace/yelp/submission/'
print "reading training data"
train_df = pd.read_csv(data_root + "train_merged.csv")


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def convert_label_to_array(str_label):
    str_label = str_label[1:-1]

    str_label = str_label.split(',')
    return [int(x) for x in str_label if len(x) > 0]


def convert_feature_to_vector(str_feature):
    str_feature = str_feature[1:-1]
    str_feature = str_feature.split(',')
    return [np.float32(x) for x in str_feature]


print "converting features and labels from raw data"
labels = np.array([convert_label_to_array(y) for y in train_df['label']])
features = np.array([convert_feature_to_vector(x) for x in train_df['feature vector']])
features = features[:, :-9]
print "normalizing features"
features = normalize(np.append(normalize(features[:, :8192]), normalize(features[:, 8192:]), axis=1))

mlb = MultiLabelBinarizer()
print "defining network"
input_size = features.shape[1]
hidden_nodes = 15
batch_size = None
num_labels = 9
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, input_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size, input_size))

    # Variables.
    weights_1 = tf.Variable(
        tf.truncated_normal([input_size, hidden_nodes]))
    biases_1 = tf.Variable(tf.zeros([hidden_nodes]))

    weights_2 = tf.Variable(tf.truncated_normal([hidden_nodes, num_labels]))
    biases_2 = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
    relus = tf.nn.relu(logits_1)
    logits = tf.matmul(relus, weights_2) + biases_2
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(
        tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1), weights_2) + biases_2)
output_file_name = "tf.out"
print "initializing tensorflow session"

binarized_labels = mlb.fit_transform(labels)
feed_dict = {tf_train_dataset: features, tf_train_labels: binarized_labels}
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print "training"
    previous = 100000
    lozz = 10000
    predictions = None
    step = 0
    mf1 = 0
    while True:
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        step += 1
        if step % 10 == 0:
            mf1 = f1_score(binarized_labels, predictions > 0.012, average='micro')
            print "Micro F1 score: ", mf1
            print "Individual Class F1 score: ", f1_score(binarized_labels, predictions > 0.0115, average=None)
            if mf1 > 0.85:
                break
    # accuracy(binarized_predicted_labels, labels)

    print "training done going over chunks for testing"
    files = ["xaa", "xab", "xac", "xad", "xae", "xaf"]
    header = True

    for chunk in files:
        print "chunk: " + chunk
        t = time.time()
        print "reading csv"
        test_df = pd.read_csv(data_root + chunk)
        # test_features = test_df['feature vector'].values
        print "converting feature vector"
        test_features = np.array([convert_feature_to_vector(x) for x in test_df['feature vector']])
        print "normalizing feature vector"
        test_features = normalize(
            np.append(normalize(test_features[:, :8192]), normalize(test_features[:, 8192:]), axis=1))
        print "creating feed"
        feed_dict = {tf_test_dataset: test_features}
        print "getting results"
        binarized_predicted_labels = session.run([test_prediction], feed_dict=feed_dict)
        print "results ready"
        predicted_labels = mlb.inverse_transform(binarized_predicted_labels[0] > 0.0115)
        print "chunk done"
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
