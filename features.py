caffe_root = "/hpc/sw/caffe-2015.11.30-gpu/"
caffe_source = "/home/ml0501/erdi/caffe/"
data_root = "/home/ml0501/yelp/"


import os
import sys

import numpy as np

sys.path.insert(0, caffe_root + 'python')

import caffe

## Use GPU
caffe.set_device(0)
caffe.set_mode_gpu()


def extract_features(images, layer='fc7'):
    net = caffe.Net(caffe_source + 'models/bvlc_reference_caffenet/deploy.prototxt',
                    caffe_source + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                    caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(
        1))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB]]

    num_images = len(images)
    net.blobs['data'].reshape(num_images, 3, 227, 227)
    net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data', caffe.io.load_image(x)), images)
    out = net.forward()

    combined_features = np.append(net.blobs['fc7'].data, net.blobs['fc6'].data, axis=1)
    return combined_features / np.linalg.norm(combined_features)


# extract image features and save it to .h5

# Initialize files
import h5py

# f.close()
f = h5py.File(data_root + 'train_image_fc67features.h5', 'w')
filenames = f.create_dataset('photo_id', (0,), maxshape=(None,), dtype='|S54')
feature = f.create_dataset('feature', (0, 8192), maxshape=(None, 8192))
f.close()

import pandas as pd

train_photos = pd.read_csv(data_root + 'train_photo_to_biz_ids.csv')
train_folder = data_root + 'train_photos/'
train_images = [os.path.join(train_folder, str(x) + '.jpg') for x in train_photos['photo_id']]  # get full filename

num_train = len(train_images)
print "Number of training images: ", num_train
batch_size = 500

# Training Images
for i in range(0, num_train, batch_size):
    images = train_images[i: min(i + batch_size, num_train)]
    features = extract_features(images, layer='fc7')
    num_done = i + features.shape[0]
    f = h5py.File(data_root + 'train_image_fc67features.h5', 'r+')
    f['photo_id'].resize((num_done,))
    f['photo_id'][i: num_done] = np.array(images)
    f['feature'].resize((num_done, features.shape[1]))
    f['feature'][i: num_done, :] = features
    f.close()
    if num_done % 20000 == 0 or num_done == num_train:
        print "Train images processed: ", num_done


### Check the file content

f = h5py.File(data_root+'train_image_fc67features.h5','r')
print 'train_image_features.h5:'
for key in f.keys():
    print key, f[key].shape

print "\nA photo:", f['photo_id'][0]
print "Its feature vector (first 10-dim): ", f['feature'][0][0:10], " ..."
f.close()

f = h5py.File(data_root+'test_image_fc67features.h5','w')
filenames = f.create_dataset('photo_id',(0,), maxshape=(None,),dtype='|S54')
feature = f.create_dataset('feature',(0,4096), maxshape = (None,4096))
f.close()


test_photos = pd.read_csv(data_root+'test_photo_to_biz.csv')
test_folder = data_root+'test_photos/'
test_images = [os.path.join(test_folder, str(x)+'.jpg') for x in test_photos['photo_id'].unique()]
num_test = len(test_images)
print "Number of test images: ", num_test

# Test Images
for i in range(0, num_test, batch_size):
    images = test_images[i: min(i+batch_size, num_test)]
    features = extract_features(images, layer='fc7')
    num_done = i+features.shape[0]

    f= h5py.File(data_root+'test_image_fc67features.h5','r+')
    f['photo_id'].resize((num_done,))
    f['photo_id'][i: num_done] = np.array(images)
    f['feature'].resize((num_done,features.shape[1]))
    f['feature'][i: num_done, :] = features
    f.close()
    if num_done%20000==0 or num_done==num_test:
        print "Test images processed: ", num_done

### Check the file content
f = h5py.File(data_root+'test_image_fc67features.h5','r')
for key in f.keys():
    print key, f[key].shape
print "\nA photo:", f['photo_id'][0]
print "feature vector: (first 10-dim)", f['feature'][0][0:10], " ..."
f.close()