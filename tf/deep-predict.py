import tensorflow as tf
import pickle
import numpy as np
import boto
import os

from boto.s3.key import Key
from alexnet import AlexNet

def predict(event, context):
    conn = boto.connect_s3("AKIAJRKPLMXU3JRGWYCA","LFfFCpaEsdCCz4KiEBKoTFS5ehXIcPjsPq3yqxjj")
    classifier = event['classifier']
    test_bucket = conn.get_bucket(event['bucket_from'])
    model_bucket = conn.get_bucket(event['model_bucket'])
    result_bucket = conn.get_bucket(event['result_bucket'])
    model_bucket_name = event['model_bucket_name']

    image_num = event["image_num"]
    
    print(str(image_num))
    train_key = test_bucket.get_key(str(image_num) + "-processed.npy")
    train_key.get_contents_to_filename('/tmp/ready_matrix.npy')

    if classifier == 'neural':
        key = model_bucket.get_key(model_bucket_name)
    elif classifier == 'knn':
        key = model_bucket.get_key('nn')
    elif classifier == 'decision_tree':
        key = model_bucket.get_key('dt')
    elif classifier == 'forest':
        key = model_bucket.get_key('forest')
    elif classifier == 'bagging':
        key = model_bucket.get_key('bagging')

    key_checkpoint = model_bucket.get_key(model_bucket_name + '-checkpoint')
    key_index = model_bucket.get_key(model_bucket_name + '-index')
    key_data = model_bucket.get_key(model_bucket_name + '-data')

    key.get_contents_to_filename('/tmp/model.meta')
    key_checkpoint.get_contents_to_filename('/tmp/checkpoint')
    key_index.get_contents_to_filename('/tmp/model.index')
    key_data.get_contents_to_filename('/tmp/model.data-00000-of-00001')

    print("preparation ready")

    with open("/tmp/ready_matrix.npy", "rb") as ready_matrix:
        X = np.load(ready_matrix)[np.newaxis, :]

    X_converted = X.astype(np.float)

    print("data loaded")
    train_layers = ['fc8', 'fc7', 'fc6']

    tf.reset_default_graph()

    # Create tf placeholders for X and y
    X_test = tf.placeholder(tf.float32, [1, 227, 227, int(event["num_channels"])], name="X_test")

    # for distinction between having 0.5 dropout in training and 0 dropout in test
    keep_prob = tf.placeholder(tf.float32)

    alexnet_pretrained = model_bucket.get_key('bvlc_alexnet.npy')
    alexnet_pretrained.get_contents_to_filename('/tmp/bvlc_alexnet.npy')

    # initialize AlexNet model
    model = AlexNet(X_test, keep_prob, event["num_classes"], train_layers, True)

    # Link variable to model output
    score = model.fc8

    # used for predict
    prediction = tf.nn.softmax(score, name="predict")

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # Load the pretrained weights into the non-trainable layer
        model.load_initial_weights(sess)
        os.remove('/tmp/bvlc_alexnet.npy')

        old_saver = tf.train.import_meta_graph('/tmp/model.meta')
        old_saver.restore(sess, '/tmp/model')

        predictions = sess.run(prediction, feed_dict={X_test: X_converted, keep_prob: 1})

    print("finished predicting")
    print(predictions)
    final_predictions = np.argmax(predictions, axis=1)
    if event["label_style"] == "array":
        reshaped_predictions = np.reshape(final_predictions, (int(event["yscale"]), int(event["xscale"])))
        np.savetxt('/tmp/results', reshaped_predictions, fmt = '%i')
    else:
        np.savetxt('/tmp/results', final_predictions)

    result_k = result_bucket.new_key(event['result_name'])

    result_k.set_contents_from_filename("/tmp/results")

    result_k.make_public()
    print(event["result_bucket"], event["result_name"])

    return 1