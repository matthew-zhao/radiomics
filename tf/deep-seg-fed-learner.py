import tensorflow as tf
import pickle
import numpy as np
import boto
import argparse
import boto3
import json
import random
import os

from boto.s3.key import Key

def classify(event, context):
    # clear tmp directory
    for file in os.listdir("/tmp"):
        filepath = os.path.join("/tmp", file)
        try:
            if os.path.isfile(filepath):
                os.unlink(filepath)
        except Exception as e:
            print(e)

    conn = boto.connect_s3("AKIAJRKPLMXU3JRGWYCA","LFfFCpaEsdCCz4KiEBKoTFS5ehXIcPjsPq3yqxjj")
    b = conn.get_bucket(event['bucket_from'])
    labels = conn.get_bucket(event['bucket_from_labels'], validate=False)
    model_bucket = conn.get_bucket('model-train')

    model_bucket_name = event["model_bucket_name"]

    image_name = event["image_name"]
    image_num = event["image_num"]
    i = str(event["machine_num"])

    print(image_name)

    X_key = b.get_key(image_name + '-processed.npy')
    X_key.get_contents_to_filename('/tmp/ready_matrix.npy')
    Y_key = labels.get_key(image_num + 'label-processed.npy')
    Y_key.get_contents_to_filename('/tmp/ready_labels.npy')

    # TODO: Added an extra dimension at the beginning, might change
    with open("/tmp/ready_matrix.npy", "rb") as ready_matrix:
        X = np.load(ready_matrix)[np.newaxis, :]

    with open("/tmp/ready_labels.npy", "rb") as ready_labels:
        y = np.load(ready_labels)

    # once done with matrix and labels, remove them to save space
    os.remove("/tmp/ready_matrix.npy")
    os.remove("/tmp/ready_labels.npy")

    print("finished reading from bucket")

    averager_model = model_bucket.get_key(model_bucket_name)
    averager_index = model_bucket.get_key(model_bucket_name + '-index')
    averager_data = model_bucket.get_key(model_bucket_name + '-data')
    averager_checkpoint = model_bucket.get_key(model_bucket_name + '-checkpoint')

    model_k = model_bucket.new_key(model_bucket_name + i)
    model_index = model_bucket.new_key(model_bucket_name + '-index' + i)
    model_data = model_bucket.new_key(model_bucket_name + '-data' + i)
    model_checkpoint = model_bucket.new_key(model_bucket_name + '-checkpoint' + i)


    print("About to train")
    tf.reset_default_graph()

    # # Create tf placeholders for X and y
    # X_train = tf.placeholder(tf.float32, [int(event["batch_size"]), 572, 572, int(event["num_channels"])], name="X_train")
    # y_train = tf.placeholder(tf.float32, [None, int(event["num_classes"])], name="y_train")

    # # for distinction between having 0.5 dropout in training and 0 dropout in test
    # keep_prob = tf.placeholder(tf.float32)

    # # Link variable to model output
    # score = model.fc8

    # # used for predict
    # prediction = tf.nn.softmax(score, name="predict")

    # # Op for calculating the loss
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y_train), name="loss")

    # # Train op
    # with tf.name_scope("train"):
    #     # Get gradients of all trainable variables
    #     #gradients = tf.gradients(cost, var_list)
    #     #gradients = list(zip(gradients, var_list))

    #     # Create optimizer and apply gradient descent to the trainable variables
    #     #optimizer = tf.train.AdamOptimizer(learning_rate)
    #     #train_op = optimizer.apply_gradients(grads_and_vars=gradients)
    #     optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(-cost, name="grad_descent")

    # # Evaluation op
    # with tf.name_scope("accuracy"):
    #     correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y_train, 1))
    #     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    train_layers = unet.get_train_layers()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # Load the pretrained weights into the non-trainable layer
        model.load_initial_weights(sess)

        print("training from averager model")

        averager_model.get_contents_to_filename('/tmp/model.meta')
        averager_index.get_contents_to_filename('/tmp/model.index')
        averager_data.get_contents_to_filename('/tmp/model.data-00000-of-00001')
        averager_checkpoint.get_contents_to_filename('/tmp/checkpoint')

        saver = tf.train.import_meta_graph('/tmp/model.meta')
        #saver = tf.train.Saver(var_list)
        saver.restore(sess, '/tmp/model')

        #graph = tf.get_default_graph()

        X_train = graph.get_tensor_by_name("X_train:0")
        y_train = graph.get_tensor_by_name("y_train:0")

        cost = tf.get_collection('loss')[0]
        optimizer = tf.get_collection('grad_descent')[0]

        #minibatch_size = 32
        #m = X.shape[0]
        with tf.variable_scope("params", reuse=True):
            _, cost = sess.run([optimizer, cost], feed_dict={X_train: X, y_train: y, keep_prob: event["dropout_rate"]})

        os.remove('/tmp/model.meta')
        os.remove('/tmp/model.index')
        os.remove('/tmp/model.data-00000-of-00001')
        os.remove('/tmp/checkpoint')

        with tf.variable_scope("params", reuse=True):
            var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
            saver = tf.train.Saver(var_list)
            saver.save(sess, '/tmp/model' + i)

    print("done training")
    
    model_k.set_contents_from_filename("/tmp/model" + i + ".meta")
    model_index.set_contents_from_filename("/tmp/model" + i + ".index")
    model_data.set_contents_from_filename("/tmp/model" + i + ".data-00000-of-00001")
    model_checkpoint.set_contents_from_filename("/tmp/checkpoint")

    model_k.make_public()
    model_index.make_public()
    model_data.make_public()
    model_checkpoint.make_public()
    return 0
