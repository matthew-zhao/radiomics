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
    conn = boto.connect_s3("AKIAJRKPLMXU3JRGWYCA","LFfFCpaEsdCCz4KiEBKoTFS5ehXIcPjsPq3yqxjj")
    b = conn.get_bucket(event['bucket_from'])
    labels = conn.get_bucket(event['bucket_from_labels'], validate=False)
    model_bucket = conn.get_bucket('models-train')

    model_bucket_name = event["model_bucket_name"]

    image_name = event["image_name"]
    image_num = event["image_num"]
    i = str(event["machine_num"])

    print(image_name)

    X_key = b.get_key(image_name + '-processed.npy')
    X_key.get_contents_to_filename('/tmp/ready_matrix.npy')
    Y_key = labels.get_key(image_num + 'label-processed.npy')
    Y_key.get_contents_to_filename('/tmp/ready_labels.npy')

    with open("/tmp/ready_matrix.npy", "rb") as ready_matrix:
        X = np.load(ready_matrix)

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

    averager_model.get_contents_to_filename('/tmp/model.meta')
    averager_index.get_contents_to_filename('/tmp/model.index')
    averager_data.get_contents_to_filename('/tmp/model.data-00000-of-00001')
    averager_checkpoint.get_contents_to_filename('/tmp/checkpoint')

    model_k = model_bucket.new_key(model_bucket_name + i)
    model_index = model_bucket.new_key(model_bucket_name + '-index' + i)
    model_data = model_bucket.new_key(model_bucket_name + '-data' + i)
    model_checkpoint = model_bucket.new_key(model_bucket_name + '-checkpoint' + i)


    print("About to train")
    with tf.Session() as sess:
        print("training from averager model")
        
        saver = tf.train.import_meta_graph('/tmp/model.meta')
        saver.restore(sess, '/tmp/model')

        graph = tf.get_default_graph()
        X_train = graph.get_tensor_by_name("X_train:0")
        y_train = graph.get_tensor_by_name("y_train:0")

        cost = tf.get_collection('loss')[0]
        optimizer = tf.get_collection('grad_descent')[0]
        print(cost)
        print(optimizer)

        minibatch_size = 32
        m = X.shape[0]
        with tf.variable_scope("params", reuse=True):
            for iteration in range(50):
                iteration_cost = 0.
                num_minibatches = int(m / minibatch_size)
                minibatches = []

                # generate random minibatches
                permutation = list(np.random.permutation(m))
                shuffled_X = X[permutation, :]
                shuffled_Y = y[permutation, :]

                for k in range(num_minibatches):
                    minibatch_X = shuffled_X[k * minibatch_size : k * minibatch_size + minibatch_size, :]
                    minibatch_Y = shuffled_Y[k * minibatch_size : k * minibatch_size + minibatch_size, :]
                    minibatches.append((minibatch_X, minibatch_Y))

                if m % minibatch_size != 0:
                    minibatch_X = shuffled_X[num_minibatches*minibatch_size : m, :]
                    minibatch_Y = shuffled_Y[num_minibatches*minibatch_size : m, :]
                    minibatches.append((minibatch_X, minibatch_Y))

                for minibatch in minibatches:
                    _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X_train: minibatch[0], y_train: minibatch[1]})
                    iteration_cost += minibatch_cost / num_minibatches

                if iteration % 10 == 0:
                    print ("Cost after iteration %i: %f" % (iteration, iteration_cost))

        os.remove('/tmp/model.meta')
        os.remove('/tmp/model.index')
        os.remove('/tmp/model.data-00000-of-00001')
        os.remove('/tmp/checkpoint')

        with tf.variable_scope("params", reuse=True):
            W1 = None
            b1 = None
            W2 = None
            b2 = None
            for var in tf.global_variables():
                if var.op.name == "params/W1":
                    W1 = var
                elif var.op.name == "params/b1":
                    b1 = var
                elif var.op.name == "params/W2":
                    W2 = var
                elif var.op.name == "params/b2":
                    b2 = var
            parameters = {"W1": W1,
                          "b1": b1,
                          "W2": W2,
                          "b2": b2}
            saver = tf.train.Saver(parameters)
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
