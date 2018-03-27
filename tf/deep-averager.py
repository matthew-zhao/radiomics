import tensorflow as tf
import pickle
import numpy as np
import boto
import argparse
import boto3
import json
import random
import os
import shutil

from boto.s3.key import Key
from alexnet import AlexNet

lambda_client = boto3.client('lambda')
def classify(event, context):
    clear_tmp_dir()
    conn = boto.connect_s3("AKIAJRKPLMXU3JRGWYCA","LFfFCpaEsdCCz4KiEBKoTFS5ehXIcPjsPq3yqxjj")
    model_bucket = conn.get_bucket('model-train')

    model_bucket_name = event["model_bucket_name"]

    client = boto3.client('sqs')

    queue_url1 = client.get_queue_url(QueueName=event['queue_name'])['QueueUrl']
    queue_url2 = client.get_queue_url(QueueName=event['queue_name1'])['QueueUrl']
    image_name_pairs = []

    num_machines = int(event['num_machines'])

    while num_machines > 0:
        if num_machines > 20:
            num_to_receive_1 = 10
            num_to_receive_2 = 10
        else:
            num_to_receive_1 = num_machines / 2
            num_to_receive_2 = num_machines - num_to_receive_1

        response = client.receive_message(
            QueueUrl=queue_url1,
            AttributeNames=['All'],
            MaxNumberOfMessages=num_to_receive_1,
            MessageAttributeNames=['All'],
            VisibilityTimeout=30,
            WaitTimeSeconds=20
        )

        response2 = client.receive_message(
            QueueUrl=queue_url2,
            AttributeNames=['All'],
            MaxNumberOfMessages=num_to_receive_2,
            MessageAttributeNames=['All'],
            VisibilityTimeout=30,
            WaitTimeSeconds=20
        )

        if 'Messages' not in response and 'Messages' not in response2:
            print("Done")
            return 1

        if 'Messages' in response:
            messages = response['Messages']

            for message in messages:
                receipt_handle = message['ReceiptHandle']

                image_name = message['Body']
                image_num = image_name.split("_")[0]
                image_name_pairs.append((image_name, image_num))

                client.delete_message(
                    QueueUrl=queue_url1,
                    ReceiptHandle=receipt_handle
                )

        if 'Messages' in response2:
            messages2 = response2['Messages']

            for message in messages2:
                receipt_handle = message['ReceiptHandle']

                image_name = message['Body']
                image_num = image_name.split("_")[0]
                image_name_pairs.append((image_name, image_num))

                client.delete_message(
                    QueueUrl=queue_url2,
                    ReceiptHandle=receipt_handle
                )
        num_machines -= num_to_receive_1 + num_to_receive_2

    # Choose which layers of AlexNet to train
    train_layers = ['fc8', 'fc7', 'fc6']
    dropout_rate = 0.5
    batch_size = 1

    print("finished reading from bucket")
    # averaging is done here
    if event["called_from"] == 'timer':
        #var_list = event["var_list"] # list of variable names
        # for varname in var_list:
        #     new_W1 = np.zeros((num_features, 216))
        #     new_b1 = np.zeros((1, 216))
        #     new_W2 = np.zeros((216, int(event['num_classes'])))
        #     new_b2 = np.zeros((1, int(event['num_classes'])))
        new_var_list = []
        var_map = {}
        num_successful = 0
        for machine_num in range(int(event["num_machines"])):
            i = str(machine_num)
            tf.reset_default_graph()
            with tf.Session() as sess:
                #sess.run(tf.global_variables_initializer())
                existing_model = model_bucket.get_key(model_bucket_name + i)
                existing_index = model_bucket.get_key(model_bucket_name + '-index' + i)
                existing_data = model_bucket.get_key(model_bucket_name + '-data' + i)
                existing_checkpoint = model_bucket.get_key(model_bucket_name + '-checkpoint' + i)

                if not (existing_model and existing_index and existing_data and existing_checkpoint):
                    continue

                print("machine" + i)
                existing_model.get_contents_to_filename('/tmp/model' + i + '.meta')
                existing_index.get_contents_to_filename('/tmp/model' + i + '.index')
                existing_data.get_contents_to_filename('/tmp/model' + i + '.data-00000-of-00001')
                existing_checkpoint.get_contents_to_filename('/tmp/checkpoint')
            
                saver = tf.train.import_meta_graph('/tmp/model' + i + '.meta')
                # with tf.variable_scope("params", reuse=None):
                #     W1 = tf.get_variable("W1", [int(event["num_features"]),216], initializer=tf.zeros_initializer())
                #     b1 = tf.get_variable("b1", [1, 216], initializer=tf.zeros_initializer())
                #     W2 = tf.get_variable("W2", [216, int(event["num_classes"])], initializer=tf.zeros_initializer())
                #     b2 = tf.get_variable("b2", [1, int(event["num_classes"])], initializer=tf.zeros_initializer())
                # parameters = {"W1": W1,
                #   "b1": b1,
                #   "W2": W2,
                #   "b2": b2}
                #saver = tf.train.Saver()
                #tf.variables_initializer([W1, b1, W2, b2])
                saver.restore(sess, '/tmp/model' + i)

                #if not new_var_list:
                var_list = []
                count = 0
                # TODO: assuming that every machine will return these trainable variables in order
                for v in tf.trainable_variables():
                    if v.name.split('/')[0] in train_layers:
                        var_list.append(v)
                        if v.name not in var_map:
                            var_map[v.name] = count
                        elif var_map[v.name] != count:
                            print("Order doesn't match up!!!")
                        count += 1

                actual_var_list = np.array(sess.run(var_list))
                if not len(new_var_list):
                    new_var_list = actual_var_list
                else:
                    new_var_list = np.sum((new_var_list, actual_var_list), axis=0)

                # delete keys
                model_bucket.delete_key(model_bucket_name + i)
                model_bucket.delete_key(model_bucket_name + '-index' + i)
                model_bucket.delete_key(model_bucket_name + '-data' + i)
                model_bucket.delete_key(model_bucket_name + '-checkpoint' + i)

                # delete files
                os.remove('/tmp/model' + i + '.meta')
                os.remove('/tmp/model' + i + '.index')
                os.remove('/tmp/model' + i + '.data-00000-of-00001')
                num_successful += 1

        new_var_list = new_var_list / num_successful

        print("finished concatenating")
        tf.reset_default_graph()
        with tf.Session() as sess:
            existing_model = model_bucket.get_key(model_bucket_name)
            existing_index = model_bucket.get_key(model_bucket_name + '-index')
            existing_data = model_bucket.get_key(model_bucket_name + '-data')
            existing_checkpoint = model_bucket.get_key(model_bucket_name + '-checkpoint')

            existing_model.get_contents_to_filename('/tmp/model.meta')
            existing_index.get_contents_to_filename('/tmp/model.index')
            existing_data.get_contents_to_filename('/tmp/model.data-00000-of-00001')
            existing_checkpoint.get_contents_to_filename('/tmp/checkpoint')

            saver = tf.train.import_meta_graph('/tmp/model.meta')
            saver.restore(sess, '/tmp/model')

            # delete more files
            os.remove('/tmp/model.meta')
            os.remove('/tmp/model.index')
            os.remove('/tmp/model.data-00000-of-00001')

            #graph = tf.get_default_graph()
            var_list = []
            with tf.variable_scope("params", reuse=True):
                # assign_ops = []
                # index = 0
                for v in tf.trainable_variables():
                    # uncomment below if we figure out how to save entire model
                    if v.name.split('/')[0] in train_layers:
                        p = tf.placeholder(tf.float32, shape=new_var_list[var_map[v.name]].shape)
                        assign_ops = tf.assign(v, p)
                        var_list.append(v)
                        sess.run(assign_ops, feed_dict={p: new_var_list[var_map[v.name]]})
                clear_tmp_dir()
                saver = tf.train.Saver(var_list)
                saver.save(sess, '/tmp/model')

        model_k = existing_model
        model_index = existing_index
        model_data = existing_data
        model_checkpoint = existing_checkpoint
    # creating the model is done here
    else:
        tf.reset_default_graph()
        model_k = model_bucket.new_key(model_bucket_name)
        model_index = model_bucket.new_key(model_bucket_name + '-index')
        model_data = model_bucket.new_key(model_bucket_name + '-data')
        model_checkpoint = model_bucket.new_key(model_bucket_name + '-checkpoint')

        # Create tf placeholders for X and y
        X_train = tf.placeholder(tf.float32, [batch_size, 227, 227, int(event["num_channels"])], name="X_train")
        y_train = tf.placeholder(tf.float32, [None, int(event["num_classes"])], name="y_train")

        # for distinction between having 0.5 dropout in training and 0 dropout in test
        keep_prob = tf.placeholder(tf.float32)

        alexnet_pretrained = model_bucket.get_key('bvlc_alexnet.npy')
        alexnet_pretrained.get_contents_to_filename('/tmp/bvlc_alexnet.npy')

        # initialize AlexNet model
        model = AlexNet(X_train, keep_prob, event["num_classes"], train_layers, True)

        # Link variable to model output
        score = model.fc8

        # List of trainable variables of the layers we want to train
        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

        # used for predict
        prediction = tf.nn.softmax(score, name="predict")

        # Op for calculating the loss
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y_train), name="loss")

        # Train op
        with tf.name_scope("train"):
            # Get gradients of all trainable variables
            #gradients = tf.gradients(cost, var_list)
            #gradients = list(zip(gradients, var_list))

            # Create optimizer and apply gradient descent to the trainable variables
            #optimizer = tf.train.AdamOptimizer(learning_rate)
            #train_op = optimizer.apply_gradients(grads_and_vars=gradients)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost, var_list=var_list, name="grad_descent")

        # Evaluation op
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y_train, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

        # tf.add_to_collection('loss', cost)
        # tf.add_to_collection('grad_descent', optimizer)
        # tf.add_to_collection('predict', prediction)
        # tf.add_to_collection('accuracy', accuracy)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            # Load the pretrained weights into the non-trainable layer
            model.load_initial_weights(sess)

            os.remove('/tmp/bvlc_alexnet.npy')

            with tf.variable_scope("params", reuse=True):
                # Initialize weights and biases of neural network
                saver = tf.train.Saver(var_list)
                saver.save(sess, '/tmp/model')

    print("done training")
    
    model_k.set_contents_from_filename("/tmp/model.meta")
    model_index.set_contents_from_filename("/tmp/model.index")
    model_data.set_contents_from_filename("/tmp/model.data-00000-of-00001")
    model_checkpoint.set_contents_from_filename("/tmp/checkpoint")

    model_k.make_public()
    model_index.make_public()
    model_data.make_public()
    model_checkpoint.make_public()

    args = {"bucket_from": event['bucket_from'], "bucket_from_labels": event['bucket_from_labels'], "model_bucket_name": model_bucket_name,
            "queue_name": event['queue_name'], "queue_name1": event["queue_name1"], "num_classes": event["num_classes"], "num_machines": event["num_machines"],
            "num_channels": event["num_channels"], "dropout_rate": dropout_rate, "train_layers": train_layers, "batch_size": batch_size}
    for i in range(int(event["num_machines"])):
        if i >= len(image_name_pairs):
            continue
        args["machine_num"] = i
        image_name, image_num = image_name_pairs[i]
        args['image_num'] = image_num
        args['image_name'] = image_name
        invoke_response = lambda_client.invoke(FunctionName="deep-checkpointer", InvocationType='Event', Payload=json.dumps(args))
    invoke_response = lambda_client.invoke(FunctionName="deep-timer", InvocationType='Event', Payload=json.dumps(args))
    return 0

def clear_tmp_dir():
    # clear tmp directory
    for file in os.listdir("/tmp"):
        filepath = os.path.join("/tmp", file)
        try:
            if os.path.isfile(filepath):
                os.unlink(filepath)
        except Exception as e:
            print(e)

