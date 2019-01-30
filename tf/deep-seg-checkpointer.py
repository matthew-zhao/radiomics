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
from utils import stream_from_s3

from unet import get_train_layers
import memory_saving_gradients

# https://github.com/openai/gradient-checkpointing
# https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9
#tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory

lambda_client = boto3.client('lambda')

def classify(event, context):
    if "num_trainer" not in event:
        # averager doesn't pass num_trainer, so we know its just been averaged
        just_averaged = True
        num_trainer = 0
    else:
        just_averaged = False
        num_trainer = int(event["num_trainer"])
    # clear tmp directory
    for file in os.listdir("/tmp"):
        filepath = os.path.join("/tmp", file)
        try:
            if os.path.isfile(filepath):
                os.unlink(filepath)
        except Exception as e:
            print(e)
    filepath = None

    conn = boto.connect_s3("AKIAJRKPLMXU3JRGWYCA","LFfFCpaEsdCCz4KiEBKoTFS5ehXIcPjsPq3yqxjj")
    b = conn.get_bucket(event['bucket_from'])
    labels = conn.get_bucket(event['bucket_from_labels'], validate=False)
    model_bucket = conn.get_bucket('model-train')

    model_bucket_name = event["model_bucket_name"]

    #image_name = event["image_name"]
    #image_num = int(event["image_num"])
    #patch_num = int(event["start_patch_num"])
    i = str(event["machine_num"])

    client = boto3.client('sqs')

    queue_url1 = client.get_queue_url(QueueName=event['queue_name'])['QueueUrl']
    queue_url2 = client.get_queue_url(QueueName=event['queue_name1'])['QueueUrl']

    #num_to_receive_1 = 5
    #num_to_receive_2 = 5

    response = client.receive_message(
        QueueUrl=queue_url1,
        AttributeNames=['All'],
        MaxNumberOfMessages=1,
        MessageAttributeNames=['All'],
        VisibilityTimeout=30,
        WaitTimeSeconds=20
    )

    response2 = client.receive_message(
        QueueUrl=queue_url2,
        AttributeNames=['All'],
        MaxNumberOfMessages=1,
        MessageAttributeNames=['All'],
        VisibilityTimeout=30,
        WaitTimeSeconds=20
    )

    if 'Messages' in response and len(response['Messages']) > 0:
        messages = response['Messages']

        for message in messages:
            receipt_handle = message['ReceiptHandle']

            image_dict = json.loads(message['Body'])
            #print(image_dict)
            image_name = image_dict['actual_name']
            image_num = image_dict['image_num']
            patch_num = int(image_dict['patch_num'])
            num_patches = image_dict['num_patches']

            client.delete_message(
                QueueUrl=queue_url1,
                ReceiptHandle=receipt_handle
            )
    elif 'Messages' in response2 and len(response2['Messages']) > 0:
        messages2 = response2['Messages']

        for message in messages2:
            receipt_handle = message['ReceiptHandle']

            image_dict = json.loads(message['Body'])
            #print(image_dict)
            image_name = image_dict['actual_name']
            image_num = image_dict['image_num']
            patch_num = int(image_dict['patch_num'])
            num_patches = image_dict['num_patches']

            client.delete_message(
                QueueUrl=queue_url2,
                ReceiptHandle=receipt_handle
            )
    else:
        # failed to execute checkpointer because no images left in queue
        return 1

    print(image_name)

    # TODO: change this when we know how to calculate # of loops based off batch size of patches vs batch size of slices
    num_loops = 1
    # TODO: implement concatenation of multiple images together correctly
    X = np.zeros(1)
    y = np.zeros(1)
    # we'll assume that all images are at least (patch #, patch width, patch height) even if they are whole images
    for individual_image in range(num_loops):
        X_single = np.load(stream_from_s3(b, os.path.join(image_name, str(image_num) + '-processed.npy'), None))
        y_single = np.load(stream_from_s3(labels, os.path.join(image_name, str(image_num) + 'label-processed.npy'), None))

        if len(X_single.shape) < 4:
            # we need to add a channel dimension
            X_single = np.expand_dims(X_single, len(X_single.shape))
        if len(y_single.shape) < 4:
            y_single = np.expand_dims(y_single, len(y_single.shape))

        X_single = X_single[patch_num:patch_num+int(event["batch_size"]), :, :, :]
        y_single = y_single[patch_num:patch_num+int(event["batch_size"]), :, :, :]
        if len(X.shape) < 2:
            X = X_single
        else:
            X = np.concatenate((X, X_single))

        if len(y.shape) < 2:
            y = y_single
        else:
            y = np.concatenate((y, y_single))
    b = None
    y_single = None
    X_single = None

    print("finished reading from bucket")

    if just_averaged:
        averager_model = model_bucket.get_key(model_bucket_name)
        averager_index = model_bucket.get_key(model_bucket_name + '-index')
        averager_data = model_bucket.get_key(model_bucket_name + '-data')
        averager_checkpoint = model_bucket.get_key(model_bucket_name + '-checkpoint')
    else:
        averager_model = model_bucket.get_key(model_bucket_name + i)
        averager_index = model_bucket.get_key(model_bucket_name + '-index' + i)
        averager_data = model_bucket.get_key(model_bucket_name + '-data' + i)
        averager_checkpoint = model_bucket.get_key(model_bucket_name + '-checkpoint' + i)

    averager_model.get_contents_to_filename('/tmp/model.meta')
    averager_index.get_contents_to_filename('/tmp/model.index')
    averager_data.get_contents_to_filename('/tmp/model.data-00000-of-00001')
    averager_checkpoint.get_contents_to_filename('/tmp/checkpoint')

    print("About to train")
    tf.reset_default_graph()

    with tf.Session() as sess:
        #init = tf.global_variables_initializer()
        #sess.run(init)

        # Load the pretrained weights into the non-trainable layer
        # model.load_initial_weights(sess)

        print("training from averager model")

        saver = tf.train.import_meta_graph('/tmp/model.meta')
        #saver = tf.train.Saver(var_list)
        saver.restore(sess, '/tmp/model')
        saver = None

        graph = tf.get_default_graph()

        X_train = graph.get_tensor_by_name("X_train:0")
        y_train = graph.get_tensor_by_name("y_train:0")
        mode = graph.get_tensor_by_name("mode:0")
        graph = None

        #cost = tf.get_collection('loss')[0]
        optimizer = tf.get_collection('grad_descent')[0]

        #minibatch_size = 32
        #m = X.shape[0]
        with tf.variable_scope("params", reuse=True):
            optimized = sess.run([optimizer], feed_dict={X_train: X, y_train: y, mode: True})

        os.remove('/tmp/model.meta')
        os.remove('/tmp/model.index')
        os.remove('/tmp/model.data-00000-of-00001')
        os.remove('/tmp/checkpoint')

        with tf.variable_scope("params", reuse=True):
            train_layers = get_train_layers()
            var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
            saver = tf.train.Saver(var_list)
            saver.save(sess, '/tmp/model' + i)

    print("done training")

    if just_averaged:
        model_k = model_bucket.new_key(model_bucket_name + i)
        model_index = model_bucket.new_key(model_bucket_name + '-index' + i)
        model_data = model_bucket.new_key(model_bucket_name + '-data' + i)
        model_checkpoint = model_bucket.new_key(model_bucket_name + '-checkpoint' + i)
    else:
        model_k = averager_model
        model_index = averager_index
        model_data = averager_data
        model_checkpoint = averager_checkpoint
    
    model_k.set_contents_from_filename("/tmp/model" + i + ".meta")
    model_index.set_contents_from_filename("/tmp/model" + i + ".index")
    model_data.set_contents_from_filename("/tmp/model" + i + ".data-00000-of-00001")
    model_checkpoint.set_contents_from_filename("/tmp/checkpoint")

    model_k.make_public()
    model_index.make_public()
    model_data.make_public()
    model_checkpoint.make_public()

    # TODO: Right now, we are hardcoding the # of checkpointers to 10, but we'll probably want to change this later
    if num_trainer < 9:
        args = {"bucket_from": event['bucket_from'], "bucket_from_labels": event['bucket_from_labels'], "model_bucket_name": model_bucket_name,
                "queue_name": event['queue_name'], "queue_name1": event["queue_name1"], "num_classes": event["num_classes"], "num_machines": event["num_machines"],
                "num_channels": event["num_channels"], "dropout_rate": event["dropout_rate"], "train_layers": event["train_layers"], "batch_size": event["batch_size"], "epoch": event["epoch"],
                "num_epochs": event["num_epochs"], "machine_num": i, "num_trainer": num_trainer + 1}
        invoke_response = lambda_client.invoke(FunctionName="deep-seg-checkpointer", InvocationType='Event', Payload=json.dumps(args))
    return 0
