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

from StringIO import StringIO
from boto.s3.key import Key
from utils import check_message, cleanup_s3_bucket, stream_from_s3

import unet
import memory_saving_gradients

# https://github.com/openai/gradient-checkpointing
# https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9
# tf.__dict__["gradients"] = memory_saving_gradients.gradients_collection

lambda_client = boto3.client('lambda')

def classify(event, context):
    clear_tmp_dir()
    conn = boto.connect_s3("AKIAJRKPLMXU3JRGWYCA","LFfFCpaEsdCCz4KiEBKoTFS5ehXIcPjsPq3yqxjj")
    model_bucket = conn.get_bucket('model-train')

    model_bucket_name = event["model_bucket_name"]

    client = boto3.client('sqs')

    queue_url1 = client.get_queue_url(QueueName=event['queue_name'])['QueueUrl']
    queue_url2 = client.get_queue_url(QueueName=event['queue_name1'])['QueueUrl']
    #image_name_pairs = []

    num_epochs = int(event["num_epochs"])
    epoch_num = int(event["epoch"])

    num_machines = int(event['num_machines'])

    b = conn.get_bucket(event['bucket_from'])
    labels = conn.get_bucket(event['bucket_from_labels'], validate=False)

    if not check_message(client, queue_url1) and not check_message(client, queue_url2):
        if epoch_num < num_epochs:
            all_keys = []
            # this is inefficient, will be much better to keep track of all the names and store it somewhere
            for item in b.list():
                if item.key.endswith(".npy"):
                    all_keys.append(item.key[:-14])
            # shuffle the order of the image names so we get some variability
            random.shuffle(all_keys)
            # put all the image names back into the queue
            for image_key in all_keys:
                print(image_key)
                num_patch_key = b.get_key(image_key + "-numpatches.txt")
                num_patches = int(StringIO(num_patch_key.get_contents_as_string()))
                actual_name, image_num = image_key.rsplit(os.sep, 1)
                for i in range(num_patches):
                    num = random.randrange(1,3)
                    if num == 1:
                        queue_url = client.get_queue_url(QueueName=event["queue_name"])
                    else:
                        queue_url = client.get_queue_url(QueueName=event["queue_name1"])
                    json_msg = json.dumps({"actual_name": str(actual_name), "image_num": str(image_num), "patch_num": str(i), "num_patches": str(num_patches)})
                    response = client.send_message(QueueUrl=queue_url['QueueUrl'], MessageBody=json_msg, MessageDeduplicationId="deduplicationId" + str(image_num), MessageGroupId="groupId")
            epoch_num += 1
        else:
            print("Done")
            msg_queue_url = client.get_queue_url(QueueName='called-' + event["queue_name"])['QueueUrl']

            #retained_messages = []
            message = check_message(client, msg_queue_url)
            while message is not None:
                # if there are dev/test sets to test on, then we'll execute from preprocess2
                # otherwise we are officially finished
                # TODO CHECK: not sure 0 is needed here
                receipt_handle = message['ReceiptHandle']
                # TODO CHECK: convert dictionary to json?
                lambda_payload = message['Body']
                sns_client = boto3.client('sns', region_name='us-west-2')
                # deep-seg-preprocess2 is subscribed to this SNS topic, so it'll receive the 
                # payload from preprocessing1 test/dev set if exists and run with it
                response = sns_client.publish(
                    TargetArn='arn:aws:sns:us-west-2:314083849514:finished_training',
                    Message=lambda_payload,
                    MessageStructure='json'
                )
                # delete message from queue
                client.delete_message(
                    QueueUrl=msg_queue_url,
                    ReceiptHandle=receipt_handle
                )
                message = check_message(client, msg_queue_url)
            cleanup_s3_bucket(b)
            cleanup_s3_bucket(labels)
            return 0

    #num_msgs = num_machines * 
    #while num_machines > 0:
    #    if num_machines > 20:
    #        num_to_receive_1 = 10
    #        num_to_receive_2 = 10
    #    else:
    #        num_to_receive_1 = num_machines / 2
    #        num_to_receive_2 = num_machines - num_to_receive_1

        # response = client.receive_message(
        #     QueueUrl=queue_url1,
        #     AttributeNames=['All'],
        #     MaxNumberOfMessages=num_to_receive_1,
        #     MessageAttributeNames=['All'],
        #     VisibilityTimeout=30,
        #     WaitTimeSeconds=20
        # )

        # if 'Messages' in response:
        #     messages = response['Messages']

        #     for message in messages:
        #         receipt_handle = message['ReceiptHandle']

        #         image_dict = json.loads(message['Body'])
        #         #print(image_dict)
        #         actual_name = image_dict['actual_name']
        #         image_num = image_dict['image_num']
        #         patch_num = image_dict['patch_num']
        #         num_patches = image_dict['num_patches']
        #         image_name_pairs.append((actual_name, image_num, patch_num))

        #         client.delete_message(
        #             QueueUrl=queue_url1,
        #             ReceiptHandle=receipt_handle
        #         )

        # response2 = client.receive_message(
        #     QueueUrl=queue_url2,
        #     AttributeNames=['All'],
        #     MaxNumberOfMessages=num_to_receive_2,
        #     MessageAttributeNames=['All'],
        #     VisibilityTimeout=30,
        #     WaitTimeSeconds=20
        # )

        # if 'Messages' in response2:
        #     messages2 = response2['Messages']

        #     for message in messages2:
        #         receipt_handle = message['ReceiptHandle']

        #         image_dict = json.loads(message['Body'])
        #         #print(image_dict)
        #         actual_name = image_dict['actual_name']
        #         image_num = image_dict['image_num']
        #         patch_num = image_dict['patch_num']
        #         num_patches = image_dict['num_patches']
        #         image_name_pairs.append((actual_name, image_num, patch_num))

        #         client.delete_message(
        #             QueueUrl=queue_url2,
        #             ReceiptHandle=receipt_handle
        #         )
        # num_machines -= num_to_receive_1 + num_to_receive_2

    dropout_rate = 0.5
    batch_size = 3
    train_layers = unet.get_train_layers()

    print("finished reading from bucket")
    # averaging is done here
    if event["called_from"] == 'timer':
        new_var_list = np.array([])
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
                saver.restore(sess, '/tmp/model' + i)

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
                        #print(var_map)
                        p = tf.placeholder(tf.float32, shape=new_var_list[var_map[v.name]].shape)
                        assign_ops = tf.assign(v, p)
                        var_list.append(v)
                        sess.run(assign_ops, feed_dict={p: new_var_list[var_map[v.name]]})
                clear_tmp_dir()
                #saver = tf.train.Saver(var_list)
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
        # batch size by 2d slice
        X_train = tf.placeholder(tf.float32, [None, None, None, int(event["num_channels"])], name="X_train")
        y_train = tf.placeholder(tf.float32, [None, None, None, int(event["num_channels"])], name="y_train")
        mode = tf.placeholder(tf.bool, name='mode')

        # for distinction between having 0.5 dropout in training and 0 dropout in test
        keep_prob = tf.placeholder(tf.float32)

        model_prediction = unet.create_unet(X_train, mode)

        # Link variable to model output
        #score = model.fc8

        # List of trainable variables of the layers we want to train
        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
        print([v.name for v in var_list])

        # used for predict
        #prediction = tf.nn.softmax(score, name="predict")

        # Train op
        # Get gradients of all trainable variables
        # Op for calculating the loss
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y_train), name="loss")
        cost = unet.loss_dice(model_prediction, y_train)
        raw_gradients = memory_saving_gradients.gradients_memory(cost, var_list)
        gradients = list(zip(raw_gradients, var_list))

        with tf.name_scope("train"):
            # Create optimizer and apply gradient descent to the trainable variables
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            train_op = optimizer.apply_gradients(grads_and_vars=gradients, name="grad_descent")
            #optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(-cost, name="grad_descent")

        # Evaluation op
        #with tf.name_scope("accuracy"):
        #    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y_train, 1))
        #    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

        tf.add_to_collection('loss', cost)
        tf.add_to_collection('grad_descent', train_op)
        #tf.add_to_collection('predict', prediction)
        #tf.add_to_collection('accuracy', accuracy)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            print([n.name for n in tf.get_default_graph().as_graph_def().node])
            sess.run(init)

            #os.remove('/tmp/bvlc_alexnet.npy')

            with tf.variable_scope("params", reuse=True):
                # Initialize weights and biases of neural network
                saver = tf.train.Saver()
                saver.save(sess, '/tmp/model')

    print("done training")
    
    #batch = unet.init_batch(batch_size, root=root)

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
            "num_channels": event["num_channels"], "dropout_rate": dropout_rate, "train_layers": train_layers, "batch_size": batch_size, "epoch": epoch_num,
            "num_epochs": num_epochs}
    for i in range(int(event["num_machines"])):
        args["machine_num"] = i
        invoke_response = lambda_client.invoke(FunctionName="deep-seg-checkpointer", InvocationType='Event', Payload=json.dumps(args))
    invoke_response = lambda_client.invoke(FunctionName="deep-seg-timer", InvocationType='Event', Payload=json.dumps(args))
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

