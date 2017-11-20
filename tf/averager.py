import tensorflow as tf
import pickle
import numpy as np
import boto
import argparse
import boto3
import json
import random

from boto.s3.key import Key

lambda_client = boto3.client('lambda')
def classify(event, context):
    conn = boto.connect_s3("AKIAIMQLHJNMP6DOUM4A","8dJAfPZlTjMR1SOcOetImclAmT+G02VkQiuHefdY")
    model_bucket = conn.get_bucket('models-train')

    model_bucket_name = event["model_bucket_name"]

    client = boto3.client('sqs')

    #num = random.randrange(1,3)
    #if num == 1:
    #    queue_url = client.get_queue_url(QueueName=event['queue_name'])['QueueUrl']
    #else:
    #    queue_url = client.get_queue_url(QueueName=event['queue_name1'])['QueueUrl']

    queue_url1 = client.get_queue_url(QueueName=event['queue_name'])['QueueUrl']
    queue_url2 = client.get_queue_url(QueueName=event['queue_name1'])['QueueUrl']
    image_name_pairs = []

    num_to_receive_1 = int(event['num_machines']) / 2
    num_to_receive_2 = int(event['num_machines']) - num_to_receive_1

    num_actually_received_1 = 0

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

    print("finished reading from bucket")
    # averaging is done here
    if event["called_from"] == 'timer':
        num_features = int(event["num_features"])
        # new_W1 = tf.get_variable("W1", [num_features,216], initializer=tf.zeros_initializer())
        # new_b1 = tf.get_variable("b1", [1, 216], initializer=tf.zeros_initializer())
        # new_W2 = tf.get_variable("W2", [216, event["num_classes"]], initializer=tf.zeros_initializer())
        # new_b2 = tf.get_variable("b2", [1, event["num_classes"]], initializer=tf.zeros_initializer())
        new_W1 = np.zeros((num_features, 216))
        new_b1 = np.zeros((1, 216))
        new_W2 = np.zeros((216, int(event['num_classes'])))
        new_b2 = np.zeros((1, int(event['num_classes'])))
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
            
                #saver = tf.train.import_meta_graph('/tmp/model' + i + '.meta')
                with tf.variable_scope("params", reuse=None):
                    W1 = tf.get_variable("W1", [int(event["num_features"]),216], initializer=tf.zeros_initializer())
                    b1 = tf.get_variable("b1", [1, 216], initializer=tf.zeros_initializer())
                    W2 = tf.get_variable("W2", [216, int(event["num_classes"])], initializer=tf.zeros_initializer())
                    b2 = tf.get_variable("b2", [1, int(event["num_classes"])], initializer=tf.zeros_initializer())
                parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
                saver = tf.train.Saver(parameters)
                tf.variables_initializer([W1, b1, W2, b2])
                saver.restore(sess, '/tmp/model' + i)
                # graph = tf.get_default_graph()
                # W1_save = graph.get_tensor_by_name("params/W1:0")
                # b1_save = graph.get_tensor_by_name("params/b1:0")
                # W2_save = graph.get_tensor_by_name("params/W2:0")
                # b2_save = graph.get_tensor_by_name("params/b2:0")

                # new_W1 = new_W1 + W1_save
                # new_b1 = new_b1 + b1_save
                # new_W2 = new_W2 + W2_save
                # new_b2 = new_b2 + b2_save
                new_W1 = new_W1 + W1.eval(sess)
                new_b1 = new_b1 + b1.eval(sess)
                new_W2 = new_W2 + W2.eval(sess)
                new_b2 = new_b2 + b2.eval(sess)
                num_successful += 1
        new_W1 = new_W1 / num_successful
        new_b1 = new_b1 / num_successful
        new_W2 = new_W2 / num_successful
        new_b2 = new_b2 / num_successful

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
            graph = tf.get_default_graph()
            with tf.variable_scope("params", reuse=True):
                # W1 = tf.get_variable("W1")
                # b1 = tf.get_variable("b1")
                # W2 = tf.get_variable("W2")
                # b2 = tf.get_variable("b2")
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
                assign_w1 = tf.assign(W1, new_W1)
                assign_b1 = tf.assign(b1, new_b1)
                assign_w2 = tf.assign(W2, new_W2)
                assign_b2 = tf.assign(b2, new_b2)

                sess.run([assign_w1, assign_b1, assign_w2, assign_b2])

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
        X_train = tf.placeholder(tf.float32, [None, int(event["num_features"])], name="X_train")
        y_train = tf.placeholder(tf.float32, [None, int(event["num_classes"])], name="y_train")
        num_features = int(event['num_features'])

        parameters = get_parameters(num_features, event)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Forward prop
        # Here, the notation is S1 = relu(X*W1 + b1) and S2 = relu(W2*S1 + b2)
        S1 = tf.nn.relu(tf.add(tf.matmul(X_train, W1), b1))
        S2 = tf.add(tf.matmul(S1, W2), b2)

        prediction = tf.nn.softmax(S2, name="predict")

        # Calculate cost/cross-entropy loss (our final layer is softmax)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=S2, labels=y_train), name="loss")

        # Backprop done by tensorflow (thank god :D)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost, name="grad_descent")

        tf.add_to_collection('loss', cost)
        tf.add_to_collection('grad_descent', optimizer)
        tf.add_to_collection('predict', prediction)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            sess.run(parameters)

            with tf.variable_scope("params", reuse=True):
                # Initialize weights and biases of neural network
                saver = tf.train.Saver()
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

    args = {"bucket_from": event['bucket_from'], "bucket_from_labels": event['bucket_from_labels'], "model_bucket_name": model_bucket_name, "num_items": event['num_items'],
            "queue_name": event['queue_name'], "queue_name1": event["queue_name1"], "num_classes": event["num_classes"], "num_features": num_features, "num_machines": event["num_machines"]}
    for i in range(int(event["num_machines"])):
        if i >= len(image_name_pairs):
            continue
        args["machine_num"] = i
        image_name, image_num = image_name_pairs[i]
        args['image_num'] = image_num
        args['image_name'] = image_name
        invoke_response = lambda_client.invoke(FunctionName="checkpointer", InvocationType='Event', Payload=json.dumps(args))
    invoke_response = lambda_client.invoke(FunctionName="timer", InvocationType='Event', Payload=json.dumps(args))
    return 0

def get_parameters(num_features, event):
    with tf.variable_scope("params"):
        W1 = tf.get_variable("W1", [num_features,216], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", [1, 216], initializer=tf.zeros_initializer())
        W2 = tf.get_variable("W2", [216, int(event["num_classes"])], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2", [1, int(event["num_classes"])], initializer=tf.zeros_initializer())
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
