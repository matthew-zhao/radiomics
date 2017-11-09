#from sklearn.neural_network import MLPClassifier
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
# Uses MLP Neural Net classifier to train a model
def classify(event, context):
    conn = boto.connect_s3("AKIAIMQLHJNMP6DOUM4A","8dJAfPZlTjMR1SOcOetImclAmT+G02VkQiuHefdY")
    b = conn.get_bucket(event['bucket_from'])
    labels = conn.get_bucket(event['bucket_from_labels'], validate=False)
    #image_name = event["image_name"]
    model_bucket = conn.get_bucket('models-train')

    model_bucket_name = event["model_bucket_name"]
    #image_num = event["image_num"]
    #num = int(image_num)

    #X_key = b.get_key('ready_matrix.npy')
    #Y_key = labels.get_key('ready_labels.npy')
    #bucket_list = b.list()

    client = boto3.client('sqs')

    num = random.randrange(1,3)
    if num == 1:
        queue_url = client.get_queue_url(QueueName=event['queue_name'])['QueueUrl']
    else:
        queue_url = client.get_queue_url(QueueName=event['queue_name1'])['QueueUrl']

    response = client.receive_message(
        QueueUrl=queue_url,
        AttributeNames=['All'],
        MaxNumberOfMessages=1,
        MessageAttributeNames=['All'],
        VisibilityTimeout=2,
        WaitTimeSeconds=20
    )

    #print(response)

    if 'Messages' not in response:
        # remove flag key
        #b.delete_key("called")

        # TODO: Uncomment delete queue for production
        # client.delete_queue(QueueUrl=queue_url)

        # it's over
        print("Done")
        return 1

    message = response['Messages'][0]
    receipt_handle = message['ReceiptHandle']

    image_name = message['Body']
    image_num = image_name.split("_")[0]
    #num = int(image_num)

    client.delete_message(
        QueueUrl=queue_url,
        ReceiptHandle=receipt_handle
    )

    print("Received and deleted message")

    X_key = b.get_key(image_name + '-processed.npy')
    X_key.get_contents_to_filename('/tmp/ready_matrix.npy')
    Y_key = labels.get_key(image_num + 'label-processed.npy')
    Y_key.get_contents_to_filename('/tmp/ready_labels.npy')

    existing_model = model_bucket.get_key(model_bucket_name)
    existing_index = model_bucket.get_key(model_bucket_name + '-index')
    existing_data = model_bucket.get_key(model_bucket_name + '-data')
    existing_checkpoint = model_bucket.get_key(model_bucket_name + '-checkpoint')

    print("finished reading from bucket")

    #X_key.get_contents_to_filename('/tmp/ready_matrix.npy')
    #Y_key.get_contents_to_filename('/tmp/ready_labels.npy')

    with open("/tmp/ready_matrix.npy", "rb") as ready_matrix:
        X = np.load(ready_matrix)

    with open("/tmp/ready_labels.npy", "rb") as ready_labels:
        y = np.load(ready_labels)
    

    print("About to train")
    with tf.Session() as sess:
        if existing_model:
            print("training from existing model")
            existing_model.get_contents_to_filename('/tmp/model.meta')
            existing_index.get_contents_to_filename('/tmp/model.index')
            existing_data.get_contents_to_filename('/tmp/model.data-00000-of-00001')
            existing_checkpoint.get_contents_to_filename('/tmp/checkpoint')
            # with open("/tmp/key", "rb") as keyfile:
            #     contents = keyfile.read()
            #     clf = pickle.loads(contents)
            
            saver = tf.train.import_meta_graph('/tmp/model.meta')
            saver.restore(sess, '/tmp/model')


            graph = tf.get_default_graph()
            X_train = graph.get_tensor_by_name("X_train:0")
            y_train = graph.get_tensor_by_name("y_train:0")

            #cost = graph.get_tensor_by_name("loss:0")
            #optimizer = graph.get_tensor_by_name("grad_descent:0")
            #cost = graph.get_operation_by_name("loss")
            #optimizer = graph.get_operation_by_name("grad_descent")
            cost = tf.get_collection('loss')[0]
            optimizer = tf.get_collection('grad_descent')[0]
            print(cost)
            print(optimizer)
            # for op in graph.get_operations():
            #     print(str(op.name))

            model_k = existing_model
            model_index = existing_index
            model_data = existing_data
            model_checkpoint = existing_checkpoint
        else:
            model_k = model_bucket.new_key(model_bucket_name)
            model_index = model_bucket.new_key(model_bucket_name + '-index')
            model_data = model_bucket.new_key(model_bucket_name + '-data')
            model_checkpoint = model_bucket.new_key(model_bucket_name + '-checkpoint')
            #clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(216,), random_state=1, warm_start=True, max_iter=1000)

            # Create tf placeholders for X and y
            X_train = tf.placeholder(tf.float32, [None, X.shape[1]], name="X_train")
            y_train = tf.placeholder(tf.float32, [None, y.shape[1]], name="y_train")
            num_features = X.shape[1]

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

            init = tf.global_variables_initializer()
            sess.run(init)

            # Initialize weights and biases of neural network
            saver = tf.train.Saver()

            tf.add_to_collection('loss', cost)
            tf.add_to_collection('grad_descent', optimizer)
            tf.add_to_collection('predict', prediction)

        # TODO: this may not be true if things are not one-hot encoded
        #clf.classes_ = [0, 1]

        minibatch_size = 32
        m = X.shape[0]
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

            #parameters = sess.run(parameters)

        saver.save(sess, '/tmp/model')


    #clf.partial_fit(X, y, classes=event["num_classes"])

    print("done training")


    #s = pickle.dumps(clf)

    #model_k = model_bucket.new_key(event['model_name'])

    #model_k = model_bucket.new_key('nm')

    #with open("/tmp/model", "wb") as model:
    #    model.write(s)
    
    model_k.set_contents_from_filename("/tmp/model.meta")
    model_index.set_contents_from_filename("/tmp/model.index")
    model_data.set_contents_from_filename("/tmp/model.data-00000-of-00001")
    model_checkpoint.set_contents_from_filename("/tmp/checkpoint")

    model_k.make_public()
    model_index.make_public()
    model_data.make_public()
    model_checkpoint.make_public()

    args = {"bucket_from": event['bucket_from'], "bucket_from_labels": event['bucket_from_labels'], "model_bucket_name": model_bucket_name, "image_num": image_num, "num_items": event['num_items'],
            "queue_name": event['queue_name'], "queue_name1": event["queue_name1"], "num_classes": event["num_classes"]}
    invoke_response = lambda_client.invoke(FunctionName="neuralnet_tf", InvocationType='Event', Payload=json.dumps(args))
    return 0

def get_parameters(num_features, event):
    W1 = tf.get_variable("W1", [num_features,216], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [1, 216], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [216, event["num_classes"]], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [1, event["num_classes"]], initializer=tf.zeros_initializer())
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
