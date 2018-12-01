import tensorflow as tf
import pickle
import numpy as np
import boto
import os

from boto.s3.key import Key
from utils import check_message, receive_and_delete_message

lambda_client = boto3.client('lambda')

def predict(event, context):
    conn = boto.connect_s3("AKIAJRKPLMXU3JRGWYCA","LFfFCpaEsdCCz4KiEBKoTFS5ehXIcPjsPq3yqxjj")
    classifier = event['classifier']
    test_bucket = conn.get_bucket(event['bucket_from'])
    model_bucket = conn.get_bucket(event['model_bucket'])
    result_bucket = conn.get_bucket(event['result_bucket'])
    model_bucket_name = event['model_bucket_name']

    result_name = event["result_name"]
    train_key = test_bucket.get_key(result_name + "-processed.npy")
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
        X = np.load(ready_matrix)

    X_converted = X.astype(np.float)

    print("data loaded")

    tf.reset_default_graph()

    # Create tf placeholders for X and y
    X_test = tf.placeholder(tf.float32, [None, event["xscale"], event["yscale"]], name="X_test")

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # Load the pretrained weights into the non-trainable layer
        # model.load_initial_weights(sess)

        old_saver = tf.train.import_meta_graph('/tmp/model.meta')
        old_saver.restore(sess, '/tmp/model')

        graph = tf.get_default_graph()
        X_train = graph.get_tensor_by_name("X_train:0")

        prediction_func = tf.get_collection('predict')[0]

        predictions = sess.run(prediction, feed_dict={X_test: X_converted, keep_prob: 1})

    print("finished predicting")
    print(predictions)
    if event["label_style"] == "array":
        reshaped_predictions = np.reshape(final_predictions, (int(event["yscale"]), int(event["xscale"])))
        np.savetxt('/tmp/results', reshaped_predictions, fmt = '%i')
    else:
        np.savetxt('/tmp/results', final_predictions)

    result_k = result_bucket.new_key(result_name)

    result_k.set_contents_from_filename("/tmp/results")

    result_k.make_public()
    print(event["result_bucket"], result_name)

    # this will only be in lambda payload if we are analyzing dev/test sets during "train" mode
    if "final_labels_bucket" in event:
        predict_done_queue_url = client.get_queue_url(QueueName='predict-' + model_bucket_name + '.fifo')['QueueUrl']
        msg = receive_and_delete_message()
        if msg == None:
            # call analyze_results
            args = {"label_style": event["label_style"], "result_bucket": event["result_bucket"], "labels_key_name": event["labels_key_name"], "labels_bucket": event["final_labels_bucket"], "images_bucket": event["bucket_from"]}
            if "IOU_threshold" in event:
                args["IOU_threshold"] = event["IOU_threshold"]
            invoke_response = lambda_client.invoke(FunctionName="analyze_results", InvocationType='Event', Payload=json.dump(args))
    else:
        # delete image from bucket_from
        test_bucket.delete_key(result_name + "-processed.npy")


    return 1