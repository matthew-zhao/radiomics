# preprocessing3
import numpy as np
import boto
import boto3
import json
from boto.s3.key import Key
import json
import boto3
import random

# Function to be called by lambda2 when all images are done being preprocessed
# to squish them together
lambda_client = boto3.client('lambda')

def squish(event, context):
    conn = boto.connect_s3("AKIAIMQLHJNMP6DOUM4A","8dJAfPZlTjMR1SOcOetImclAmT+G02VkQiuHefdY")
    b = conn.get_bucket(event['bucket_from'])
    has_labels = event["has_labels"]
    is_train = event["is_train"]
    image_name = event["image_name"]
    feature = event["feature"]
    model_bucket_name = event["model_bucket_name"]
    image_num = event["image_num"]
    queue_name = event["queue_name"]
    queue_name1 = event["queue_name1"]

    bucket_list = b.list()

    if has_labels:
        labels = conn.get_bucket(event['bucket_from_labels'])

    i = 0
    training_arr = None
    label_arr = None

    for l in bucket_list:
        # Save content of file into tempfile on lambda
        # TODO: this step may cause issues b/c of .npy data lost?
        if l.key == "matrix" + image_name + '_' + feature + ".npy":
            l.get_contents_to_filename("/tmp/" + str(l.key))

            with open("/tmp/" + str(l.key), "rb") as npy:
                training_arr = np.load(npy)

            if has_labels:
                gen_key = "matrix" + image_name + ".npy"
                label_matrix = labels.get_key(gen_key)
                label_matrix.get_contents_to_filename("/tmp/labels-" + gen_key)
                with open("/tmp/labels-" + gen_key, "rb") as npy_label:
                    label_arr = np.load(npy_label)
        i += 1

    X_converted = training_arr.astype(np.float16)
    X_train = X_converted / 255

    if has_labels:
        y_converted = label_arr.astype(np.float16)
        targets = y_converted.reshape(-1)
        y_train = np.eye(5)[targets.astype('int8')]

    # Create new buckets for the array and its corresponding labels
    if is_train:
        b2 = conn.get_bucket('training-arrayfinal')
    else:
        b2 = conn.get_bucket('testing-arrayfinal')

    k = b2.new_key(str(image_num) + "_" + feature + "-processed.npy")

    # Save the numpy arrays to temp .npy files on lambda
    upload_path = '/tmp/resized-matrix.npy'
    np.save(upload_path, X_train)

    # Take the tempfile that has the concatanated final array and set the contents
    # of the new bucket's key
    k.set_contents_from_filename(upload_path)

    # make the file public so it can be accessed publicly
    # using a URL like http://s3.amazonaws.com/bucket_name/key
    k.make_public()

    if has_labels:
        labels2 = conn.get_bucket('training-labelsfinal')
        k2 = labels2.new_key(str(image_num) + "label-processed.npy")

        upload_path_labels = '/tmp/resized-labels.npy'
        np.save(upload_path_labels, y_train)
        #np.save(upload_path_labels, label_arr)

        k2.set_contents_from_filename(upload_path_labels)
        k2.make_public()



    #if not training, each preprocessing3 calls a predict
    if not is_train:
        args = {"classifier": "neural", "bucket_from": "testing-arrayfinal", "model_bucket": "models-train", "model_bucket_name": model_bucket_name, "result_bucket": "result-labels", "num_items": i, "image_name": image_name, "feature": feature, "result_name": image_name + str(image_num),
            "image_num": str(image_num)}
        invoke_response = lambda_client.invoke(FunctionName="predict", InvocationType='Event', Payload=json.dumps(args))

    else: 
        sqs = boto3.client('sqs')
        num = random.randrange(1,3)
        if num == 1:
            queue_url = sqs.get_queue_url(QueueName=queue_name)
        else:
            queue_url = sqs.get_queue_url(QueueName=queue_name1)
        response = sqs.send_message(QueueUrl=queue_url['QueueUrl'], MessageBody=str(image_num) + '_' + feature, MessageDeduplicationId="deduplicationId" + str(image_num) + '_' + feature, MessageGroupId="groupId")
        print(response)

        b3 = conn.get_bucket("training-arrayfinal")
        called = b3.get_key("called")

        #only invoke neuralnet.py once, by the first preprocessing3 to finish
        if called is None:
            print("Neuralnet invoked from " + str(image_num))
            with open("/tmp/called", "wb") as flag:
                flag.write("True")
            k = b3.new_key("called")
            k.set_contents_from_filename("/tmp/called")
            k.make_public()

            args = {"bucket_from": "training-arrayfinal", "bucket_from_labels" : "training-labelsfinal", "model_bucket_name": model_bucket_name, "image_num": str(image_num), "num_items": i, "image_name": image_name, "queue_name": queue_name, "queue_name1": queue_name1, "num_classes": event["num_classes"]}
            invoke_response = lambda_client.invoke(FunctionName="neuralnet_checkpoint", InvocationType='Event', Payload=json.dumps(args))


    return 0