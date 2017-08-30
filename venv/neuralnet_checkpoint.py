from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
import boto
import argparse
import boto3
import json

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

    queue_url = client.get_queue_url(QueueName=event['queue_name'])['QueueUrl']

    response = client.receive_message(
        QueueUrl=queue_url,
        AttributeNames=['All'],
        MaxNumberOfMessages=1,
        MessageAttributeNames=['All'],
        VisibilityTimeout=2,
        WaitTimeSeconds=0
    )

    #print(response)

    if 'Messages' not in response:
        # remove flag key
        b.delete_key("called")

        # TODO: Uncomment delete queue for production
        # client.delete_queue(QueueUrl=queue_url)

        # it's over
        print("Done")
        return 1

    message = response['Messages'][0]
    receipt_handle = message['ReceiptHandle']

    image_num = message['Body']
    num = int(image_num)

    client.delete_message(
        QueueUrl=queue_url,
        ReceiptHandle=receipt_handle
    )

    print("Received and deleted message")

    X_key = b.get_key(image_num + '-processed.npy')
    X_key.get_contents_to_filename('/tmp/ready_matrix.npy')
    Y_key = labels.get_key(image_num + 'label-processed.npy')
    Y_key.get_contents_to_filename('/tmp/ready_labels.npy')

    existing_model = model_bucket.get_key(model_bucket_name)

    print("finished reading from bucket")

    #X_key.get_contents_to_filename('/tmp/ready_matrix.npy')
    #Y_key.get_contents_to_filename('/tmp/ready_labels.npy')

    with open("/tmp/ready_matrix.npy", "rb") as ready_matrix:
        X = np.load(ready_matrix)

    with open("/tmp/ready_labels.npy", "rb") as ready_labels:
        y = np.load(ready_labels)

    print("About to train")

    if existing_model:
        existing_model.get_contents_to_filename('/tmp/key')
        with open("/tmp/key", "rb") as keyfile:
            contents = keyfile.read()
            clf = pickle.loads(contents)
        model_k = existing_model
    else:
        clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(216,), random_state=1, warm_start=True, max_iter=20)

        model_k = model_bucket.new_key(model_bucket_name)

    # TODO: this may not be true if things are not one-hot encoded
    #clf.classes_ = [0, 1]

    clf.partial_fit(X, y, classes=[0, 1, 2, 3, 4])

    print("done training")

    s = pickle.dumps(clf)

    #model_k = model_bucket.new_key(event['model_name'])

    #model_k = model_bucket.new_key('nm')

    with open("/tmp/model", "wb") as model:
        model.write(s)
    
    model_k.set_contents_from_filename("/tmp/model")

    model_k.make_public()

    args = {"bucket_from": event['bucket_from'], "bucket_from_labels": event['bucket_from_labels'], "model_bucket_name": model_bucket_name, "image_num": image_num, "num_items": event['num_items'],
            "queue_name": event['queue_name']}
    invoke_response = lambda_client.invoke(FunctionName="neuralnet_checkpoint", InvocationType='Event', Payload=json.dumps(args))
    return 0

