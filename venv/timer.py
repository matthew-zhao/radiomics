import time
import boto3
import json

lambda_client = boto3.client('lambda')

def invoke_lambda(event, context):

    is_train = event["is_train"]
    has_labels = event["has_labels"]

    # 4mins 50secs = 290 secs
    time.sleep(290)

    #invoke preprocessing3

    # supervised learning
    if is_train && has_labels:
        msg = {"bucket_from": "training-array", "bucket_from_labels": "training-labels", "is_train": is_train}
    # unsupervised learning
    elif is_train && !has_labels:
        msg = {"bucket_from": "training-array", "bucket_from_labels": "", "is_train": is_train}
    # prediction
    else:
        msg = {"bucket_from": "testing-array", "bucket_from_labels": "", "is_train": is_train}

    lambda_client.invoke(FunctionName="preprocessing3", InvocationType='Event', Payload=json.dumps(msg))
