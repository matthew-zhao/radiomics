import time
import boto3
import json

lambda_client = boto3.client('lambda')

def invoke_lambda(event, context):

    # 4mins 50secs = 290 secs
    time.sleep(290)

    #invoke preprocessing3

    args = {"bucket_from": event['bucket_from'], "bucket_from_labels": event['bucket_from_labels'], "model_bucket_name": event["model_bucket_name"], "image_num": event["image_num"],
            "queue_name": event['queue_name'], "queue_name1": event["queue_name1"], "num_classes": event["num_classes"], "called_from": "timer", "num_machines": event["num_machines"],
            "num_channels": event["num_channels"]}

    lambda_client.invoke(FunctionName="deep-seg-averager", InvocationType='Event', Payload=json.dumps(args))
