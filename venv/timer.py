import time
import boto3
import json

lambda_client = boto3.client('lambda')

def invoke_lambda(event, context):

	is_train = event["is_train"]

	# 4mins 50secs = 290 secs
	time.sleep(290)

	#invoke preprocessing3
    msg = {"bucket_from": "training-array", "bucket_from_labels": "training-labels", "is_train": is_train}
    lambda_client.invoke(FunctionName="preprocessing3", InvocationType='Event', Payload=json.dumps(msg))
