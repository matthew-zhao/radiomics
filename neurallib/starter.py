# this lambda function starts everything by taking
# user input and putting everything into lambda queue

import numpy as np
import boto
import boto3
import json

lambda_client = boto3.client('lambda')

def start(event, context):
    sqs = boto3.client('sqs')
    queue_name = event["username"] + '.fifo'
    response = sqs.create_queue(
        QueueName=queue_name,
        Attributes={
            'FifoQueue': 'true',
            'MessageRetentionPeriod': '300',
            'ContentBasedDeduplication': 'true'
        }
    )
    
    queue_url = sqs.get_queue_url(QueueName=queue_name)
    response = sqs.send_message(QueueUrl=queue_url['QueueUrl'], MessageBody=str(image_num), MessageDeduplicationId="deduplicationId" + str(image_num), MessageGroupId="groupId")

    # invoke lambda consumer

