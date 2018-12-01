import boto3
import boto
from StringIO import StringIO

# method will only look at whether there is one message in queue
def check_message(client, queue_url):
    response = client.receive_message(
        QueueUrl=queue_url,
        AttributeNames=['All'],
        MaxNumberOfMessages=1,
        MessageAttributeNames=['All'],
        VisibilityTimeout=30,
        WaitTimeSeconds=20
    )

    if 'Messages' not in response or len(response['Messages']) <= 0:
        return None
    return response['Messages'][0]

# returns response
def receive_and_delete_message(client, queue_url):
    message = check_message(client, queue_url)
    if not msg:
        return None
    receipt_handle = message['ReceiptHandle']
    # TODO CHECK: convert dictionary to json?
    msg_body = message['Body']
    # delete message from queue
    client.delete_message(
        QueueUrl=queue_url,
        ReceiptHandle=receipt_handle
    )
    return msg_body

def cleanup_s3_bucket(bucket):
    # want to delete all items in bucket so we can reuse intermediary s3 buckets
    # and so that we don't accumulate items, which incur cost
    all_keys = []
    for key in bucket.list():
        all_keys.append(key.key)
    bucket.delete_keys(all_keys, quiet=True)

def stream_from_s3(bucket=None, key_name=None, key=None):
    if not key:
        key = bucket.get_key(key_name)
    data_string = StringIO(key.get_contents_as_string())
    return data_string
