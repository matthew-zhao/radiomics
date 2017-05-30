import boto3
import json
import boto.ec2
import time

def trigger_handler(event, context):
    #Get IP addresses of EC2 instances
    conn = boto.ec2.connect_to_region("us-west-2", aws_access_key_id='AKIAJNKBSOROJ44BVHBQ', \
        aws_secret_access_key='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')

    #images = conn.get_all_images()
    id = 'i-085e912626ec91bd6'
    instance = conn.start_instances(id)[0]

    while instance.update() != "running":
        time.sleep(5)

    host = instance.ip_address

    classifier = event['classifier']
    bucket_training = event['bucket_training']
    bucket_labels = event['bucket_labels']

    #Invoke worker function for each IP address
    client = boto3.client('lambda')

    print "Invoking worker_function on " + host
    args = {"IP": host, "classifier": classifier, "bucket_training": bucket_training, "bucket_labels": bucket_labels}
    invokeResponse=client.invoke(
        FunctionName='worker_function',
        InvocationType='Event',
        LogType='Tail',
        Payload=json.dumps(args)
    )
    print invokeResponse

    return{
        'message' : "Trigger function finished"
    }