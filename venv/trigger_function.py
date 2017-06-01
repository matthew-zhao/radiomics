import boto3
import json
import boto.ec2
import time

def trigger_handler(event, context):
    #Get IP addresses of EC2 instances
    conn = boto.ec2.connect_to_region("us-west-2", aws_access_key_id='AKIAJNKBSOROJ44BVHBQ', \
        aws_secret_access_key='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')

    # First check if any ec2 instances are available (in the stopped mode)
    # and just use the first one available, if there exists one
    reservations = conn.get_all_reservations(filters={'instance-state-name': 'stopped'})
    if len(reservations) > 0:
        instance_id = reservations[0].instances[0].id
        instance = conn.start_instances(instance_id)[0]
    else:
        #images = conn.get_all_images()
        #img = images[0]

        img = 'ami-5cc6a43c'
        reservation = conn.run_instances(img, instance_type='r4.xlarge', subnet_id='subnet-aab984dd', 
                                 security_group_ids=['sg-70df8f0b'], key_name='ec2-lambda-compute-node')
        instance = reservation.instances[0]
        instance_id = instance.id

    #id = 'i-085e912626ec91bd6'

    while instance.update() != "running":
        time.sleep(5)

    host = instance.ip_address

    classifier = event['classifier']
    bucket_training = event['bucket_training']
    bucket_labels = event['bucket_labels']

    #Invoke worker function for each IP address
    client = boto3.client('lambda')

    print "Invoking worker_function on " + host
    args = {"IP": host, "classifier": classifier, "bucket_training": bucket_training, 
            "bucket_labels": bucket_labels, "instance_id": instance_id}
            
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