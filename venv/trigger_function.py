import boto3
import json

def trigger_handler(event, context):
    #Get IP addresses of EC2 instances
    client = boto3.client('ec2')
    instDict=client.describe_instances(
            Filters=[{'Name':'tag:Environment','Values':['Dev']}]
        )


    classifier = event['classifier']
    bucket_training = event['bucket_training']
    bucket_labels = event['bucket_labels']

    hostList=[]
    for r in instDict['Reservations']:
        for inst in r['Instances']:
            hostList.append(inst['PublicIpAddress'])

    #Invoke worker function for each IP address
    client = boto3.client('lambda')
    for host in hostList:
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