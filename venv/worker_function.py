import boto
import paramiko
import boto3
from boto.s3.key import Key

def worker_handler(event, context):
    s3_client = boto3.client('s3')
    #Download private key file from secure S3 bucket
    conn = boto.connect_s3("AKIAIMQLHJNMP6DOUM4A","8dJAfPZlTjMR1SOcOetImclAmT+G02VkQiuHefdY")
    bucket_name = 's3-keys'
    b = conn.get_bucket(bucket_name)
    bucket_list = b.list()
    for l in bucket_list:
        if l.key[-4:] == ".pem":
            l.get_contents_to_filename("/tmp/" + str(l.key))

    k = paramiko.RSAKey.from_private_key_file("/tmp/ec2-lambda-compute-node.pem")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())


    classifier = event['classifier']
    bucket_training = event['bucket_training']
    bucket_labels = event['bucket_labels']


    host=event['IP']
    print "Connecting to " + host
    c.connect(hostname = host, username = "ec2-user", pkey = k )
    print "Connected to " + host

    command1 = "python" + " /usr/local/radiomics/" + classifier + ".py" + " -f " + bucket_training + " -b " + bucket_labels
    commands = [
        
        #"chmod 777 /usr/local/radiomics/" + classifier + ".py",
        command1
        
        ]
    for command in commands:
        print "Executing {}".format(command)
        stdin , stdout, stderr = c.exec_command(command)
        # print stdout.read()
        # print stderr.read()

    return
    {
        'message' : "Script execution completed. See Cloudwatch logs for complete output"
    }