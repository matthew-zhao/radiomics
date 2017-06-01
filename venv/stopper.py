
import boto.ec2


def lambda_handler(event, context):
    instance_id = event['instance_id']

    conn = boto.ec2.connect_to_region("us-west-2", aws_access_key_id='AKIAJNKBSOROJ44BVHBQ', \
        aws_secret_access_key='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')
    
    id_list = [instance_id]
    conn.stop_instances(id_list)

    