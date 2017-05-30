# http://stackoverflow.com/questions/31714788/can-an-aws-lambda-function-call-another
# Lambda 1

import boto3
import json
import dropbox
import csv
import scipy

lambda_client = boto3.client('lambda')

def invoke_lambda(event, context):
    dclient = dropbox.client.DropboxClient(event["auth_token"])
    client = dropbox.Dropbox(event["auth_token"])
    metadata = dclient.metadata(event["folder_name"])
    is_train = event["is_train"]
    paths = []
    folder_name = event["folder_name"]
    auth_token = event["auth_token"]
    has_labels = event["has_labels"]
    if not has_labels:
        has_labels = ""
    shape_dir = None
    filter_size = 3
    last = False

    conn = boto.connect_s3("AKIAIMQLHJNMP6DOUM4A","8dJAfPZlTjMR1SOcOetImclAmT+G02VkQiuHefdY")
    b = conn.get_bucket(event['bucket_from'])

    csv_key = b.get_key('trainLabels.csv')
    csv_key.get_contents_to_filename("/tmp/trainLabels.csv")


    for content in metadata['contents']: 
        if content['is_dir'] == False:
            paths.append(content['path']) #adds files to paths
            #paths is list of paths to files, has extensions
            
    
    c_reader = csv.reader(open('/tmp/trainLabels.csv', 'r'), delimiter = ',')
    columns = list(zip(*c_reader))
    images = columns[0][1:]
    levels = list(map(int, columns[1][1:]))
    label_dict = dict(zip(images, levels))

    image_list = []

    #gets rid of image paths who do not have a label in label_dict
    for image_path in paths:
        image_name = image_path.split("/")[-1]
        actual_name, extension = image_name.split(".")

        if not label_dict.has_key(actual_name):
            paths.remove(image_paths)


    for key in paths: 
 

        image_path = key
        image_name = image_path.split("/")[-1]
        actual_name, extension = image_name.split(".")
        label = label_dict[actual_name]

        if has_labels:
            args = {"image_path": image_path, "image_name": actual_name, "label": label, "filter_size": filter_size, "auth_token": event["auth_token"], "is_train": event["is_train"], "has_labels": has_labels}
        else:
            args = {"image_path": image_path, "image_name": actual_name, "label": None, "filter_size": filter_size, "auth_token": event["auth_token"], "is_train": event["is_train"], "has_labels": has_labels}


        invoke_response = lambda_client.invoke(FunctionName="preprocessing2", InvocationType='Event', Payload=json.dumps(args))



    #invoke the timer function, which sleeps for 4:50mins before invoking preprocessing3
    args = {"is_train": is_train, "has_labels": has_labels}
    invokeTimer = lambda_client.invoke(FunctionName="timer", InvocationType='Event', Payload=json.dumps(args))



