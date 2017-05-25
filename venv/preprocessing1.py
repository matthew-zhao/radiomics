# http://stackoverflow.com/questions/31714788/can-an-aws-lambda-function-call-another
# Lambda 1

#this is a comment

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
    paths = []
    folder_name = event["folder_name"]
    auth_token = event["auth_token"]
    has_labels = event["has_labels"]
    shape_dir = None
    filter_size = 3
    last = False

    for content in metadata['contents']: 
        if content['is_dir'] == False:
            paths.append(content['path']) #adds files to paths
            #paths is list of paths to files, has extensions
            
    if has_labels:
        c_reader = csv.reader(open('trainLabels.csv', 'r'), delimiter = ',')
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



        total_count = len(paths)
        counter = 1

        for key in paths: #
            if counter == total_count:
                last = True

            image_path = key
            image_name = image_path.split("/")[-1]
            actual_name, extension = image_name.split(".")
            label = label_dict[actual_name]


            #lambdaclient.invoke()
            args = {"image_path": image_path, "image_name": actual_name, "label": label, "last": last, "filter_size": filter_size, "auth_token": event["auth_token"]}
            invoke_response = lambda_client.invoke(FunctionName="preprocessing2", InvocationType='Event', Payload=json.dumps(args))

            counter += 1




