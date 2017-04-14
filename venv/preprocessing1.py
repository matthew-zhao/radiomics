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

        for image_name in paths:
            image = folder_path + "/" + image_name
            shape_dir = None

            f, metadata = client.files_download(image)
            data = metadata.content

            actual_name, extension = image_name.split(".")

            if !label_dict.has_key(actual_name):
                paths.remove(image_name)

        total_count = len(paths)
        counter = 1


        for key in paths: #key has extension

            if counter == total_count:
                last = True

            image_name = key
            image = folder_path + "/" + image_name
            shape_dir = None

            f, metadata = client.files_download(image)
            data = metadata.content

            actual_name, extension = image_name.split(".")

            label = label_dict[actual_name]

            if extension == "dcm":
                f2 = open("/tmp/response_content.dcm", "wb")
                f2.write(data)
                f2.close()

                f2 = open("/tmp/response_content.dcm", "rb")
                ds = dicom.read_file(f2)
                img = ds.pixel_array
                f2.close()
            else:
                img = scipy.array(Image.open(StringIO(data)))


            #lambdaclient.invoke()
            args = {"image": img, "label": label, "last": last, "filter_size": filter_size}
            invoke_response = lambda_client.invoke(FunctionName="preprocessing2", InvocationType='Event', Payload=json.dumps(args))

            counter += 1




