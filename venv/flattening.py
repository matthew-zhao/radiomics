# flattens regular images and squishes

import dropbox
import base64
import scipy
import numpy as np
import boto3
import boto
from boto.s3.key import Key
import uuid
from scipy import stats
from scipy import ndimage
import pandas
from PIL import Image
from StringIO import StringIO

#s3_client = boto3.client('s3')

# this function assumes that we already have a folder with the images on dropbox
# then it will translate all jpeg images into numpy arrays and flatten them
# then it squishes them together, and uploads onto s3 for training

def retrieve(event):
    result_list = []
    dclient = dropbox.client.DropboxClient(event["auth_token"])
    client = dropbox.Dropbox(event["auth_token"])
    metadata = dclient.metadata(event["folder_name"])
    paths = []
    shape_dir = None
    for content in metadata['contents']:
        if content['is_dir'] == False:
            paths.append(content['path'])

    reader = pandas.read_csv("trainLabels.csv")
    # TODO: find way not to hardcode this
    images = list(reader.image)
    levels = list(reader.level)

    label_dict = dict(zip(images, levels))

    for image in paths:
        f, metadata = client.files_download(image)
        image_name = image.split('/')[-1]
        data = metadata.content

        actual_name, extension = image_name.split(".")

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
        
        # flatten
        flattened = img.flatten()
        label = label_dict[actual_name]
        final_arr = np.concatenate((flattened, [label]), axis=0)

        #squish
        result_list.append(final_arr) 

    return np.array(result_list)

def upload(event, context):
    number = event['number']
    conn = boto.connect_s3()
    b = conn.get_bucket('training-array')
    k = b.new_key('matrix' + str(number) + '.npy')

    training_matrix = retrieve(event)

    #download_path = '/tmp/{}{}'.format(uuid.uuid4(), key)
    upload_path = '/tmp/resized-{}'.format(k)

    #s3_client.download_file(bucket, key, download_path)
    np.save(upload_path, training_matrix)
    k.set_contents_from_filename(upload_path)

    return 0
