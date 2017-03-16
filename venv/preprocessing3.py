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

        result_list.append((img, label_dict[actual_name]))

    return result_list

def preprocess(event, context):
    #bucket = event['Records'][0]['s3']['bucket']['name']
    #key = event['Records'][0]['s3']['object']['key']
    number = event['number']
    conn = boto.connect_s3()
    b = conn.get_bucket('training-array')
    k = b.new_key('matrix' + str(number) + '.npy')
    k2 = b.new_key('matrix' + str(number) + '-label.npy')

    np_arr_tup_list = retrieve(event)
    values, labels = zip(*map(lambda arr_arg: analyze(arr_arg, event), np_arr_tup_list))

    concat = np.concatenate(values, axis=1)
    concat_label = np.concatenate(labels, axis=1)

    #download_path = '/tmp/{}{}'.format(uuid.uuid4(), key)
    upload_path = '/tmp/resized-{}'.format(k)
    upload_path_labels = '/tmp/resized-{}'.format(k2)

    #s3_client.download_file(bucket, key, download_path)
    np.save(upload_path, concat)
    np.save(upload_path_labels, concat_label)
    k.set_contents_from_filename(upload_path)
    k2.set_contents_from_filename(upload_path_labels)
    #s3_client.upload_file(upload_path, '{}resized'.format(bucket), key)
    return 0

def analyze(arr_arg, event):
    result = None
    arr = np.array(arr_arg[0])
    label = arr_arg[1]
    h = scipy.histogram(arr, 256)
    dim = len(arr.shape)

    filter_size = int(event['filter_size'])

    mean = scipy.ndimage.generic_filter(arr, scipy.mean, size = filter_size, mode = 'constant')
    
    #median = scipy.ndimage.generic_filter(arr, scipy.median, size = filter_size, mode = 'constant')
    
    #uniformity = lambda arr : scipy.sum(np.square((scipy.histogram(arr, 256)[0])))
    #uniform = scipy.ndimage.generic_filter(arr, uniformity, size = filter_size, mode = 'constant')

    #def entropy(arr):
    #    log_ret = np.log2(scipy.histogram(arr, 256)[0])
    #    log_ret[log_ret==-np.inf]=0
    #    return np.dot(scipy.histogram(arr, 256)[0], log_ret)  
    #entropy_val = scipy.ndimage.generic_filter(arr, entropy, size = filter_size, mode = 'constant')
    
    maximum = scipy.ndimage.generic_filter(arr, np.amax, size = filter_size, mode = 'constant')
    
    minimum = scipy.ndimage.generic_filter(arr, np.amin, size = filter_size, mode = 'constant')
    

    def energy(arr):
        return np.sum(np.square(arr))
    energy_val = scipy.ndimage.generic_filter(arr, energy, size = filter_size, mode = 'constant')

    #def rms(arr):
    #    return math.sqrt(energy(arr) / arr.size)
    #rms_val = scipy.ndimage.generic_filter(arr, rms, size = filter_size, mode = 'constant')

    def std(arr):
        return np.std(arr, ddof=1)
    std_val = scipy.ndimage.generic_filter(arr, std, size = filter_size, mode = 'constant')

    if dim == 3:
        total = []
        for i in range(arr.shape[0] - filter_size + 1):
            for j in range(arr.shape[1] - filter_size + 1):
                for k in range(arr.shape[2] - filter_size + 1):
                    row = arr[i:i+filter_size, j:j+filter_size, k:k+filter_size].flatten()
                    row = np.append(row, mean[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                    #row = np.append(row, median[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                    #row = np.append(row, skew[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                    #row = np.append(row, kurtosis[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                    #row = np.append(row, uniform[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                    #row = np.append(row, entropy_val[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                    row = np.append(row, maximum[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                    row = np.append(row, minimum[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                    row = np.append(row, energy_val[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                    #row = np.append(row, rms_val[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                    row = np.append(row, std_val[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                total.append(row)
        result = np.array(total)
        x = arr.shape[0] - filter_size + 1
        y = arr.shape[1] - filter_size + 1
        z = arr.shape[2] - filter_size + 1
        total_size = x * y * z
    elif dim == 2:
        total = []
        for i in range(arr.shape[0] - filter_size + 1):
            for j in range(arr.shape[1] - filter_size + 1):
                row = arr[i:i+filter_size, j:j+filter_size, k:k+filter_size].flatten()
                row = np.append(row, mean[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                #row = np.append(row, median[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                #row = np.append(row, skew[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                #row = np.append(row, kurtosis[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                #row = np.append(row, uniform[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                #row = np.append(row, entropy_val[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                row = np.append(row, maximum[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                row = np.append(row, minimum[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                row = np.append(row, energy_val[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                #row = np.append(row, rms_val[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                row = np.append(row, std_val[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
            total.append(row)
        result = np.array(total)
        x = arr.shape[0] - filter_size + 1
        y = arr.shape[1] - filter_size + 1
        total_size = x * y
    label_arr = np.full((1,total_size), label)
    return result, label_arr