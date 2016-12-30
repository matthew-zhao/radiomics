import gspread
import dropbox
import base64
import scipy
import numpy as np
from scipy import stats
from scipy import ndimage
from oauth2client.service_account import ServiceAccountCredentials

def retrieve(event):
    client = dropbox.Dropbox(event["auth_token"])
    f, metadata = client.files_download(event["from_url"])
    f_shape, metadata_shape = client.files_download(event["from_url"].strip(".txt") +"_shape.txt")
    data = metadata.content

    metadata_shape = metadata_shape.content
    args = metadata_shape.split("\n")
    
    decoded_array = base64.decodestring(data)

    array = np.frombuffer(decoded_array, dtype=eval("np." + args[-1]))
    list_args = []
    for i in range(len(args)-1):
        list_args.append(int(args[i]))
    result_array = np.reshape(array, tuple(list_args))
    return result_array

def get_credentials(event, context):
    scopes = ['https://spreadsheets.google.com/feeds']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('Unknown.json', scopes=scopes)
    gc = gspread.authorize(credentials)
    worksheet = gc.open_by_key('14roBDAm52fwpkwPA-KiXkBDLZZL04jklQHifqgrMPKE').sheet1
    np_array = retrieve(event)
    values = analyze(np_array)
    worksheet.append_row(values)
    return 0

def analyze(arr_arg):
    result = None
    arr = np.array(arr_arg)
    h = scipy.histogram(arr, 256)
    dim = len(arr.shape)
    def func(arr):
        if dim == 3:
            return arr[1:-1, 1:-1, 1:-1]
        else:
            return arr[1:-1, 1:-1]
    gf = func(scipy.ndimage.generic_filter(arr, scipy.mean, size = 3, mode = 'constant'))
    result = gf.flatten()
    gf = func(scipy.ndimage.generic_filter(arr, scipy.median, size = 3, mode = 'constant'))
    result = np.append(result, gf.flatten())
    gf = func(scipy.ndimage.generic_filter(arr, scipy.stats.skew, size = 3, mode = 'constant'))
    result = np.append(result, gf.flatten())
    gf = func(scipy.ndimage.generic_filter(arr, scipy.stats.kurtosis, size = 3, mode = 'constant'))
    result = np.append(result, gf.flatten())
    uniformity = lambda arr : scipy.sum(np.square((scipy.histogram(arr, 256)[0])))
    gf = func(scipy.ndimage.generic_filter(arr, uniformity, size = 3, mode = 'constant'))
    result = np.append(result, gf.flatten())
    def entropy(arr):
        log_ret = np.log2(scipy.histogram(arr, 256)[0])
        log_ret[log_ret==-np.inf]=0
        return np.dot(scipy.histogram(arr, 256)[0], log_ret)  
    gf = func(scipy.ndimage.generic_filter(arr, entropy, size = 3, mode = 'constant'))
    result = np.append(result, gf.flatten())
    gf = func(scipy.ndimage.generic_filter(arr, np.amax, size = 3, mode = 'constant'))
    result = np.append(result, gf.flatten())
    gf = func(scipy.ndimage.generic_filter(arr, np.amin, size = 3, mode = 'constant'))
    result = np.append(result, gf.flatten())
    return result