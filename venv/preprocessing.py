import gspread
import dropbox
import base64
import scipy
import math
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
    result = []
    arr = np.array(arr_arg)
    h = scipy.histogram(arr, 256)

    # ===========================
    # First order stats
    # ===========================

    # Mean
    result.append(scipy.mean(arr))
    # Median
    result.append(scipy.median(arr))
    # Skew
    result.append(scipy.stats.skew(arr, None))
    # Kurtosis
    result.append(scipy.stats.kurtosis(arr, None))
    # Uniformity
    result.append(scipy.sum(np.square(h[0])))
    # Entropy
    def entropy(arr):
        log_ret = np.log2(scipy.histogram(arr, 256)[0])
        log_ret[log_ret==-np.inf]=0
        return np.dot(scipy.histogram(arr, 256)[0], log_ret)  
    result.append(entropy(arr))
    # Maximum
    result.append(np.amax(arr))
    # Minimum
    result.append(np.amin(arr))
    # Energy
    energy = np.sum(np.square(arr))
    result.append(energy)
    # Root Mean Square
    result.append(math.sqrt(energy / arr.size))
    # Standard Deviation
    result.append(np.std(arr, ddof=1))
    #change
    return result