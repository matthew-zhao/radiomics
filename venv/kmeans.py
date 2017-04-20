#from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import pickle
import gspread
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials

def cluster(event, context):
    key = event['key']
    
    # scopes = ['https://spreadsheets.google.com/feeds']
    # credentials = ServiceAccountCredentials.from_json_keyfile_name('Unknown.json', scopes=scopes)
    # gc = gspread.authorize(credentials)
    # worksheet = gc.open_by_key(key).sheet1
    
    # arr = np.array(worksheet.get_all_values())
    X = arr[1:, :-1]

    X_converted = X.astype(np.float)

    kmeans = KMeans(n_clusters=10, random_state=0).fit(X)

    s = pickle.dumps(kmeans)
    worksheet2 = gc.open_by_key(key).get_worksheet(1)
    worksheet2.update_acell('A2', s)
    return 1