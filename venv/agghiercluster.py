#from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import FeatureAgglomeration
import pickle
import gspread
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials

def cluster(event, context):
    # we give the user the option to either do feature agglomerative clustering
    # (that is a dimensionality reduction technique)
    # or sample agglomerative clustering
    # (that is, strictly speaking, a clustering technique)
    key = event['key']
    is_feature = event['feat_or_sample']
    scopes = ['https://spreadsheets.google.com/feeds']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('Unknown.json', scopes=scopes)
    gc = gspread.authorize(credentials)
    worksheet = gc.open_by_key(key).sheet1
    
    arr = np.array(worksheet.get_all_values())
    X = arr[1:, :-1]

    X_converted = X.astype(np.float)

    if is_feature:
        hier_means = AgglomerativeClustering(n_clusters=10).fit(X)
    else:
        hier_means = FeatureAgglomeration(n_clusters=10).fit(X)

    s = pickle.dumps(hier_means)
    worksheet2 = gc.open_by_key(key).get_worksheet(1)
    worksheet2.update_acell('A3', s)
    return 1