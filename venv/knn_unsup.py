from sklearn.neighbors import NearestNeighbors
import pickle
import gspread
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials

def neighbors(event, context):
    key = event['key']
    scopes = ['https://spreadsheets.google.com/feeds']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('Unknown.json', scopes=scopes)
    gc = gspread.authorize(credentials)
    worksheet = gc.open_by_key(key).sheet1
    arr = np.array(worksheet.get_all_values())
    X = arr[1:, :-1]
    y = arr[1:, -1]

    X_converted = X.astype(np.float)
    y_converted = y.astype(np.float)

    # we use ball tree algo here because of a large # of features
    # Other options are brute force or KD-trees, but it would be less effective
    # due to the curse of dimensionality

    # Brute force query time grows as O[D N]
    # Ball tree query time grows as approximately O[Dlog(N)]
    # KD tree query time changes with D in a way that is difficult to precisely characterise. 
    # For small D (less than 20 or so) the cost is approximately O[Dlog(N)], 
    # and the KD tree query can be very efficient. 
    # For larger D, the cost increases to nearly O[DN], 
    # and the overhead due to the tree structure can lead to queries which are slower than brute force.
    clf = NearestNeighbors(n_neighbors=3, algorithm='ball_tree')

    clf.fit(X_converted, y_converted)

    s = pickle.dumps(clf)

    worksheet2 = gc.open_by_key(key).get_worksheet(1)
    worksheet2.update_acell('A5', s)
    return 1

