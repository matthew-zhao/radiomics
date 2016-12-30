from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import gspread
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials

def classify(event, context):
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

    # Uses Gini impurity rather than information gain (shouldn't really matter)
    # TODO: may want to set a max-depth (currently all nodes expanded
    # until all leaves are pure or less than min_samples_split samples (2 by default))

    # This random forest uses 50 decision trees.
    # It conducts both feature and sample bagging by choosing from only 75% of all available samples
    # and 33% from all available features. This creates slightly higher bias, but the 50
    # trees averaging should decrease variance by much more than bias increases by.
    # Samples are drawn w/ replacement, but features are not. 
    clf = BaggingClassifier(base_estimator=None, n_estimators=50, max_samples=0.75, max_features=0.33)

    clf = clf.fit(X_converted, y_converted)

    s = pickle.dumps(clf)

    worksheet2 = gc.open_by_key(key).get_worksheet(1)
    worksheet2.update_acell('A8', s)
    return 1

