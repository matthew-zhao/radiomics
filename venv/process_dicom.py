import boto3
import json
import dropbox
import csv
import os, errno
import boto
import dicom2nifti
import dicom2nifti.settings as settings
import numpy as np
import nibabel as nib
from boto.s3.key import Key
import nrrd
import shutil
# only needed for dcm stuff
from multiprocessing import Process, Pipe

from io import BytesIO

def upload_to_s3(bucket, file, name, conn):
    key = bucket.new_key(name)
    key.set_contents_from_file(file)
    conn.send(['success'])
    conn.close()
    #os.remove(file)

def process_dcm_series(dirname, image_key_names, paths, bucket):
#def process_dcm_series(event, context):

    # See https://github.com/icometrix/dicom2nifti/issues/36 for why we do this
    settings.disable_validate_sliceincrement()
    settings.set_gdcmconv_path('./gdcmconv')
    #settings.disable_validate_orientation()

    dcm_series_path = os.path.join('/tmp', 'dcmseries.nii')
    dicom2nifti.dicom_series_to_nifti(dirname, dcm_series_path, reorient_nifti=False)
    img = nib.load(dcm_series_path)
    img_arr = np.array(img.dataobj)

    # we convert the path to the enhanced dicom slices to a string joined with underscores
    split_path_list = os.path.normpath(dirname).split(os.path.sep)[2:]
    relative_path = os.path.join(*split_path_list)
    path_as_string = '_'.join(split_path_list)

    shutil.rmtree(dirname)
    return upload_slices(img_arr, path_as_string, paths, bucket, image_key_names, relative_path)

def upload_slices(img_arr, volume_name, paths, bucket, image_key_names, spreadsheet_name=None):
    slice_arrs = [np.squeeze(subarray) for subarray in np.dsplit(img_arr, img_arr.shape[2])]

    # create a list to keep all processes
    processes = []

    # create a list to keep connections
    parent_connections = []

    index = 0
    for indiv_slice in slice_arrs:
        #create process for each upload
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)

        upload_name = volume_name + '_' + str(index) + '.npy'
        # uncomment this if BytesIO doesn't work!!!
        #upload_path = os.path.join('/tmp', upload_name)
        output_file_object = BytesIO()
        np.save(output_file_object, indiv_slice)
        output_file_object.seek(0)

        index += 1

        image_key_names.append(spreadsheet_name + '_' + str(index))
        paths.append(upload_name)

        # create the process, pass instance and connection
        process = Process(target=upload_to_s3, args=(bucket, output_file_object, upload_name, child_conn,))
        processes.append(process)

    for process in processes:
        process.start()

    # make sure all processes have finished
    for process in processes:
        process.join()

    return image_key_names, paths