# -*- coding: utf-8 -*-
"""
dicom2nifti

@author: abrys
"""

from __future__ import print_function

import dicom2nifti.patch_pydicom_encodings

dicom2nifti.patch_pydicom_encodings.apply()

import logging

import pydicom.config as pydicom_config
from pydicom.tag import Tag

import dicom2nifti.common as common
import dicom2nifti.convert_generic as convert_generic

pydicom_config.enforce_valid_values = False
logger = logging.getLogger(__name__)


def dicom_to_nifti(dicom_input, output_file=None):
    """
    This is the main dicom to nifti conversion fuction for hitachi images.
    As input hitachi images are required. It will then determine the type of images and do the correct conversion

    Examples: See unit test

    :param output_file: file path to the output nifti
    :param dicom_input: directory with dicom files for 1 scan
    """

    assert common.is_hitachi(dicom_input)

    # TODO add validations and conversion for DTI and fMRI once testdata is available

    logger.info('Assuming anatomical data')
    return convert_generic.dicom_to_nifti(dicom_input, output_file)


