# fully connected layer
# this method takes input matrices 
# and multiplies it by a weight matrix

import numpy as np
import boto
import boto3
import json

def forward_prop(event, context):
	conn = boto.connect_s3("AKIAIMQLHJNMP6DOUM4A","8dJAfPZlTjMR1SOcOetImclAmT+G02VkQiuHefdY")