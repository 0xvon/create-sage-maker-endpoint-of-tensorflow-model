import argparse
import os
import numpy as np
import tensorflow as tf
import boto3
import re
import requests
import json
import subprocess


def handler(data, context):
    """Handle request.
    Args:
        data (obj): the request data
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, (optional) response content type
    """
    
    processed_input = _process_input(data, context)
#     response = requests.post(context.rest_uri, data=processed_input)
    return _process_output(processed_input, context)


def _process_input(data, context):
    if context.request_content_type == 'application/json':
        # pass through json (assumes it's correctly formed)
        d = data.read().decode('utf-8')
        try:
            data_list = json.loads(d)

            
            true_value = np.array(data_list['true'])
            pred_value = np.array(data_list['pred'])
#             raise ValueError('{{value: {}, type: {}}}'.format(json.loads(data_list), type(json.loads(data_list))))            
            
            mse = ((true_value - pred_value) ** 2).mean()
            
            return json.dumps({
                'mse': mse
            })
        except:
            raise ValueError('{{"error": "could not preprocess input data: {}"}}'.format(d))

    if context.request_content_type == 'text/csv':
        # very simple csv handler
        return json.dumps({
            'instances': [float(x) for x in data.read().decode('utf-8').split(',')]
        })

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(context.request_content_type or "unknown"))


def _process_output(data, context):

    response_content_type = context.accept_header
    prediction = json.loads(data)
    s3 = boto3.resource('s3', aws_access_key_id="AKIAXAGPD6STVODIEREU", aws_secret_access_key="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")
    bucket_name = "sagemaker-ap-northeast-1-481470706855"
    file = "dataset/anomaly_detection/threshold.json"
    content_object = s3.Object(bucket_name, file)
    s3.Bucket(bucket_name).download_file(file, 'threshold.json')
    try:
        f = open("./threshold.json")
        threshold = json.load(f)
        threshold_value = threshold['threshold_value']
        mse = prediction['mse']
        if mse > threshold_value:
            is_anomaly = 1
        else:
            is_anomaly = 0
                
        res = json.dumps({
            "judgement": is_anomaly
        })
        return res, response_content_type
    except:
        result = subprocess.check_output("ls")
        raise ValueError('{{"error": "could not postprocess output data", ls: {}}}'.format(result))

