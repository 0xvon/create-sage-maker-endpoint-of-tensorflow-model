import argparse
import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from sklearn.externals import joblib
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import boto3
import glob
import subprocess

def download(code_dir, file_name):
    s3 = boto3.resource('s3', aws_access_key_id="xxx", aws_secret_access_key="xxx")
    bucket_name = "xxx"
    file = file_name.split("/")[-1]
    print("filename: {}".format(file))
    content_object = s3.Object(bucket_name, file_name)
    s3.Bucket(bucket_name).download_file(file_name, os.path.join(code_dir, file))


# def input_fn(data, context):
#     if context.request_content_type == 'application/json':
#         # pass through json (assumes it's correctly formed)
#         d = data.read().decode('utf-8')
#         try:
#             scaler = joblib.load('scaler.save')
#             d = list(map(lambda x: float(x), re.sub("[\[\] ]", "", d).split(",")))
#             d = np.array(d)

#             d = d.reshape(-1, 1)
            
#             dataset_sc = scaler.transform(np.log(d)).reshape(1, -1, 1)
#             return json.dumps({
#                 'instances': dataset_sc.tolist()
#             })
#         except:
#             raise ValueError('{{"error": "could not preprocess input data: {}"}}'.format(d))

#     if context.request_content_type == 'text/csv':
#         # very simple csv handler
#         return json.dumps({
#             'instances': [float(x) for x in data.read().decode('utf-8').split(',')]
#         })

#     raise ValueError('{{"error": "unsupported content type {}"}}'.format(context.request_content_type or "unknown"))


# def output_fn(data, context):
#     if data.status_code != 200:
#         raise ValueError(data.content.decode('utf-8'))

#     response_content_type = context.accept_header
#     prediction = data.content.decode('utf-8')
#     prediction = float(re.sub("[a-z\{\}\"\\\\ \:\[\]]", "", prediction))
#     try:
#         scaler = joblib.load('scaler.save')
#         prediction = np.array([prediction]).reshape(-1, 1)
            

#         dataset_sc = np.exp(scaler.inverse_transform(prediction))
#         res = json.dumps({
#             "prediction": int(round(dataset_sc.flatten()[0]))
#         })
#         return res, response_content_type
#     except:
#         raise ValueError('{{"error": "could not postprocess output data"}}')    

def inverse_original_scale(train_dir, x):
    scaler = joblib.load(os.path.join(train_dir, 'scaler.save'))
    x = scaler.inverse_transform(x.reshape(1, -1))
    x = np.exp(x)
    return x

def _load_data(data, n_prev=50):
    docX, docY = [], []
    for i in range(len(data) - n_prev):
        if i == 0:
            continue
        docX.append(data.iloc[i - 1:i + n_prev - 1].values)
        docY.append(data.iloc[i + n_prev].values)
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY


# 学習用とテスト用データを分割、ただし分割する際に_load_data()を適用
def train_test_split(df, test_size=0.1, n_prev=50):
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)

    return (X_train, y_train), (X_test, y_test)


def preprocessing(train_dir, dataset, is_x):
    if dataset.ndim != 2:
        dataset = dataset.reshape(-1, 1)
    scaler = joblib.load(os.path.join(train_dir, 'scaler.save'))
    dataset_src = scaler.transform(np.log(dataset))
    if is_x:
        return dataset_src.reshape(-1, 24, 1)

    else:
        return dataset_src


def save(model, model_dir):
    sess = K.get_session()
    tf.saved_model.simple_save(
        sess,
        os.path.join(model_dir, 'model/1'),
        inputs={'inputs': model.input},
        outputs={t.name: t for t in model.outputs})


def train(args):
    batch_size = args.batch_size
    epochs = args.epochs
    train_dir = args.train
    code_dir = os.path.join(args.model_dir, 'code')
    os.mkdir(code_dir)
    channel_name = 'training'
    
    input_data_files = ["dataset/anomaly_detection/inference.py", "dataset/anomaly_detection/scaler.save", "dataset/anomaly_detection/requirements.txt"]
    for input_data_file in input_data_files:
        download(code_dir, input_data_file)

    # load data

    # Take the set of files and read them all into a single pandas dataframe
    # input_files = [os.path.join(train_dir, file)
    #                for file in os.listdir(train_dir)]

    # if len(input_files) == 0:
    #     raise ValueError(('There are no files in {}.\n' +
    #                       'This usually indicates that the channel ({}) was incorrectly specified,\n' +
    #                       'the data specification in S3 was incorrectly specified or the role specified\n' +
    #                       'does not have permission to access the data.').format(train_dir, channel_name))
    
    input_files = glob.glob("{}/*.csv".format(train_dir))
    
    raw_data = [ pd.read_csv(file) for file in input_files ]
    train_data = pd.concat(raw_data)

    length_of_sequences = 24
    in_out_neurons = 1
    hidden_neurons = 100
    (x_train, y_train), (x_test, y_test) = train_test_split(
        train_data[["demand"]], test_size=0.2, n_prev=length_of_sequences)

    x_train = preprocessing(train_dir, x_train, True)
    y_train = preprocessing(train_dir, y_train, False)
    x_test = preprocessing(train_dir, x_test, True)
    y_test = preprocessing(train_dir, y_test, False)

    model = Sequential()
    model.add(
        LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))
    model.add(Dense(in_out_neurons))
    model.add(Activation("linear"))

    es_cb = EarlyStopping(monitor='val_loss',
                          patience=0.01, verbose=1, mode='auto')
    model.compile(loss="mean_squared_error", optimizer="adam")

    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[es_cb],
    )
    
    pred = inverse_original_scale(train_dir, model.predict(x_test)).flatten()
    ans = inverse_original_scale(train_dir, y_test).flatten()
    
    error = (ans - pred) ** 2
    error.sort()
    threshold_value = error[int(round((-len(error) / 10))):].mean()
    threshold_json = {"threshold_value": threshold_value}
    
    with open(os.path.join(code_dir, "threshold.json"), 'w') as f:
        json.dump(threshold_json, f, ensure_ascii=False)
        
    
    s3 = boto3.resource('s3', aws_access_key_id="xxx", aws_secret_access_key="xxx")
    bucket_name = "xxx"
    file = "dataset/anomaly_detection/threshold.json"
    content_object = s3.Object(bucket_name, file)
    s3.Bucket(bucket_name).upload_file(os.path.join(code_dir, "threshold.json"), file)
    
    save(model, args.model_dir)


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=12)

    # input data and model directories
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str,
                        default=os.environ['SM_CHANNEL_TRAINING'])

    args, _ = parser.parse_known_args()

    train(args)
