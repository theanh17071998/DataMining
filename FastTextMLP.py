import io
import numpy as np
import pandas as pd
import string
import torch
from transformers import AutoModel, AutoTokenizer
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_fasttext(path):
    num_words = 100000
    fin = io.open(path, 'r', encoding='utf-8',
                  newline='\n', errors='ignore')
    n, d = fin.readline().split()
    data = {}
    i = 0
    for line in fin:
        i += 1
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array([float(val) for val in tokens[1:]])
        data[tokens[0]] /= np.linalg.norm(data[tokens[0]])
        if i > num_words:
            break

    return data


if __name__ == '__main__':
    data_ = load_fasttext('/content/drive/MyDrive/DataScience/Deeplearning/vn_fasttext/cc.vi.300.vec')
    review = pd.read_csv("/content/drive/MyDrive/DataScience/Deeplearning/mebe_tiki.csv")
    review.isnull().values.any()
    data_an_toan = zip(review['cmt'], review['giá'])
    data = pd.DataFrame(data_an_toan, columns=["review", "sentiment"])
    data = data[data['sentiment'] != 0]
    X = []
    sentences = list(data['review'])
    for sen in sentences:
        X.append(sen)
    y = data['sentiment']
    y = np.array(list(map(lambda x: 1 if x == 1 else 0, y)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    a = []
    for text in X_train:
        arr_text = text.split(' ')
        mt = []
        for ar in arr_text:
            if ar in data_:
                mt.append(data_[ar])
            else:
                mt.append(np.zeros(300))
        print(len(mt))
        a.append(mt)
    xtest1 = []
    for text in X_test:
        arr_text = text.split(' ')
        mt = []
        for ar in arr_text:
            if ar in data_:
                mt.append(data_[ar])
            else:
                mt.append(np.zeros(300))
        xtest1.append(mt)
    xtest = xtest1
    xtrain = a
    imble = 0
    imble1 = 0
    for k in xtrain:
        imble = max(imble, len(k))
    for k in xtest:
        imble1 = max(imble1, len(k))
    imble = max(imble1, imble)
    xtrain_new = []
    for k in xtrain:
        xtrain_new.append(np.vstack((((np.array(k))),np.zeros((imble-len(k),300)))))
    xtrain_new = np.array(xtrain_new)
    print(xtrain_new.shape)
    xtrain_new_test = []
    for k in xtest:
        xtrain_new_test.append(np.vstack((((np.array(k))),np.zeros((imble-len(k),300)))))
    xtrain_new_test = np.array(xtrain_new_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # model_input = tf.keras.layers.Input(shape=(imble, 300))
    # output_cnn_2 = tf.keras.layers.Conv1D(128, 2)(model_input)
    # output_cnn_2 = tf.keras.layers.MaxPool1D(imble-1)(output_cnn_2)
    # output_cnn_2 = tf.keras.layers.Flatten()(output_cnn_2)

    # output_cnn_3 = tf.keras.layers.Conv1D(128, 3)(model_input)
    # output_cnn_3 = tf.keras.layers.MaxPool1D(imble-2)(output_cnn_3)
    # output_cnn_3 = tf.keras.layers.Flatten()(output_cnn_3)

    # output_cnn_4 = tf.keras.layers.Conv1D(128, 4)(model_input)
    # output_cnn_4 = tf.keras.layers.MaxPool1D(imble-3)(output_cnn_4)
    # output_cnn_4 = tf.keras.layers.Flatten()(output_cnn_4)

    # output_cnn = tf.concat([output_cnn_2, output_cnn_3, output_cnn_4], axis=-1)

    # output_mlp = tf.keras.layers.Dense(128)(output_cnn)
    # output_mlp = tf.keras.layers.Dense(64)(output_mlp)

    # final_output = tf.keras.layers.Dense(1, activation='sigmoid')(output_mlp)

    # model = tf.keras.models.Model(inputs=model_input, outputs=final_output)

    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()
    # model.fit(xtrain_new, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.2)
    # score = model.evaluate(xtrain_new_test, y_test, verbose=1)
    # result_xtest1 = model.predict(xtrain_new_test)
    # result_xtest1 = np.array(result_xtest1.reshape(len(y_test)))
    model = Sequential()
    model.add(Dense(128, activation="relu", name="layer1"))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    model.fit(xtrain_new, y_train, batch_size=128, epochs=20, verbose=1, validation_split=0.2)

    score = model.evaluate(xtrain_new_test, y_test, verbose=1)
    result_xtest1 = model.predict(xtrain_new_test)
    # result_xtest1 = np.array(result_xtest1.reshape(len(y_test)))
    result_xtest1 = np.argmax(result_xtest1, axis=1)

    tp = 0
    fp = 0
    fn = 0
    for k, k1 in zip(y_test, result_xtest1):
        if k == 1 and k1 > 0.5:
            tp += 1
        elif k == 1:
            fn += 1
        elif k1 > 0.5:
            fp += 1

    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)

    print('p:', p)
    print('r:', r)
    print('f1:', f1)


    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])
    # target_names=ten nhan
    # classificationReport = classification_report(y_test, result_xtest1, target_names=target_names)
