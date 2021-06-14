import pickle

import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split

DOMAIN_TRAIN = {
    # 'mebe_tiki': 'giá,dịch_vụ,an_toàn,chất_lượng,ship,chính_hãng'
    
}

def model(domain,label):
    print('\n--------------{}--------------'.format(domain+' '+ label))
    review = pd.read_csv("D:\\DataScience\\Deeplearning\\{}.csv".format(domain))
    review.isnull().values.any()
    data_an_toan = zip(review['cmt'], review['{}'.format(label)])
    data = pd.DataFrame(data_an_toan,columns=["review","sentiment"])
    data = data[data['sentiment'] != 0]
    X = []
    sentences = list(data['review'])
    for sen in sentences:
        X.append(sen)
    y =data['sentiment']
    y = np.array(list(map(lambda x: 1 if x==1 else 0, y)))
    # y = np.array(list(map(lambda x: -1 if x==-1 else (1 if x== 1 else 0), y)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    phobert = AutoModel.from_pretrained("vinai/phobert-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    a = []
    for text in X_train:
        input_ids = torch.tensor([tokenizer.encode(text)])
        with torch.no_grad():
            features = phobert(input_ids)
            a.append(features[0].numpy())
    xtest1 = []
    for text in X_test:
        input_ids = torch.tensor([tokenizer.encode(text)])
        with torch.no_grad():
            features = phobert(input_ids)
            xtest1.append(features[0].numpy())
    xtest = xtest1
    xtrain = a
    imble = 0
    imble1 = 0
    for k in xtrain:
        imble =max(imble, k.shape[1])
    for k in xtest:
        imble1 = max(imble1, k.shape[1])
    imble = max(imble1,imble)
    xtrain_new = []
    for k in xtrain:
        xtrain_new.append(np.hstack([k, np.zeros([1,imble-k.shape[1] ,768])])[0])
    xtrain_new =np.array(xtrain_new)
    print(xtrain_new.shape)
    xtrain_new_test = []
    for k in xtest:
        xtrain_new_test.append(np.hstack([k, np.zeros([1, imble - k.shape[1], 768])])[0])
    xtrain_new_test = np.array(xtrain_new_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model_input = tf.keras.layers.Input(shape=(imble, 768))
    output_cnn_2 = tf.keras.layers.Conv1D(128, 2)(model_input)
    output_cnn_2 = tf.keras.layers.MaxPool1D(imble-1)(output_cnn_2)
    output_cnn_2 = tf.keras.layers.Flatten()(output_cnn_2)

    output_cnn_3 = tf.keras.layers.Conv1D(128, 3)(model_input)
    output_cnn_3 = tf.keras.layers.MaxPool1D(imble-2)(output_cnn_3)
    output_cnn_3 = tf.keras.layers.Flatten()(output_cnn_3)

    output_cnn_4 = tf.keras.layers.Conv1D(128, 4)(model_input)
    output_cnn_4 = tf.keras.layers.MaxPool1D(imble-3)(output_cnn_4)
    output_cnn_4 = tf.keras.layers.Flatten()(output_cnn_4)

    output_cnn = tf.concat([output_cnn_2, output_cnn_3, output_cnn_4], axis=-1)

    output_mlp = tf.keras.layers.Dense(128)(output_cnn)
    output_mlp = tf.keras.layers.Dense(64)(output_mlp)

    final_output = tf.keras.layers.Dense(1, activation='sigmoid')(output_mlp)

    model = tf.keras.models.Model(inputs=model_input, outputs=final_output)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    model.fit(xtrain_new, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.2)
    score = model.evaluate(xtrain_new_test, y_test, verbose=1)
    result_xtest1 = model.predict(xtrain_new_test)
    result_xtest1 = np.array(result_xtest1.reshape(len(y_test)))

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


if __name__ == '__main__':
    for domain in DOMAIN_TRAIN:
        for label in DOMAIN_TRAIN[domain].split(','):
            model(domain, label)


