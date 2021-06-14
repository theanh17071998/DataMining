import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import pickle
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split

DOMAIN_TRAIN = {
    'mebe_tiki': 'giá,dịch_vụ,an_toàn,chất_lượng,ship,chính_hãng'
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    phobert = AutoModel.from_pretrained("vinai/phobert-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    a = []
    for text in X_train:
        input_ids = torch.tensor([tokenizer.encode(text)])
        with torch.no_grad():
            features = phobert(input_ids)
            a.append(features[1].numpy())
    xtrain = np.array(a)
    xtest = []
    for text in X_test:
        input_ids = torch.tensor([tokenizer.encode(text)])
        with torch.no_grad():
            features = phobert(input_ids)
            xtest.append(features[1].numpy())
    xtest1 = np.array([(i) for i in xtest])
    y_test = np.array(y_test)
    y_train = np.array(y_train)
    model = Sequential()
    model.add(Dense(128, activation="relu", name="layer1"))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    model.fit(xtrain, y_train, batch_size=128, epochs=20, verbose=1, validation_split=0.2)

    # some time later...

    # load the model from disk
    filename = 'Save_model\\finalizesssd_model_' + '{}'.format(label) + '.sav'
    model.save(filename)
    #print(result)
    score = model.evaluate(xtest1, y_test, verbose=1)
    result_xtest1 = model.predict(xtest1)
    result_xtest1 = np.array(result_xtest1.reshape(len(y_test)))


    tp = 0
    fp = 0
    fn = 0
    for k,k1 in zip(y_test,result_xtest1):
        if k == 1 and k1>0.5:
            tp += 1
        elif k ==1:
            fn +=1
        elif k1>0.5:
            fp+=1


    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)

    print('p:', p)
    print('r:', r)
    print('f1:', f1)

    print(score)
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])
if __name__ == '__main__':
    for domain in DOMAIN_TRAIN:
        for label in DOMAIN_TRAIN[domain].split(','):
            model(domain,label)