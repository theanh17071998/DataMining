from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

#dau vao la cai cot cmt ,Y thi la cot label
def feature_select(corpus, labels, k=1000000):
    """
    select top k features through chi-square test
    """
    bin_cv = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, binary=True)
    le = LabelEncoder()
    X = bin_cv.fit_transform(corpus, np.nan)
    y = le.fit_transform(labels).reshape(-1, 1)

    #k = min(X.shape[1], k)
    skb = SelectKBest(chi2, k='all')
    skb.fit(X, y)

    feature_ids = skb.get_support(indices=True)
    feature_names = bin_cv.get_feature_names()
    result = {}
    vocab = []

    for new_fid, old_fid in enumerate(feature_ids):
        feature_name = feature_names[old_fid]
        vocab.append(feature_name)

    result['text'] = vocab
    result['_score'] = list(skb.scores_)
    result['_pvalue'] = list(skb.pvalues_)

    # we only care about the final extracted feature vocabulary
    return result


input_data = [
    # '/content/drive/MyDrive/DataScience/Deeplearning/mebe_tiki.csv',
    # '/content/drive/MyDrive/DataScience/Deeplearning/mebe_shopee.csv',
    '/content/drive/MyDrive/DataScience/Deeplearning/tech_tiki.csv',
    '/content/drive/MyDrive/DataScience/Deeplearning/tech_shopee.csv',

]
LABEL = {
    '/content/drive/MyDrive/DataScience/Deeplearning/tech_tiki.csv': 'giá,dịch_vụ,ship,hiệu_năng,chính_hãng,cấu_hình,phụ_kiện,mẫu_mã',
    '/content/drive/MyDrive/DataScience/Deeplearning/tech_shopee.csv': 'giá,dịch_vụ,ship,hiệu_năng,chính_hãng,cấu_hình,phụ_kiện,mẫu_mã',
    # '/content/drive/MyDrive/DataScience/Deeplearning/mebe_tiki.csv': 'giá,dịch_vụ,an_toàn,chất_lượng,ship,chính_hãng',
    # '/content/drive/MyDrive/DataScience/Deeplearning/mebe_shopee.csv': 'giá,dịch_vụ,an_toàn,chất_lượng,ship,chính_hãng',
    # 'data_train\\hotel_data.csv': 'aspect1,aspect2,aspect3,aspect4,aspect5'

}
from sklearn.datasets import load_iris

if __name__ == '__main__':
    for f in input_data:

        print('-----------------{}---------------'.format(f))

        label = LABEL[f].split(',')

        for l in label:
            name = f.split('/')
            name = name[6].split('.')
            name = name[0]

            #f_out = open('chi2\\label_{}'.format(l) + '_{}.csv'.format(name), 'w', encoding='utf-8')
            df = pd.read_csv(f, encoding='utf-8')

            # df = df[df[l] != 0]
            data = df['cmt'].astype(str)

            data_label = df[l]
            data_train = []
            data_train_dict = []
            for k in data:
                data_train.append(k)
            for k1 in data_label:
                data_train_dict.append(abs(k1))

            file_out = feature_select(data_train, data_train_dict)
            df = pd.DataFrame(file_out)
            df =df.sort_values('_score', ascending=False)
            df.to_csv('/content/drive/MyDrive/DataScience/Deeplearning/chi2/label_{}'.format(l) + '_{}'.format(name + str(f.split('/')[1])), encoding='utf-8')
