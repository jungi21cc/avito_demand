import pandas as pd
import numpy as np
import gc
import string
from googletrans import Translator
from datetime import datetime
from sklearn import preprocessing
translator = Translator()
    
def cleanName(text):
    try:
        textProc = text.lower()
        # textProc = " ".join(map(str.strip, re.split('(\d+)',textProc)))
        #regex = re.compile(u'[^[:alpha:]]')
        #textProc = regex.sub(" ", textProc)
        textProc = re.sub('[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]', '', textProc)
        textProc = " ".join(textProc.split())
        return textProc
    except: 
        return "name error"


def load_data():
    print("new data2")
    train = pd.read_csv('../train1.csv')
    test = pd.read_csv('../test1.csv')
    y = train.deal_probability   
    df_all = pd.concat([train, test], axis = 0)

    del train, test
    gc.collect()
    return df_all, y 

def params_(df_all):
    print("params_1,2,3")

    df_all['features'] = df_all.apply(lambda row: ' '.join([str(row['param_1']), str(row['param_2']), str(row['param_3'])]),axis=1)
    print('fill missing values')
    russ_no_para_trans = translator.translate('no parameter provided', dest='ru').text
    df_all['features'].fillna(russ_no_para_trans)
    print('regular expression')
    df_all['features'] = re.sub('nan',  '', df_all['features'])  
    #df_all.drop(["parent_category_name","category_name"],axis=1,inplace=True)
    df_all.drop(["param_1","param_2","param_3"],axis=1,inplace=True)

    gc.collect()
    return df_all

def price_log(df_all):
    print('price log')

    df_all['price'] = np.log(df_all['price'] + 1)
    price_mean = np.median(df_all.price)
    df_all["price"].fillna(price_mean,inplace=True)

    gc.collect()
    return df_all

def item_seq_log(df_all):
    print('item_seq log')

    df_all['item_seq_number'] = np.log(df_all['item_seq_number'] + 1)
    item_mean = np.median(df_all.item_seq_number)
    df_all["item_seq_number"].fillna(item_mean,inplace=True)

    gc.collect()
    return df_all

def activation_date(df_all):
    print('activation date')

    df_all['activation_date'] =  pd.to_datetime(df_all['activation_date'], format = "%Y-%m-%d")

    df_all["Weekday"] = df_all['activation_date'].dt.weekday
    df_all["month"] = df_all['activation_date'].dt.month
    df_all["Weekd of Year"] = df_all['activation_date'].dt.week
    df_all["days"] = df_all['activation_date'].dt.day

    df_all.drop(["activation_date"],axis=1,inplace=True)
    
    return df_all



def label_encoding(df_all):
    categorical = ["user_id","region","city","parent_category_name","category_name","user_type","image_top_1","Time_zone"]
    print('label encoding')

    # Encoder:
    lbl = preprocessing.LabelEncoder()
    for col in categorical:
        #df_all[col].fillna('Unknown')
        df_all[col] = lbl.fit_transform(df_all[col].astype(str))

    return df_all

def split_train_test(df_all, y):
    print('train test split')
    train1 = df_all[:len(y)]
    test1 = df_all[len(y):]

    train1.to_csv('../train1.csv', index=False)
    test1.to_csv('../test1.csv', index=False)
    print('complete : ', train1.shape, test1.shape)
    
def main():
    df_all, y = load_data()
    df_all = params_(df_all)
    df_all = price_log(df_all)
    df_all = item_seq_log(df_all)
    df_all = activation_date(df_all)
    df_all = label_encoding(df_all)
    split_train_test(df_all, y)

if __name__ == "__main__":
    main()


