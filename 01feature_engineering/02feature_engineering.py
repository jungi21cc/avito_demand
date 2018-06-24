import pandas as pd
import numpy as np
import gc
import string

def load_data():
    print("new data")

    gp = pd.read_csv("../gp.csv")
    train = pd.read_csv('../train.csv')
    test = pd.read_csv('../test.csv')
    train = train.merge(gp, on='user_id', how='left')
    test = test.merge(gp, on='user_id', how='left')
    y = train.deal_probability

    df_all = pd.concat([train, test], axis = 0)

    del gp, train, test
    
    return df_all, y 

def region_city(df_all):
    print("region / city")  

    df_all['region_city'] = df_all.apply(lambda row: ' '.join([str(row['region']), str(row['city'])]),axis=1)
    df_all.drop(["region","city"],axis=1,inplace=True)

    return df_all

def category_name(df_all):
    print("parent_category_name / category_name")   

    df_all['categories'] = df_all.apply(lambda row: ' '.join([str(row['parent_category_name']), str(row['category_name'])]),axis=1)

    return df_all

def params_(df_all):
    print("params_1,2,3")

    df_all['features'] = df_all.apply(lambda row: ' '.join([str(row['param_1']), str(row['param_2']), str(row['param_3'])]),axis=1)
    df_all.drop(["param_1","param_2","param_3"],axis=1,inplace=True)
    df_all['features'].fillna(df_all.category_name)
    df_all.drop(["parent_category_name","category_name"],axis=1,inplace=True)

    return df_all

def price_log(df_all):
    print('price log')

    df_all['price'] = np.log(df_all['price'] + 1)
    
    return df_all

def item_seq_log(df_all):
    print('item_seq log')

    df_all['item_seq_number'] = np.log(df_all['item_seq_number'] + 1)

    return df_all

def activation_date(df_all):
    print('activation date')

    return df_all

def split_train_test(df_all, y):
    train1 = df_all[:len(y)]
    test1 = df_all[len(y):]

    train1.to_csv('../train1.csv', index=False)
    test1.to_csv('../test1.csv', index=False)
    
def main():
    df_all, y = load_data()
    df_all = region_city(df_all)
    df_all = category_name(df_all)
    df_all = params_(df_all)
    df_all = price_log(df_all)
    df_all = item_seq_log(df_all)
    df_all = activation_date(df_all)
    split_train_test(df_all, y)

if __name__ == "__main__":
    main()


