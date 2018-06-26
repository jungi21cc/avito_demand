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
    print("new data")

    gp = pd.read_csv("../gp.csv")
    train = pd.read_csv('../train.csv')
    test = pd.read_csv('../test.csv')
    train = train.merge(gp, on='user_id', how='left')
    test = test.merge(gp, on='user_id', how='left')
    y = train.deal_probability

    df_all = pd.concat([train, test], axis = 0)

    del gp, train, test
    gc.collect()
    return df_all, y 

def region_city(df_all):
    print("region / city")  

    ###Regional
    region_map = {"Свердловская область" : "Sverdlovsk oblast",
                "Самарская область" : "Samara oblast",
                "Ростовская область" : "Rostov oblast",
                "Татарстан" : "Tatarstan",
                "Волгоградская область" : "Volgograd oblast",
                "Нижегородская область" : "Nizhny Novgorod oblast",
                "Пермский край" : "Perm Krai",
                "Оренбургская область" : "Orenburg oblast",
                "Ханты-Мансийский АО" : "Khanty-Mansi Autonomous Okrug",
                "Тюменская область" : "Tyumen oblast",
                "Башкортостан" : "Bashkortostan",
                "Краснодарский край" : "Krasnodar Krai",
                "Новосибирская область" : "Novosibirsk oblast",
                "Омская область" : "Omsk oblast",
                "Белгородская область" : "Belgorod oblast",
                "Челябинская область" : "Chelyabinsk oblast",
                "Воронежская область" : "Voronezh oblast",
                "Кемеровская область" : "Kemerovo oblast",
                "Саратовская область" : "Saratov oblast",
                "Владимирская область" : "Vladimir oblast",
                "Калининградская область" : "Kaliningrad oblast",
                "Красноярский край" : "Krasnoyarsk Krai",
                "Ярославская область" : "Yaroslavl oblast",
                "Удмуртия" : "Udmurtia",
                "Алтайский край" : "Altai Krai",
                "Иркутская область" : "Irkutsk oblast",
                "Ставропольский край" : "Stavropol Krai",
                "Тульская область" : "Tula oblast"}

    df_all['region_en'] = df_all['region'].apply(lambda x : cleanName(region_map[x]))

    regional = pd.read_csv("../regional.csv", index_col = [0])
    regional["region_en"] = regional.index
    regional["region_en"] = regional["region_en"].apply(lambda x: cleanName(x))

    df_all = df_all.merge(regional, on = "region_en", how = "left").drop("region_en", axis = 1)
    df_all["Total_population"] = np.log(df_all["Total_population"]+1)
    df_all["Total_population"].fillna(df_all.Total_population.mean(),inplace=True)

    gc.collect()
    return df_all

def category_name(df_all):
    print("parent_category_name / category_name")   

#    df_all['categories'] = df_all.apply(lambda row: ' '.join([str(row['parent_category_name']), str(row['category_name'])]),axis=1)
    
    numeric = ["image_blurrness_score", "avg_days_up_user", "avg_times_up_user", "n_user_items", "Density_of_region(km2)", "Rural_%", "Urban%"]
    for col in numeric:
        df[col].fillna(-1, inplace = True)



    gc.collect()
    return df_all

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

    df_all.drop(["activation_date","image"],axis=1,inplace=True)
    
    return df_all



def label_encoding(df_all):
    categorical = ["user_id","region","city","parent_category_name","category_name","user_type","image_top_1","Time_zone"]
    print("Encoding :",categorical)

    # Encoder:
    lbl = preprocessing.LabelEncoder()
    for col in categorical:
        #df_all[col].fillna('Unknown')
        df_all[col] = lbl.fit_transform(df_all[col].astype(str))

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
    df_all = label_encoding(df_all)
    split_train_test(df_all, y)

if __name__ == "__main__":
    main()


