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
    print('merge user periods')
    train = train.merge(gp, on='user_id', how='left')
    test = test.merge(gp, on='user_id', how='left')
    y = train.deal_probability
    print('merge image blurrness')
    train_blur = pd.read_csv('../train_blurrness.csv')
    test_blur = pd.read_csv('../test_blurrness.csv')
    train = train.merge(train_blur, on='item_id', how='left')
    test = test.merge(test_blur, on='item_id', how='left')
    
    df_all = pd.concat([train, test], axis = 0)

    del gp, train, test, train_blur, test_blur
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

    print('merge region')
    df_all = df_all.merge(regional, on = "region_en", how = "left").drop("region_en", axis = 1)
    df_all["Total_population"] = np.log(df_all["Total_population"]+1)
    df_all["Total_population"].fillna(df_all.Total_population.mean(),inplace=True)

    del regional
    gc.collect()
    return df_all

def category_name(df_all):
    print("parent_category_name / category_name")   
#    df_all['categories'] = df_all.apply(lambda row: ' '.join([str(row['parent_category_name']), str(row['category_name'])]),axis=1)
    print('fill null numeric data')
    numeric = ["image_blurrness_score", "avg_days_up_user", "avg_times_up_user", "n_user_items", "Density_of_region(km2)", "Rural_%", "Urban%"]
    for col in numeric:
        temp_median = np.mean(df_all[col])
        df_all[col].fillna(temp_median, inplace = True)
        print('complete : ', col)
        
    print('image flag')
    df_all['image'] = df_all['image'].map(lambda x: 1 if len(str(x)) >0 else 0)

    gc.collect()
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
    df_all = region_city(df_all)
    df_all = category_name(df_all)
    split_train_test(df_all, y)

if __name__ == "__main__":
    main()


