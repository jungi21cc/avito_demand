import pandas as pd
import numpy as np
import gc
import string
import scipy

def load_data():
    print('load data')
    used_cols = ['item_id', 'user_id']
    train = pd.read_csv('../train.csv', usecols=used_cols)
    train_active = pd.read_csv('../train_active.csv', usecols=used_cols)
    test = pd.read_csv('../test.csv', usecols=used_cols)
    test_active = pd.read_csv('../test_active.csv', usecols=used_cols)
    train_periods = pd.read_csv('../periods_train.csv', parse_dates=['date_from', 'date_to'])
    test_periods = pd.read_csv('../periods_test.csv', parse_dates=['date_from', 'date_to'])

    return train, test, train_active, test_active, train_periods, test_periods

def concat_all_data(train, test, train_active, test_active, train_periods, test_periods):
    print('concat all data')
    all_samples = pd.concat([
        train,
        train_active,
        test,
        test_active
    ]).reset_index(drop=True)
    all_samples.drop_duplicates(['item_id'], inplace=True)

    del train_active, test_active
    gc.collect()

    all_periods = pd.concat([
        train_periods,
        test_periods
    ])

    del train_periods, test_periods
    gc.collect()

    all_periods['days_up'] = all_periods['date_to'].dt.dayofyear - all_periods['date_from'].dt.dayofyear

    return all_samples, all_periods
    
def days_up_groupby(all_samples, all_periods):
    print("days up groupby")
    gp = all_periods.groupby(['item_id'])[['days_up']]
    gp_df = pd.DataFrame()
    gp_df['days_up_sum'] = gp.sum()['days_up']
    gp_df['times_put_up'] = gp.count()['days_up']
    gp_df.reset_index(inplace=True)
    gp_df.rename(index=str, columns={'index': 'item_id'})

    all_periods.drop_duplicates(['item_id'], inplace=True)
    all_periods = all_periods.merge(gp_df, on='item_id', how='left')

    del gp, gp_df
    gc.collect()

    return all_samples, all_periods

def periods_merge(all_samples, all_periods):
    print('periods merge')
    all_periods = all_periods.merge(all_samples, on='item_id', how='left')
    gp = all_periods.groupby(['user_id'])[['days_up_sum', 'times_put_up']].mean().reset_index() \
        .rename(index=str, columns={
            'days_up_sum': 'avg_days_up_user',
            'times_put_up': 'avg_times_up_user'
        })

    n_user_items = all_samples.groupby(['user_id'])[['item_id']].count().reset_index() \
        .rename(index=str, columns={
            'item_id': 'n_user_items'
        })
    return gp, n_user_items


def user_id_merge(gp, n_user_items):
    print('user id merge')
    gp = gp.merge(n_user_items, on='user_id', how='outer')
    gp.to_csv('../gp.csv', index=False)
    
def main():
    train, test, train_active, test_active, train_periods, test_periods = load_data()
    all_samples, all_periods = concat_all_data(train, test, train_active, test_active, train_periods, test_periods)
    all_samples, all_periods = days_up_groupby(all_samples, all_periods)
    gp, n_user_items = periods_merge(all_samples, all_periods)
    user_id_merge(gp, n_user_items)
    
if __name__ == "__main__":
    main()