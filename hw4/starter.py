#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import sys

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


categorical = ['PUlocationID', 'DOlocationID']

def read_data(year, month):
    filename = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet'

    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    return df


def make_prediction(df):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print(f'Mean Prediction: {y_pred.mean()}')

    df_result = pd.DataFrame()
    
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    return df_result


def save_results(df_result, output_file = "results_df.parquet"):
        df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


if __name__ == '__main__':
    year = int(sys.argv[1]) # 2021
    month = int(sys.argv[2]) # 3

    df = read_data(year, month)
    df_result = make_prediction(df)
    save_results(df_result)














