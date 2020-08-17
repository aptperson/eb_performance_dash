from datetime import datetime
import json

import boto3
import s3fs

import pandas as pd

from boto3.dynamodb.conditions import Key, Attr


db = boto3.resource('dynamodb', 'ap-southeast-2')

# table_field_dict = {'asx_dividends': ['asx_dividends', 'dividends'],
#                     'asx_splits': ['asx_splits', 'splits'],
#                     'asx_prices': ['asx_raw_prices_2', 'OHLCV'],
#                     'asx_trade_universe': ['asx_trade_universe', 'universe'],
#                     'asx_trade_open': ['asx_trade_open_prices', 'open_prices'],
#                     'asx_position_pnl': ['asx_position_pnl', 'pnl'],
#                     'asx_trade_series': ['asx_trade_series', 'series']}


table_field_dict = {'asx_dividends': ['asx_dividends', 'dividends'],
                    'asx_splits': ['asx_splits', 'splits'],
                    'asx_prices': ['asx_raw_prices_2', 'OHLCV'],
                    'asx_trade_universe': ['asx_trade_universe', 'universe'],
                    'asx_trade_open': ['asx_trade_open_prices', 'open_prices'],
                    'asx_trade_series': ['asx_trade_series', 'series'],
                    'asx_position_pnl': ['asx_position_pnl', 'pnl'],
                    'asx_trade_open_prices': ['asx_trade_open_prices', 'open_prices'],
                    'asx_position_pnl_2': ['asx_position_pnl_2', 'pnl'],
                    'asx_trade_open_prices_2': ['asx_trade_open_prices_2', 'open_prices']}


def get_df_from_s3(fn, bucket = 'signallambda-dev-large-df-storage/'):
    s3 = s3fs.S3FileSystem(anon=False)
    s3_files = s3.ls(bucket)
    signal_file = [fn in b for b in s3_files]
    signal_file = [i for (i, v) in zip(s3_files, signal_file) if v]
    if len(signal_file) > 0:
        print('attempting to read: {}'.format(signal_file[0]))
        try:
            with s3.open(signal_file[0], 'rb') as f:
                signal_df = pd.read_csv(f)
            print('much_read success')
        except Exception as e:
            print(repr(e))
    else:
        print('could not find file with string {}\nfiles in bucket:{}'.format(fn, s3_files))
        signal_df = pd.DataFrame()
    return(signal_df)



def query_asx_table_date_range(f, t, table, verbose = 1, symbols = None):
    if verbose > 0:
        start_time = datetime.now()
    table_name, table_field = table_field_dict[table]
    b_dates = pd.DataFrame({'date': pd.bdate_range(f, t)})
    b_dates = b_dates.date.apply(lambda x: str(x)[:10])
    result_list = []
    N=len(b_dates)
    if verbose > 0:
        print('fetching data for date range: {} to {}'.format(b_dates.min(), b_dates.max()))
    for i, date in enumerate(b_dates):
        if (verbose > 0) and (i % 10 == 0):
            print('fetching data for {}/{}, {}'.format(i, N, date))
        result_df = query_result_to_df(date, db.Table(table_name), table_field)
        if symbols is not None:
            result_df = result_df.loc[result_df.symbol.isin(symbols)]
        result_list.append(result_df)            
    raw_data_df = pd.concat(result_list, ignore_index = True)

    # raw_data_df.rename({'last': 'close'}, axis = 1, inplace = True)
    if verbose > 0:
        print('query took {}s'.format((datetime.now() - start_time).seconds))
    return(raw_data_df)
    

def column_rename(df, table_name):
    if table_name == 'asx_splits':
        if 'split_date' in df:
            df.drop('split_date', axis = 1, inplace = True)
        df.rename({'split ratio': 'split_ratio', 'date': 'split_date'}, axis = 1, inplace = True)
    elif table_name == 'asx_raw_prices_2':
        # if 'last' in df:
        df.rename({'last': 'close'}, axis = 1, inplace = True)
    else:
        pass
    return(df)

    
def query_result_to_df(date, table_name, field):
    result = table_name.query(KeyConditionExpression=Key('date').eq(date))
    if result['Count'] > 0:
        result_df = pd.DataFrame(json.loads(result['Items'][0][field]))
        result_df.columns = [c.lower() for c in result_df.columns]
        result_df = column_rename(result_df, table_name.name)
        result_df['date'] = date
    else:
        result_df = pd.DataFrame()
    return(result_df)


if __name__ == "__main__":
    result = query_asx_table_date_range('2019', '2020', 'asx_prices')
    print(result.shape)
    print(result.head())