from datetime import datetime, timedelta
from pytz import timezone

import boto3
import numpy as np
import pandas as pd

from src.query_utils import query_asx_table_date_range, get_df_from_s3

asx_closed_calendar = pd.DataFrame({'date': ['2019-01-01', '2019-01-28',
                                    '2019-04-19', '2019-04-22',
                                    '2019-04-25', '2019-06-10',
                                    '2019-12-25', '2019-12-26',
                                    '2020-01-01', '2020-01-27',
                                    '2020-04-10', '2020-04-13',
                                    '2020-04-25', '2020-06-08',
                                    '2020-12-25', '2020-12-28']})


def prepare_table(df):
    table_columns = [{"name": i, "id": i} for i in df.columns]
    table_data = df.to_dict('records')
    return(table_columns, table_data)


def prepare_plot_data(df):
    plot_data = df.copy() #.groupby(['date', 'open_date']).percent_return.mean().to_frame('percent_returns').reset_index()
    plot_data.sort_values('date', inplace = True)
    agg_data = plot_data.groupby(['date', 'open_date']).percent_return.mean().to_frame('percent_return').reset_index()
    agg_data['symbol'] = '#PORTFOLIO#'
    return(pd.concat([plot_data, agg_data]))

def get_event_today(event, date_format):
    try:
        today = event['date']
    except:
        print('CloudWatch triggering, event:')
        print(event)    # today = '2020-07-21'
        today = get_current_date_tz(out_format = date_format)
    return(today)


def gen_trading_dates(date_format = '%Y-%m-%d'):
    today = get_current_date_tz(out_format = None)
    current_year = today.year
    current_month = today.month

    pnl_month_start = gen_first_trading_day_month(current_year, current_month, '%Y-%m-%d').date.values[0]
    print(f'today: {today}\nyear: {current_year}\nmonth: {current_month}')
    print(f'Start of pnl month start: {pnl_month_start}')
          
    year, month = gen_last_month_year(today)
    print(f'Signal:\nyear: {year}\nmonth: {month}')

    signal_date = get_valid_trade_open_date(year = year, month = month).date.values[0]
    signal_date = str(signal_date)[:10]
    print(f'getting open prices for {signal_date}')
    
    pnl_month_end = get_next_months_last_trade_date(signal_date, date_format).date.values[0]
    print(f'End of pnl month {pnl_month_end}')

    today = str(today)[:10]
    
    return(today, signal_date, pnl_month_start, pnl_month_end)


def insert_open_prices(pnl_df, open_price_df):
    
    op_df = open_price_df.copy()
    op_df['percent_return'] = 0
    op_df['stopped_return'] = 0
    op_df['dollor_return'] = 0
    op_df['open_date'] = op_df.date.values
    op_df['trade_open'] = op_df.close.values
    op_df['stopped'] = False
    pnl_df = pd.concat([pnl_df, op_df])
    pnl_df.sort_values('date', inplace = True)
    pnl_df.reset_index(inplace = True, drop = True)
    return(pnl_df)


def get_performance_data(signal_date, pnl_month_start, pnl_month_end ):

    trade_universe_df = query_asx_table_date_range(signal_date, signal_date, 'asx_trade_universe', verbose = 1)
    open_price_df = query_asx_table_date_range(signal_date, signal_date, 'asx_trade_open_prices', verbose = 1)
    pnl_df = query_asx_table_date_range(pnl_month_start, pnl_month_end, 'asx_position_pnl', verbose = 1)
    print(f'{trade_universe_df.shape[0]} in trade universe df')
    print(f'{open_price_df.shape[0]} in open price df')
    print(f'{pnl_df.shape[0]} in pnl df')

    return(trade_universe_df, open_price_df, pnl_df)


def performance_caluclation(open_price_df, current_price_df):
    open_price_df.rename({'date': 'open_date', 'close': 'trade_open'}, axis = 1, inplace = True)
    current_price_df = pd.merge(left = open_price_df[['open_date', 'trade_open', 'symbol']],
                                left_on = 'symbol',
                                right = current_price_df,
                                right_on = 'symbol',
                                how = 'left')
    current_price_df['dollor_return'] = current_price_df.close - current_price_df.trade_open
    current_price_df['percent_return'] = current_price_df.dollor_return / current_price_df.trade_open
    return(current_price_df)


def put_to_table(put_data, table):
    print('putting data to table: {}'.format(table.name))
    try:
        table.put_item(
            Item=put_data
        )
        print('much put success')
        return(1)
    except:
        print('failure put much')
        return(0)


def gen_first_trading_day_month(year, month, out_format = None):

    open_dates = get_asx_open_dates(out_format, True)
    first_trading_day_month = open_dates.groupby(['year', 'month']).first().reset_index()
    mask = (first_trading_day_month.year == year) & (first_trading_day_month.month == month)

    return(first_trading_day_month.loc[mask])


def get_asx_open_dates(out_format=None, meta_data=False):

    bdates = pd.DataFrame({'date': pd.bdate_range('2019', '2021')})

    if meta_data:
        bdates['day'] = bdates.date.dt.day
        bdates['weekday'] = bdates.date.dt.weekday
        bdates['month'] = bdates.date.dt.month
        bdates['year'] = bdates.date.dt.year

    asx_closed = asx_closed_calendar.copy(deep=True)
    asx_closed['date'] = asx_closed.date.apply(datetime.strptime, args = ['%Y-%m-%d'])

    mask = ~bdates.date.isin(asx_closed.date)
    bdates = bdates.loc[mask]
    
    if out_format is not None:
        bdates['date'] = bdates.date.apply(datetime.strftime,
                                            args = [out_format])

    return(bdates)


def is_asx_open(date, date_format):
    open_dates = get_asx_open_dates(date_format).date.values
    if date in open_dates:
        return(True)
    else:
        return(False)


def gen_last_trading_day_month(out_format = None):

    open_dates = get_asx_open_dates(out_format, True)
    last_trading_day_month = open_dates.groupby(['year', 'month']).last()

    return(last_trading_day_month.reset_index())


def get_valid_trade_open_date(date = None, date_format=None, year = None, month = None):
    trade_dates = gen_last_trading_day_month(date_format)
    if date_format is not None:
        date = datetime.strptime(date, date_format)
    if date is not None:
        mask = (trade_dates.month == date.month) & (trade_dates.year == date.year)
    elif year is not None:
        mask = (trade_dates.month == month) & (trade_dates.year == year)
    else:
        print('you must pass either [date] or [year, month]')
    return(trade_dates.loc[mask])


def get_next_months_last_trade_date(date, date_format=None):
    trade_dates = gen_last_trading_day_month(date_format)
    if date_format is not None:
        date = datetime.strptime(date, date_format)

    next_month = gen_next_month(date.month)
    if next_month == 1:
        print('year has changed, need a new calendar')
    mask = (trade_dates.month == next_month) & (trade_dates.year == date.year)
    return(trade_dates.loc[mask])


def gen_next_month(month):
    m = month % 12
    return(m + 1)


def gen_last_month_year(date, date_format=None):
    if date_format is not None:
        date = datetime.strptime(date, date_format)
    month = date.month
    year = date.year
    if month == 1:
        month = 12
        year = year - 1
    else:
        month = month - 1
    return(year, month)


def is_valid_trade_open_date(date, date_format = None):
    last_b_day_month = get_valid_trade_open_date(date, date_format)
    if last_b_day_month.date.values[0] == date:
        out = True
    else:
        out = False
    return(out)


def get_current_date_tz(tz = 'Australia/Sydney', out_format = "%Y-%m-%d"):
    out = datetime.now(timezone(tz))
    if out_format is not None:
        out = datetime.strftime(out, out_format)
    return(out)


def previous_date(date, date_format, N):
     if isinstance(date, datetime):
          return(date - timedelta(days = N))
     else:
          return(datetime.strptime(date, date_format) - timedelta(days = N))


def datetime_as_str(date, date_format):
    return(datetime.strftime(date, date_format))


def corporate_action_adjust_cols(df, col):
    if not isinstance(col, list):
        col = [col]
    if ('adj_factor' in df) and ('split_adj_factor' in df):
        adj_factor = df.adj_factor * df.split_adj_factor
    elif 'split_adj_factor' in df:
        adj_factor = df.split_adj_factor
    elif 'adj_factor' in df:
        adj_factor = df.adj_factor
    else:
        adj_factor = 1
    
    for c in col:
        r_c = 'raw_' + c
        df.rename({c: r_c}, inplace = True, axis = 1)
        try:
            df[c] = df[r_c].astype(np.float64) * adj_factor
        except:
            print('string at\n')
            print([i for i,row in df[[r_c]].iterrows() if isinstance(row[r_c],str)])
    return df


def period_N_SKIP_ret(period_close, N, SKIP):

    skip_close = period_close.groupby(
        'symbol').close.shift(SKIP)
    N_close = period_close.groupby(
        'symbol').close.shift(N)

    period_close['skip_close'] = skip_close
    period_close['N_close'] = N_close
    feature_data = period_close.groupby('symbol').tail(1)
    
    return(feature_data.skip_close / feature_data.N_close - 1)



def query_join_int_symbols(db, df, verbose = 0):
    int_symbols_df = conn_select_to_df(db,
                           'SELECT Symbol, SymbolInt FROM AsxListed')
    if verbose > 0:
        print('change the case on the SymbolInt db column names')
    int_symbols_df.rename({'SymbolInt': 'symbol_int', 'Symbol': 'symbol'}, inplace = True, axis = 1)
    return(join_int_symbols(df, int_symbols_df))


def join_int_symbols(price_df, int_symbols_df):
    return(pd.merge(left = price_df,
                   left_on = ['symbol'],
                   right = int_symbols_df[['symbol', 'symbol_int']],
                   right_on = ['symbol'],
                   how = 'left'))


def fill_forward(df):
    # df.rename({'close': 'raw_close', 'low': 'raw_low'}, inplace = True, axis = 1)
    price_ff = df.groupby('symbol')[['open', 'close', 'low']].fillna(method = 'ffill')
    df['close'] = price_ff['close']
    df['low'] = price_ff['low']
    df['open'] = price_ff['open']
    return df


def expand_dates_online(df):
    # date_cols = ['date', 'year', 'month', 'day']
    date_cols = ['date']
    all_dates = df.groupby(date_cols).size().reset_index()[date_cols]
    all_symbols = df.symbol.unique()
    all_dates_df = []
    for s in all_symbols:
        symbol_dates = all_dates.copy()
        symbol_dates['symbol'] = s
        all_dates_df.append(symbol_dates)
        
    all_dates_df = pd.concat(all_dates_df)

    join_cols = ['date', 'symbol']
    df = pd.merge(left = all_dates_df,
            left_on = join_cols,
            right = df,
            right_on = join_cols,
            how = 'left')
    return(df)


def percent_monthly_returns(df):
    month_close = df.groupby(['symbol', 'year', 'month'])['close'].last().to_frame('month_close')
    lag_month_close = month_close.groupby(['symbol'])['month_close'].shift(1).to_frame('lag_month_close')
    month_close = pd.concat([month_close, lag_month_close], axis = 1)
    month_close['monthly_r'] = month_close['month_close'] / month_close['lag_month_close'] - 1
    df = pd.merge(left = df,
            left_on = ['symbol', 'year', 'month'],
            right = month_close['monthly_r'],
            right_index = True,
            how = 'left')
    return df


def add_week_year_info(raw_stock_data, mask = None, date_col = 'index'):
    
    if not isinstance(mask, pd.Series):
        mask = raw_stock_data.index.isin(raw_stock_data.index)
    
    if date_col == 'index':
        raw_stock_data.loc[mask, 'day'] = raw_stock_data[mask].index.day # - raw_stock_data.date.astype('datetime64[M]') + 1
        raw_stock_data.loc[mask,'month'] = raw_stock_data[mask].index.month #astype('datetime64[M]').astype(int) % 12 + 1
        raw_stock_data.loc[mask, 'year'] = raw_stock_data[mask].index.year  #.astype('datetime64[Y]').astype(int) + 1970
        raw_stock_data.loc[mask, 'weekday'] = raw_stock_data[mask].index.weekday
        raw_stock_data.loc[mask, 'week'] = raw_stock_data[mask].index.week
    elif date_col == 'dt':
        raw_stock_data.loc[mask, 'day'] = raw_stock_data[mask].date.dt.day # - raw_stock_data.date.astype('datetime64[M]') + 1
        raw_stock_data.loc[mask,'month'] = raw_stock_data[mask].date.dt.month #astype('datetime64[M]').astype(int) % 12 + 1
        raw_stock_data.loc[mask, 'year'] = raw_stock_data[mask].date.dt.year  #.astype('datetime64[Y]').astype(int) + 1970
        raw_stock_data.loc[mask, 'weekday'] = raw_stock_data[mask].date.dt.weekday
        raw_stock_data.loc[mask, 'week'] = raw_stock_data[mask].date.dt.week
    else:
        print('incorectly specified date col')
    
    return raw_stock_data

def add_date_information(raw_stock_data, meta_data=None, date_format = '%Y%m%d', date_col='date'):

    if meta_data:
        raw_stock_data.index = pd.DatetimeIndex(
            raw_stock_data.date, tz = meta_data['5. Time Zone']).tz_convert('Australia/Sydney')
        if 'raw_date' not in raw_stock_data:
            raw_stock_data.rename(columns = {date_col: "raw_date"}, inplace = True)
        raw_stock_data = add_week_year_info(raw_stock_data, date_col = 'index')
    else:
        if 'raw_date' not in raw_stock_data:
            raw_stock_data.rename(columns = {date_col: "raw_date"}, inplace = True)
        raw_stock_data['date'] = pd.to_datetime(raw_stock_data.raw_date, format = date_format)
        raw_stock_data = add_week_year_info(raw_stock_data, date_col = 'dt')

    raw_stock_data.sort_values('date', inplace = True)

    return raw_stock_data


def read_format_dividends(date_range = None, verbose = 0, local = False):
    # fn = 'meta_data/asx_dividends_clean_F1996T2020.csv'):
    # dividend_df = pd.read_csv(fn)
    # dividend_df['date'] = pd.to_datetime(dividend_df['Ex-dividend date'], format = '%d %b %Y')
    # dividend_df.rename({'code_clean': 'symbol'}, inplace = True, axis = 1)

    dividend_df = local_or_query(date_range, 'asx_dividends', verbose, local)

    # drop a dividend, this is a hack, need to think of a better way to do this
    # MPower Group Limited	MPR	(MPR)	ORDINARY FULLY PAID
    mask = (dividend_df.symbol == 'MPR') & (dividend_df.dividend == 1.43) & (dividend_df.date == '20070625')
    dividend_df = dividend_df.loc[~mask]

    # dividend_df = query_join_int_symbols(asx_db, dividend_df)
    # dividend_df['date'] = pd.to_datetime(dividend_df.date, format = '%Y-%m-%d')
    return(dividend_df)



def local_or_query(date_range, table, verbose = 1, local = False, symbols = None):
    query_aws = True
    if local:
        fn = 'data/{}.csv'.format(table)
        try:
            data_df = pd.read_csv(fn)
            set(data_df.symbol.unique()).union
            print('local copy found at {}'.format(fn))
            query_aws = False
        except:
            print('local file does not exist, querying')
            
    if query_aws:
        data_df = query_asx_table_date_range(f = date_range[0], t = date_range[1], table = table, verbose = verbose, symbols = symbols)
        if local:
            print('saving local copy to {}'.format(fn))
            data_df.to_csv(fn, index = False)
            print('done')

    return(data_df)


def local_or_s3(fn, bucket = 'signallambda-dev-large-df-storage/', local = True):
    query_aws = True
    if local:
        local_fn = 'data/{}.csv'.format(fn)
        try:
            data_df = pd.read_csv(local_fn)
            set(data_df.symbol.unique()).union
            print('local copy found at {}'.format(local_fn))
            query_aws = False
        except:
            print('local file does not exist, querying')
            
    if query_aws:
        data_df = get_df_from_s3(fn, bucket)
        if local:
            print('saving local copy to {}'.format(local_fn))
            data_df.to_csv(local_fn, index = False)
            print('done')

    return(data_df)



def read_format_splits(date_range = None, verbose = 1, local = False):

    splits_df = local_or_query(date_range, 'asx_splits', verbose, local)

    # splits_df = query_asx_table_date_range(f = date_range[0], t = date_range[1], table = 'asx_dividends', verbose = verbose)

    if not splits_df.empty:
        splits_df['date'] = pd.to_datetime(splits_df['split_date'])
        splits_df['symbol'] = [sym[1].split(')')[0] for sym in splits_df.company.str.split('(')]
        splits_df['symbol'] = [sym[0:3] for sym in splits_df.symbol]
        splits = splits_df['split_ratio'].str.split(':')
        splits = np.array([[pd.to_numeric(s[1]), pd.to_numeric(s[0]), ] for s in splits.values])
        splits_df['split_l'] = splits[:,1]
        splits_df['split_r'] = splits[:,0]
        out_cols = ['date', 'split_date', 'symbol', 'split_l',
                    'split_r', 'manual_lookup']
        splits_df = splits_df[out_cols].drop_duplicates()
    return(splits_df)
