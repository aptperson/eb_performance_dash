import pandas as pd


from src.plot_utils import plot_ohlc, plot_ts
from src.plot_utils import plot_groupby_ts
from src.utils import gen_trading_dates, get_performance_data, insert_open_prices
from src.query_utils import get_df_from_s3, query_asx_table_date_range
from src.query_utils import get_pkl_from_s3


def prepare_table(df):
    table_columns = [{"name": i, "id": i} for i in df.columns]
    table_data = df.to_dict('records')
    return(table_columns, table_data)


def get_all_dash_datasets(N):
    today, signal_date, pnl_month_start, pnl_month_end = gen_trading_dates()

    # get all of the required data sets
    trade_universe_df, open_price_df, pnl_df = get_performance_data(signal_date, pnl_month_start, pnl_month_end)
    benchmark_data = get_benchmark_data()
    best_stratergy_parameters, backtest_pnl_series = get_backtest_data()


    ###### munge the data ######
    # portfolio level
    pnl_df = insert_open_prices(pnl_df, open_price_df)
    portfolio_benchmark_data = filter_benchmark_data(benchmark_data, [signal_date, min(pnl_month_end, today)])
    portfolio_plot_data = prepare_performance_df(pnl_df, trade_universe_df, portfolio_benchmark_data, N)
    portfolio_plot_data.reset_index(drop = True, inplace = True)

    # symbol level
    universe_plot_data = prepare_universe_df(pnl_df, trade_universe_df, N)
    universe_plot_data.reset_index(drop = True, inplace = True)

    # backtest level
    backtest_df = prepare_backtest_df(backtest_pnl_series, benchmark_data)

    return([portfolio_plot_data.to_json(),
            universe_plot_data.to_json(),
            backtest_df.to_json(),
            best_stratergy_parameters.to_json()])


def get_backtest_data():
    best_stratergy_parameters, backtest_pnl_series = get_backtest_pkl(fn = 'Stratergy_results_2020-09-16_insample')
    best_stratergy_parameters_oos, backtest_pnl_series_oos = get_backtest_pkl(fn = 'Stratergy_results_2020-09-16_outofsample')
    backtest_pnl_series = merge_backtest_samples(backtest_pnl_series, backtest_pnl_series_oos)
    best_stratergy_parameters = merge_backtest_parameters(best_stratergy_parameters, best_stratergy_parameters_oos)
    print(f'loaded the best stratergy parameters:')
    print(best_stratergy_parameters)
    return(best_stratergy_parameters, backtest_pnl_series)
    

def merge_backtest_parameters(best_stratergy_parameters, best_stratergy_parameters_oos):
    best_stratergy_parameters = pd.DataFrame(best_stratergy_parameters, index=[0])
    best_stratergy_parameters['data_split'] = 'train'
    best_stratergy_parameters_oos = pd.DataFrame(best_stratergy_parameters_oos, index=[1])
    best_stratergy_parameters_oos['data_split'] = 'val'
    return(pd.concat([best_stratergy_parameters, best_stratergy_parameters_oos]))


def merge_backtest_samples(backtest_pnl_series, backtest_pnl_series_oos):
    backtest_pnl_series_oos['portfolio_cpnl'] = backtest_pnl_series_oos.portfolio_cpnl * backtest_pnl_series.portfolio_cpnl.values[-1]
    return(pd.concat([backtest_pnl_series, backtest_pnl_series_oos], ignore_index=True))


def get_backtest_pkl(fn):
    best_stratergy_parameters = get_pkl_from_s3(fn)
    backtest_pnl_series = best_stratergy_parameters.pop('pnl_series')
    backtest_pnl_series['symbol'] = best_stratergy_parameters['stratergy']
    return(best_stratergy_parameters, backtest_pnl_series)


def prepare_backtest_df(backtest_pnl_series, benchmark_data, cpnl_col_name='portfolio_cpnl'):
    benchmark_data = filter_benchmark_data(benchmark_data,
                                            date_range=[backtest_pnl_series.date.min(),
                                            backtest_pnl_series.date.max()],
                                            symbol = 'XJOA',
                                            cpnl_col_name=cpnl_col_name,
                                            offset=0)
    benchmark_data['symbol'] = 'XJT'
    return(pd.concat([backtest_pnl_series[['symbol', 'date', cpnl_col_name]], benchmark_data], ignore_index=True))


def filter_benchmark_data(benchmark_data, date_range=None, symbol = 'XJT', cpnl_col_name='percent_returns', offset = 1):
    benchmark_symbol_df = benchmark_data.loc[benchmark_data.symbol == symbol].copy()
    mask = (benchmark_symbol_df.date >= date_range[0]) & (benchmark_symbol_df.date <= date_range[1])
    benchmark_symbol_df = benchmark_symbol_df.loc[mask]
    benchmark_symbol_df[cpnl_col_name] = benchmark_symbol_df.close / benchmark_symbol_df.close.values[0] - offset
    return(benchmark_symbol_df[['symbol', 'date', cpnl_col_name]])


def prepare_performance_df(pnl_df, trade_universe_df, benchmark_data, N):
    topN_data = filter_to_stratergy(pnl_df, trade_universe_df, N = 10)

    topN_plot_data = topN_data.groupby(['date']).percent_return.mean().to_frame('percent_returns').reset_index()
    topN_plot_data['symbol'] = f'#TOP {N} PORT_NO_STOP'
    
    topN_plot_data_stop = topN_data.groupby(['date']).stopped_return.mean().to_frame('percent_returns').reset_index()
    topN_plot_data_stop['symbol'] = f'#TOP {N} PORT_TRAILING_STOP'

    topN_frog_agg_data = filter_to_stratergy(pnl_df, trade_universe_df, N = 10, stratergy='frog_agg')
    topN_plot_data_frog_agg = topN_frog_agg_data.groupby(['date']).percent_return.mean().to_frame('percent_returns').reset_index()
    topN_plot_data_frog_agg['symbol'] = f'#TOP {N} PORT_FROG_AGG'

    plot_data = pd.concat([topN_plot_data, topN_plot_data_stop, topN_plot_data_frog_agg, benchmark_data])
    plot_data.sort_values('date', inplace = True)

    return(plot_data)


def prepare_universe_df(pnl_df, trade_universe_df, N = 10, stratergies = ['agg_mom', 'frog_agg']):
    
    universe_df=[]
    for stratergy in stratergies:
        universe_df.append(prepare_stratergy_df(pnl_df, trade_universe_df, N, stratergy))

    return(pd.concat(universe_df))


def prepare_stratergy_df(pnl_df, trade_universe_df, N = 10, stratergy='agg_mom'):
    stratergy_df = filter_to_stratergy(pnl_df, trade_universe_df, N, stratergy)

    stratergy_stopped_data = stratergy_df.groupby(['date', 'open_date']).stopped_return.mean().to_frame('stopped_return').reset_index()
    stratergy_stopped_data['symbol'] = f'#PORT_TS_{stratergy}'

    stratergy_raw_data = stratergy_df.groupby(['date', 'open_date']).percent_return.mean().to_frame('percent_return').reset_index()
    stratergy_raw_data['symbol'] = f'#PORT_{stratergy}'
    return(pd.concat([stratergy_df, stratergy_raw_data, stratergy_stopped_data]))


def filter_to_stratergy(pnl_df, trade_universe_df, N = 10, stratergy = 'agg_mom'):
    if stratergy == 'agg_mom':
        stratergy_symbols = trade_universe_df.sort_values('agg_mom').tail(N).symbol.values
    elif stratergy == 'frog_agg':
        frog_mom_df = trade_universe_df.loc[trade_universe_df.frog_momentum < 0]
        stratergy_symbols = frog_mom_df.sort_values('agg_mom').tail(N).symbol.values
    else:
        print(f'Unknown stratergy {stratergy}, defaulting to agg_mom stratergy')
        stratergy='agg_mom'
        stratergy_symbols = trade_universe_df.sort_values('agg_mom').tail(N).symbol.values
    stratergy_pnl_df = pnl_df.loc[pnl_df.symbol.isin(stratergy_symbols)].copy()
    stratergy_pnl_df['stratergy'] = stratergy
    stratergy_pnl_df.sort_values('date', inplace = True)
    return(stratergy_pnl_df)


def prepare_universe_table_data(trade_universe_df, stratergy):
    # df = filter_to_stratergy(pnl_df, trade_universe_df, N)
    df = trade_universe_df.loc[trade_universe_df.stratergy == stratergy]
    df = df.groupby('symbol')[['close', 'volume', 'high_water_mark', 'percent_return',
                               'historical_vol', 'stop_level',
                               'stopped', 'stopped_return']].last().reset_index().round(4).sort_values('percent_return')
    return(prepare_table(df))


def prepare_portfolio_table_data(df):
    df = pd.concat([
        # df.groupby('symbol').percent_returns.first().to_frame('open'),
                    df.groupby('symbol').percent_returns.last().to_frame('current'),
                    df.groupby('symbol').percent_returns.min().to_frame('min'),
                    df.groupby('symbol').percent_returns.max().to_frame('max')
                    ], axis = 1).round(4).reset_index()
    return(prepare_table(df))


def get_benchmark_data(benchmark_symbols=['XJT']):
    index_df = get_df_from_s3('benchmark_indices')
    index_df.sort_values('date', inplace = True)
    index_df = index_df.groupby(['date', 'symbol']).tail(1)
    return(index_df)


def period_signal(signal_date, universe = True):
    signal_df = get_df_from_s3(signal_date)
    print(f'signal_df shape {signal_df.shape}')
    if universe:
        trade_universe_df = query_asx_table_date_range(signal_date, signal_date, 'asx_trade_universe', verbose = 1)
        print(f'trade_universe_df shape {trade_universe_df.shape}')
    else:
        trade_universe_df = pd.DataFrame()
    return(signal_df, trade_universe_df)


def prepare_next_period_universe_table_data(next_trade_universe_df, N):
    print('cols:')
    print(next_trade_universe_df.columns)
    if 'frog_momentum' in next_trade_universe_df:
        cols = ['symbol', 'n12_skip1_returns', 'n9_skip1_returns', 'n6_skip1_returns',
                'n3_skip1_returns', 'n1_skip0_returns', 'months_positive', 'frog_momentum', 'historical_vol',
                'agg_mom']
    else:
        cols = ['symbol', 'n12_skip1_returns', 'n9_skip1_returns', 'n6_skip1_returns',
        'n3_skip1_returns', 'n1_skip0_returns', 'na_count', 'na_mean', 'historical_vol',
        'agg_mom']
    out = next_trade_universe_df[cols].sort_values('agg_mom').tail(20)
    out = out.round(4)
    return(prepare_table(out))