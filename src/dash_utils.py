import pandas as pd


from src.plot_utils import plot_ohlc, plot_ts
from src.plot_utils import plot_groupby_ts
from src.utils import gen_trading_dates, get_performance_data, insert_open_prices
from src.query_utils import get_df_from_s3


def prepare_table(df):
    table_columns = [{"name": i, "id": i} for i in df.columns]
    table_data = df.to_dict('records')
    return(table_columns, table_data)


def prepare_performance_df(pnl_df, trade_universe_df, N, date_range):
    plot_data = pnl_df.groupby(['date']).percent_return.mean().to_frame('percent_returns').reset_index()
    plot_data['symbol'] = '#PORT_NO_STOP'
    # performance_fig_no_stop = plot_ts(plot_data, x_col='date', y_col = 'percent_returns', log=False)
    
    plot_data_stop = pnl_df.groupby(['date']).stopped_return.mean().to_frame('percent_returns').reset_index()
    plot_data_stop['symbol'] = '#PORT_TRAILING_STOP'

    topN_data = filter_to_top_N(pnl_df, trade_universe_df, N = 10)

    topN_plot_data = topN_data.groupby(['date']).percent_return.mean().to_frame('percent_returns').reset_index()
    topN_plot_data['symbol'] = f'#TOP {N} PORT_NO_STOP'
    # performance_fig_no_stop = plot_ts(plot_data, x_col='date', y_col = 'percent_returns', log=False)
    
    topN_plot_data_stop = topN_data.groupby(['date']).stopped_return.mean().to_frame('percent_returns').reset_index()
    topN_plot_data_stop['symbol'] = f'#TOP {N} PORT_TRAILING_STOP'

    benchmark_data = get_benchmark_data(date_range)

    plot_data = pd.concat([plot_data, plot_data_stop, topN_plot_data, topN_plot_data_stop, benchmark_data])
    plot_data.sort_values('date', inplace = True)
    print(plot_data)

    return(plot_data)


def prepare_universe_df(pnl_df, trade_universe_df, N = 10):
    plot_data = filter_to_top_N(pnl_df, trade_universe_df, N)
    agg_data = plot_data.groupby(['date', 'open_date']).stopped_return.mean().to_frame('percent_return').reset_index()
    agg_data['symbol'] = '#PORT_TRAILING_STOP'
    plot_data['percent_return'] = plot_data.stopped_return
    return(pd.concat([plot_data, agg_data]))


def filter_to_top_N(pnl_df, trade_universe_df, N = 10):
    topN = trade_universe_df.tail(N).symbol.values
    topN_data = pnl_df.loc[pnl_df.symbol.isin(topN)].copy() #.groupby(['date', 'open_date']).percent_return.mean().to_frame('percent_returns').reset_index()
    topN_data.sort_values('date', inplace = True)
    return(topN_data)


def prepare_universe_table_data(pnl_df, trade_universe_df, N):
    df = filter_to_top_N(pnl_df, trade_universe_df, N)
    df = df.groupby('symbol')[['close', 'volume', 'high_water_mark',
                               'historical_vol', 'stop_level',
                               'stopped', 'stopped_return']].last().reset_index().round(4).sort_values('stopped')
    return(prepare_table(df))


def prepare_portfolio_table_data(df):
    df = pd.concat([
        # df.groupby('symbol').percent_returns.first().to_frame('open'),
                    df.groupby('symbol').percent_returns.last().to_frame('current'),
                    df.groupby('symbol').percent_returns.min().to_frame('min'),
                    df.groupby('symbol').percent_returns.max().to_frame('max')
                    ], axis = 1).round(4).reset_index()
    return(prepare_table(df))


def get_benchmark_data(date_range):
    index_df = get_df_from_s3('benchmark_indices')
    index_df.sort_values('timestamp', inplace = True)
    index_df = index_df.groupby(['date', 'symbol']).tail(1)
    mask = (index_df.date >= date_range[0]) & (index_df.date <= date_range[1])
    benchmark_data = index_df.loc[mask]
    benchmark_data['return'] = benchmark_data.groupby('symbol').close.pct_change()
    benchmark_data.fillna(0, inplace = True)
    benchmark_data['percent_returns'] = 1 + benchmark_data['return']
    benchmark_data['percent_returns'] = benchmark_data.groupby('symbol').percent_returns.cumprod() -1
    return(benchmark_data[['date', 'symbol', 'percent_returns']])