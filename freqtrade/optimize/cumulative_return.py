#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
累积收益计算模块

该模块提供计算和分析回测交易累积收益的功能
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from freqtrade.data.btanalysis import load_backtest_data
from freqtrade.constants import Config, DATETIME_PRINT_FORMAT

logger = logging.getLogger(__name__)

def load_backtest_price_data(config: Config) -> Dict[str, pd.DataFrame]:
    """
    从数据目录加载价格数据

    :param config: FreqTrade 配置对象
    :return: 包含每个交易对价格数据的字典
    """
    from freqtrade.optimize.backtesting import Backtesting
    from freqtrade.exchange import Exchange
    from freqtrade.resolvers import ExchangeResolver

    exchange = ExchangeResolver.load_exchange(config)
    backtesting = Backtesting(config, exchange)
    data, timerange = backtesting.load_bt_data()

    logger.info(f"已加载 {len(data)} 个交易对的价格数据")
    return data

def get_result_dataframe(backtest_data: Dict) -> pd.DataFrame:
    """
    从回测结果中提取交易数据框

    :param backtest_data: 回测结果数据字典
    :return: 包含交易信息的DataFrame
    """
    if not backtest_data or 'results' not in backtest_data:
        logger.warning("回测数据不包含结果信息")
        return pd.DataFrame()
    return backtest_data['results']

def get_starting_balance(backtest_data: Dict) -> float:
    """
    获取回测的初始余额

    :param backtest_data: 回测结果数据字典
    :return: 初始余额
    """
    if not backtest_data or 'config' not in backtest_data:
        logger.warning("回测数据不包含配置信息")
        return 0.0
    config = backtest_data['config']
    return float(config.get('dry_run_wallet', 1000.0))

def calculate_positions(
    trades_df: pd.DataFrame,
    price_data: Dict[str, pd.DataFrame],
    initial_balance: float
) -> pd.DataFrame:
    """
    计算持仓情况和账户价值变化，保持原始数据的时间粒度

    :param trades_df: 交易结果数据框
    :param price_data: 价格数据字典
    :param initial_balance: 初始账户余额
    :return: 包含持仓和账户价值的DataFrame
    """
    if trades_df.empty:
        logger.warning("没有交易数据，无法计算持仓")
        return pd.DataFrame()

    # 确保日期列为datetime格式
    trades_df['open_date'] = pd.to_datetime(trades_df['open_date'])
    trades_df['close_date'] = pd.to_datetime(trades_df['close_date'])

    # 使用价格数据的原始时间索引
    if price_data:
        first_pair = list(price_data.keys())[0]
        date_range_df = price_data[first_pair].copy()
        date_range_df['date'] = pd.to_datetime(date_range_df['date'])
        date_range = date_range_df['date']
    else:
        start_date = trades_df['open_date'].min()
        end_date = trades_df['close_date'].max()
        date_range = pd.date_range(start=start_date, end=end_date, freq='1H')

    positions = pd.DataFrame(index=date_range)
    positions.index.name = 'date'
    # 转换成时间戳
    positions['time'] = positions.index.astype('int64') // 10 ** 6
    positions['balance'] = initial_balance

    # 为所有交易对初始化持仓列（不仅仅是交易中的交易对）
    for pair in price_data.keys():
        positions[f'{pair}_position'] = 0.0
        positions[f'{pair}_value'] = 0.0
    positions['total_position_value'] = 0.0

    # 处理每笔交易
    for _, trade in trades_df.iterrows():
        #print(trade['orders'])
        pair = trade['pair']
        open_date = pd.Timestamp(trade['open_date'])
        open_idx = positions.index.get_indexer([open_date], method='nearest')[0]

        # 开仓
        stake_amount = float(trade['stake_amount'])
        positions.iloc[open_idx:, positions.columns.get_loc('balance')] -= stake_amount

        amount = float(trade['amount'])
        if trade.get('is_short', False):
            amount = -amount
        positions.iloc[open_idx:, positions.columns.get_loc(f'{pair}_position')] += amount

        # 平仓
        if pd.notna(trade['close_date']):
            close_date = pd.Timestamp(trade['close_date'])
            profit = float(trade['profit_abs'])
            close_idx = positions.index.get_indexer([close_date], method='nearest')[0]
            positions.iloc[close_idx:, positions.columns.get_loc('balance')] += stake_amount + profit
            positions.iloc[close_idx:, positions.columns.get_loc(f'{pair}_position')] -= amount

    # 计算收盘价和BNH收益率
    for pair in price_data.keys():
        pair_price_data = price_data[pair].copy()
        pair_price_data['date'] = pd.to_datetime(pair_price_data['date'])
        pair_price_data = pair_price_data.set_index('date')

        # 使用第一个有效收盘价作为初始价格
        initial_close = pair_price_data['close'].iloc[0] if not pair_price_data.empty else None
        positions[f'{pair}_close'] = np.nan

        for i, timestamp in enumerate(positions.index):
            closest_date = get_closest_date(pair_price_data.index, timestamp)
            if closest_date is not None:
                price = pair_price_data.loc[closest_date, 'close']
                positions.iloc[i, positions.columns.get_loc(f'{pair}_close')] = price
                positions.iloc[i, positions.columns.get_loc(f'{pair}_value')] = (
                    positions.iloc[i, positions.columns.get_loc(f'{pair}_position')] * price
                )

        if initial_close and initial_close > 0:
            positions[f'{pair}_bh_return'] = (positions[f'{pair}_close'] / initial_close) - 1
            positions[f'{pair}_bh_return_pct'] = positions[f'{pair}_bh_return'] * 100

    # 计算总持仓价值和账户价值
    position_value_columns = [col for col in positions.columns if col.endswith('_value')]
    positions['total_position_value'] = positions[position_value_columns].sum(axis=1)
    positions['total_account_value'] = positions['balance'] + positions['total_position_value']

    # 计算策略累计收益率
    positions['cumulative_return'] = (positions['total_account_value'] / initial_balance) - 1
    positions['cumulative_return_pct'] = positions['cumulative_return'] * 100

    return positions

def get_closest_date(date_index: pd.DatetimeIndex, target_date: pd.Timestamp) -> Optional[pd.Timestamp]:
    """
    在日期索引中找到最接近目标日期的日期

    :param date_index: 日期索引
    :param target_date: 目标日期
    :return: 最接近的日期，如果没有找到则返回None
    """
    if len(date_index) == 0:
        return None

    future_dates = date_index[date_index >= target_date]
    past_dates = date_index[date_index <= target_date]

    future_date = future_dates[0] if len(future_dates) > 0 else None
    past_date = past_dates[-1] if len(past_dates) > 0 else None

    if future_date and past_date:
        return future_date if (future_date - target_date) < (target_date - past_date) else past_date
    return future_date or past_date

def analyze_backtest_result(
    backtest_result: Dict,
    price_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    分析回测结果，计算累积收益

    :param backtest_result: 回测结果数据
    :param price_data: 价格数据字典
    :return: 每日账户价值和收益率DataFrame
    """
    trades_df = get_result_dataframe(backtest_result)
    if trades_df.empty:
        logger.warning("没有交易数据，无法分析回测结果")
        return pd.DataFrame()

    initial_balance = get_starting_balance(backtest_result)
    if initial_balance <= 0:
        logger.warning("初始余额无效，使用默认值1000")
        initial_balance = 1000.0

    return calculate_positions(trades_df, price_data, initial_balance)

if __name__ == "__main__":
    from freqtrade.configuration import Configuration
    import sys

    if len(sys.argv) < 2:
        print("用法: python cumulative_return.py <配置文件路径>")
        sys.exit(1)

    config = Configuration.from_files([sys.argv[1]])
    price_data = load_backtest_price_data(config.get_config())

    print(f"已加载的交易对: {list(price_data.keys())}")
    if price_data:
        first_pair = list(price_data.keys())[0]
        #print(f"\n{first_pair} 的前5行数据:")
        #print(price_data[first_pair].head())
