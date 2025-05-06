#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
累积收益计算模块

该模块提供计算和分析回测交易累积收益的功能
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
from numba import njit, jit

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

@njit
def _find_closest_index(timestamps: np.ndarray, target_ts: int) -> int:
    """
    用Numba优化的函数，找到时间戳数组中最接近目标时间戳的索引
    
    :param timestamps: 时间戳数组
    :param target_ts: 目标时间戳
    :return: 最接近的索引
    """
    if len(timestamps) == 0:
        return -1
    
    # 使用二分查找找到最接近的索引
    left, right = 0, len(timestamps) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if timestamps[mid] == target_ts:
            return mid
        elif timestamps[mid] < target_ts:
            left = mid + 1
        else:
            right = mid - 1
    
    # 如果没有找到精确匹配，找到最接近的
    if left >= len(timestamps):
        return right
    if right < 0:
        return left
    
    # 返回更接近的索引
    if abs(timestamps[left] - target_ts) < abs(timestamps[right] - target_ts):
        return left
    return right

@njit
def _update_positions_numba(
    balance_array: np.ndarray,
    position_arrays: List[np.ndarray],
    value_arrays: List[np.ndarray],
    trade_data: np.ndarray,
    timestamps: np.ndarray,
    pair_indices: np.ndarray
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    使用Numba优化的函数，更新持仓和账户余额
    
    :param balance_array: 余额数组
    :param position_arrays: 每个交易对的持仓数组列表
    :param value_arrays: 每个交易对的价值数组列表  
    :param trade_data: 交易数据数组 [open_ts, close_ts, pair_idx, stake_amount, amount, profit, is_short]
    :param timestamps: 时间戳数组
    :param pair_indices: 交易对索引映射
    :return: 更新后的余额数组和持仓数组列表
    """
    for i in range(len(trade_data)):
        open_ts = trade_data[i, 0]
        close_ts = trade_data[i, 1]
        pair_idx = int(trade_data[i, 2])
        stake_amount = trade_data[i, 3]
        amount = trade_data[i, 4]
        profit = trade_data[i, 5]
        is_short = trade_data[i, 6]
        
        # 找到最接近的开仓时间索引
        open_idx = _find_closest_index(timestamps, open_ts)
        if open_idx < 0:
            continue
        
        # 更新开仓后的余额和持仓
        for j in range(open_idx, len(balance_array)):
            balance_array[j] -= stake_amount
        
        if is_short:
            amount = -amount
            
        for j in range(open_idx, len(position_arrays[pair_idx])):
            position_arrays[pair_idx][j] += amount
        
        # 如果有平仓
        if not np.isnan(close_ts) and close_ts > 0:
            close_idx = _find_closest_index(timestamps, close_ts)
            if close_idx < 0:
                continue
                
            # 更新平仓后的余额和持仓
            for j in range(close_idx, len(balance_array)):
                balance_array[j] += stake_amount + profit
                
            for j in range(close_idx, len(position_arrays[pair_idx])):
                position_arrays[pair_idx][j] -= amount
    
    return balance_array, position_arrays

@njit
def _calculate_position_values_numba(
    position_arrays: List[np.ndarray],
    value_arrays: List[np.ndarray],
    close_arrays: List[np.ndarray]
) -> List[np.ndarray]:
    """
    使用Numba优化的函数，计算持仓价值
    
    :param position_arrays: 每个交易对的持仓数组列表
    :param value_arrays: 每个交易对的价值数组列表
    :param close_arrays: 每个交易对的收盘价数组列表
    :return: 更新后的价值数组列表
    """
    for pair_idx in range(len(position_arrays)):
        for i in range(len(position_arrays[pair_idx])):
            if not np.isnan(close_arrays[pair_idx][i]):
                value_arrays[pair_idx][i] = position_arrays[pair_idx][i] * close_arrays[pair_idx][i]
    
    return value_arrays

@njit
def _calculate_total_values_numba(
    balance_array: np.ndarray,
    value_arrays: List[np.ndarray],
    total_position_value: np.ndarray,
    total_account_value: np.ndarray,
    initial_balance: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用Numba优化的函数，计算总价值和收益率
    
    :param balance_array: 余额数组
    :param value_arrays: 每个交易对的价值数组列表
    :param total_position_value: 总持仓价值数组
    :param total_account_value: 总账户价值数组
    :param initial_balance: 初始余额
    :return: 更新后的总持仓价值、总账户价值和累积收益率
    """
    cumulative_return = np.zeros_like(total_account_value)
    
    for i in range(len(balance_array)):
        total_pos_value = 0.0
        for pair_idx in range(len(value_arrays)):
            if not np.isnan(value_arrays[pair_idx][i]):
                total_pos_value += value_arrays[pair_idx][i]
        
        total_position_value[i] = total_pos_value
        total_account_value[i] = balance_array[i] + total_pos_value
        
        if initial_balance > 0:
            cumulative_return[i] = (total_account_value[i] / initial_balance) - 1
    
    return total_position_value, total_account_value, cumulative_return

def calculate_positions(
    trades_df: pd.DataFrame,
    price_data: Dict[str, pd.DataFrame],
    initial_balance: float
) -> pd.DataFrame:
    """
    计算持仓情况和账户价值变化，保持原始数据的时间粒度，使用Numba优化性能

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

    # 为所有交易对初始化持仓列
    pair_list = list(price_data.keys())
    pair_to_idx = {pair: i for i, pair in enumerate(pair_list)}
    
    for pair in pair_list:
        positions[f'{pair}_position'] = 0.0
        positions[f'{pair}_value'] = 0.0
    positions['total_position_value'] = 0.0

    # 准备Numba兼容的数组
    time_array = positions['time'].values
    balance_array = np.full(len(positions), initial_balance)
    position_arrays = [np.zeros(len(positions)) for _ in range(len(pair_list))]
    value_arrays = [np.zeros(len(positions)) for _ in range(len(pair_list))]
    close_arrays = [np.full(len(positions), np.nan) for _ in range(len(pair_list))]
    
    # 准备交易数据数组 [open_ts, close_ts, pair_idx, stake_amount, amount, profit, is_short]
    trade_data = []
    for _, trade in trades_df.iterrows():
        pair = trade['pair']
        if pair in pair_to_idx:
            open_ts = pd.Timestamp(trade['open_date']).value // 10 ** 6
            
            close_ts = np.nan
            if pd.notna(trade['close_date']):
                close_ts = pd.Timestamp(trade['close_date']).value // 10 ** 6
            
            pair_idx = pair_to_idx[pair]
            stake_amount = float(trade['stake_amount'])
            amount = float(trade['amount'])
            profit = float(trade['profit_abs'])
            is_short = float(trade.get('is_short', False))
            
            trade_data.append([open_ts, close_ts, pair_idx, stake_amount, amount, profit, is_short])
    
    if trade_data:
        trade_array = np.array(trade_data)
        
        # 使用Numba优化的函数更新持仓和余额
        balance_array, position_arrays = _update_positions_numba(
            balance_array, 
            position_arrays, 
            value_arrays,
            trade_array, 
            time_array, 
            np.array(list(pair_to_idx.values()))
        )
    
    # 计算收盘价和持仓价值
    for pair_idx, pair in enumerate(pair_list):
        pair_price_data = price_data[pair].copy()
        pair_price_data['date'] = pd.to_datetime(pair_price_data['date'])
        pair_price_data = pair_price_data.set_index('date')
        
        # 初始价格
        initial_close = pair_price_data['close'].iloc[0] if not pair_price_data.empty else None
        
        # 转换价格数据的时间戳以便快速查找
        price_timestamps = pair_price_data.index.astype('int64') // 10 ** 6
        price_timestamps_array = price_timestamps.values
        price_close_array = pair_price_data['close'].values
        
        # 为每个时间点找到最接近的价格
        for i, timestamp in enumerate(time_array):
            idx = _find_closest_index(price_timestamps_array, timestamp)
            if idx >= 0:
                close_arrays[pair_idx][i] = price_close_array[idx]
        
        positions[f'{pair}_close'] = close_arrays[pair_idx]
        
        # 计算BNH收益率
        if initial_close and initial_close > 0:
            positions[f'{pair}_bh_return'] = (positions[f'{pair}_close'] / initial_close) - 1
            positions[f'{pair}_bh_return_pct'] = positions[f'{pair}_bh_return'] * 100
    
    # 使用Numba优化的函数计算持仓价值
    value_arrays = _calculate_position_values_numba(position_arrays, value_arrays, close_arrays)
    
    # 更新DataFrame中的持仓和价值
    for pair_idx, pair in enumerate(pair_list):
        positions[f'{pair}_position'] = position_arrays[pair_idx]
        positions[f'{pair}_value'] = value_arrays[pair_idx]
    
    # 计算总账户价值和收益率
    total_position_value = np.zeros(len(positions))
    total_account_value = np.zeros(len(positions))
    cumulative_return = np.zeros(len(positions))
    
    total_position_value, total_account_value, cumulative_return = _calculate_total_values_numba(
        balance_array, 
        value_arrays, 
        total_position_value, 
        total_account_value,
        initial_balance
    )
    
    positions['balance'] = balance_array
    positions['total_position_value'] = total_position_value
    positions['total_account_value'] = total_account_value
    positions['cumulative_return'] = cumulative_return
    positions['cumulative_return_pct'] = cumulative_return * 100
    
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
    
    # 转换为时间戳以使用优化的查找函数
    timestamps = date_index.astype('int64').values
    target_ts = target_date.value
    
    idx = _find_closest_index(timestamps, target_ts)
    if idx >= 0:
        return date_index[idx]
    return None

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
