import requests
import pandas as pd
import logging
import re

logger = logging.getLogger(__name__)


def fetch_and_merge_data(strategyid, start_time, end_time, url):
    # url = 'https://corrai.tech/app-api/coin/freqtrade/getData'
    params = {
        'strategyId': strategyid,
        'startTime': start_time,
        'endTime': end_time
    }
    headers = {
        'tenant-id': '1'
    }
    response = requests.get(url, params=params, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}")

    data = response.json()

    # 提取 primaryData
    primary_data = data['data']['primaryData']
    primary_df = pd.DataFrame(primary_data)

    # 提取 dataframes 数据
    dataframes = data['data']['dataframes']

    # 如果 primaryData 只有 open_time 列，我们确保它仍然有效
    if 'open_time' in primary_df.columns and len(primary_df.columns) == 1:
        primary_df = primary_df[['open_time']]  # 保证只有 open_time 列时处理正确

    # 合并所有 DataFrame
    merged_df = primary_df[['open_time']].copy()  # 创建一个初始的 DataFrame 只包含 open_time

    # 合并 primaryData 和 dataframes 的数据
    for df_data in dataframes:
        df = pd.DataFrame(df_data)
        if len(df.columns) > 1:  # 如果 df 不仅仅有 open_time 列
            # 合并数据，根据 open_time 合并
            merged_df = pd.merge(merged_df, df, on='open_time', how='outer')
    # 合并 primaryData 中的主指标数据
    if len(primary_df.columns) > 1:  # 确保 primary_df 不仅有 open_time 列才进行合并
        merged_df = pd.merge(merged_df, primary_df, on='open_time', how='outer')
    # 将 open_time 转换为日期格式，单位是毫秒，所以需要除以1000
    merged_df['date'] = pd.to_datetime(merged_df['open_time'], unit='ms', utc=True)
    return merged_df


def parse_condition_and_assign(dataframe: pd.DataFrame, condition: str, column_to_assign: str,
                               value_to_assign: int = 1):
    # 处理复合条件 (使用 | 和 & 连接的条件)
    # condition = condition.replace("&", " and ").replace("|", " or ")
    logger.info(f"Corr backtest input condition is: {condition}")
    # 动态替换列名为 dataframe['column'] 格式
    columns = dataframe.columns.tolist()
    escaped_columns = [re.escape(col) for col in columns]
    pattern = r'(?<!\w)(' + '|'.join(escaped_columns) + r')(?!\w)'

    modified_condition = re.sub(pattern, r"dataframe['\1']", condition)
    modified_condition = re.sub(r"(dataframe\['\w+'\][^&|()]+)", r"(\1)", modified_condition)

    logger.info(f"Corr backtest dataframe's condition is: {modified_condition}")
    try:
        condition_result = eval(modified_condition, {}, {"dataframe": dataframe})
    except Exception as e:
        raise ValueError(f"Error parsing condition: {e}")

    # 将计算结果赋值给指定的列
    # column_to_assign = 'enter_long' 进场信号, 'exit_long' 出场信号
    dataframe.loc[condition_result, column_to_assign] = value_to_assign
    return dataframe


# if __name__ == '__main__':
#     # 使用示例：
#     df = fetch_and_merge_data(1895303419358031873, "2024-02-27 21:22:08", "2025-02-27 21:22:08",
#                               "http://localhost:48080/app-api/coin/freqtrade/getData")
#
#     # 输出结果
#     print(df)