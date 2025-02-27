import requests
import pandas as pd
import re


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
    # 解析条件字符串，假设条件是 "字段1>字段2"
    # 使用正则表达式来提取操作符前后两部分
    match = re.match(r'(\w+)([<>!=]=?)(\w+)', condition)

    if not match:
        raise ValueError(f"Invalid condition format: {condition}")

    left_operand, operator, right_operand = match.groups()

    if operator == '>':
        condition_result = dataframe[left_operand] > dataframe[right_operand]
    elif operator == '<':
        condition_result = dataframe[left_operand] < dataframe[right_operand]
    elif operator == '==':
        condition_result = dataframe[left_operand] == dataframe[right_operand]
    elif operator == '!=':
        condition_result = dataframe[left_operand] != dataframe[right_operand]
    elif operator == '>=':
        condition_result = dataframe[left_operand] >= dataframe[right_operand]
    elif operator == '<=':
        condition_result = dataframe[left_operand] <= dataframe[right_operand]
    else:
        raise ValueError(f"Unsupported operator: {operator}")

    # 将计算结果赋值给指定的列
    # column_to_assign = 'enter_long' 进场函数
    dataframe.loc[condition_result, column_to_assign] = value_to_assign
    return dataframe


if __name__ == '__main__':
    # 使用示例：
    strategyid = 1892971586960027650
    start_time = "2024-02-10 00:00:00"
    end_time = "2024-02-10 23:59:59"
    df = fetch_and_merge_data(strategyid, start_time, end_time)

    # 输出结果
    print(df)