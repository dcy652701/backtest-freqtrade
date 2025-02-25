import requests
import pandas as pd


def fetch_and_merge_data(strategyid, start_time, end_time):
    url = 'http://localhost:48080/app-api/coin/freqtrade/getData'
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
    # 合并所有 DataFrame
    merged_df = primary_df[['open_time']].copy()  # 创建一个初始的 DataFrame 只包含 open_time

    # 合并 primaryData 和 dataframes
    for df_data in dataframes:
        df = pd.DataFrame(df_data)
        # 根据 open_time 合并
        merged_df = pd.merge(merged_df, df, on='open_time', how='outer')

    # 合并 primaryData 中的主指标数据
    merged_df = pd.merge(merged_df, primary_df, on='open_time', how='outer')

    # 转换时间戳 返回最终的 DataFrame
    merged_df['open_time'] = pd.to_datetime(merged_df['open_time'], unit='ms')
    return merged_df


def parse_condition_and_assign(dataframe: pd.DataFrame, condition: str, column_to_assign: str,
                               value_to_assign: int = 1):
    # 解析条件字符串，假设条件是 "字段1>字段2"
    left_operand, operator, right_operand = condition.split(' ')

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

# if __name__ == '__main__':
#     # 使用示例：
#     strategyid = 1892971586960027650
#     start_time = "2024-02-10 00:00:00"
#     end_time = "2024-02-10 23:59:59"
#     df = fetch_and_merge_data(strategyid, start_time, end_time)
#
#     # 输出结果
#     print(df)