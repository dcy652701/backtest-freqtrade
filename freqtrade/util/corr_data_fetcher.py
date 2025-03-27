import requests
import pandas as pd
import logging
import re
import io

logger = logging.getLogger(__name__)


def fetch_and_merge_data(strategyid, start_time, end_time, url, user_id):
    # url = 'https://corrai.tech/app-api/coin/freqtrade/getData'
    params = {
        'strategyId': strategyid,
        'startTime': start_time,
        'endTime': end_time,
        'userId': user_id
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


def upload_dataframe_to_oss(dataframe: pd.DataFrame, path: str, oss_upload_url: str) -> str:
    """
    将 DataFrame 转换为 CSV 并上传到 OSS。
    :param dataframe: 需要上传的 Pandas DataFrame
    :param path: OSS 目标路径 (如 "backtest-data/strategy_xxx_data.csv")
    :param oss_upload_url: OSS 上传接口 URL
    :return: 上传后的文件路径
    """
    # 生成与 strategy 相关的本地文件名
    filename = path.split("/")[-1]  # 从 path 提取文件名，例如 "strategy_xxx_data.csv"

    # 将 DataFrame 转换为 CSV
    csv_buffer = io.BytesIO()
    dataframe.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_buffer.seek(0)

    # 计算文件大小
    # content_length = str(len(csv_buffer.getvalue()))
    # 获取文件大小
    csv_buffer.seek(0, io.SEEK_END)
    file_size = csv_buffer.tell()  # 获取字节大小
    csv_buffer.seek(0)

    # 构造 multipart/form-data 请求
    files = {'file': (filename, csv_buffer, 'text/csv')}  # 用动态 filename
    data = {'path': path}
    headers = {
        'Content-Length': str(file_size),
        'tenant-id': '1'
    }

    # 发送 POST 请求
    response = requests.post(oss_upload_url, files=files, data=data, headers=headers)

    if response.status_code == 200:
        result = response.json()
        if result.get("code") == 0:
            return result["data"]  # 返回上传后的文件路径
        else:
            raise Exception(f"OSS 上传失败: {result}")
    else:
        raise Exception(f"HTTP 请求失败，状态码: {response.status_code}, 响应: {response.text}")


if __name__ == '__main__':
    corr_data = fetch_and_merge_data(1905273721054367746, "2020-03-27 08:50:25", "2025-03-27 08:50:25",
                         "http://localhost:48080/app-api/coin/freqtrade/getData", "290")

    print(corr_data)