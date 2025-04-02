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
    # # 合并 primaryData 中的主指标数据
    # if len(primary_df.columns) > 1:  # 确保 primary_df 不仅有 open_time 列才进行合并
    #     merged_df = pd.merge(merged_df, primary_df, on='open_time', how='outer')
    # 将 open_time 转换为日期格式，单位是毫秒，所以需要除以1000
    merged_df['date'] = pd.to_datetime(merged_df['open_time'], unit='ms', utc=True)
    # merged_df.to_csv("debug0331.csv")
    return merged_df


def parse_condition_and_assign(dataframe: pd.DataFrame, condition: dict, trade_style: int, signal_type: str):
    """
    解析并在 DataFrame 中生成交易信号
    参数:
        dataframe: pd.DataFrame - 交易数据
        condition: dict - 交易条件 {"long": "btc_close_1h > btc_sma_30", "short": "btc_close_1h < btc_sma_30"}
        trade_style: int - 交易风格 (0=Long Only, 1=Short Only, 2=Long & Short)
        signal_type: str - 交易信号类型 ("entry" 或 "exit")
    返回:
        pd.DataFrame - 处理后的 DataFrame
    """
    logger.info(f"Corr backtest input condition: {condition}")

    columns = dataframe.columns.tolist()
    escaped_columns = [re.escape(col) for col in columns]
    pattern = r'(?<!\w)(' + '|'.join(escaped_columns) + r')(?!\w)'

    # 根据信号类型确定要写入的列名
    long_column = "enter_long" if signal_type == "entry" else "exit_long"
    short_column = "enter_short" if signal_type == "entry" else "exit_short"

    # 解析 Long 信号条件
    if trade_style in [0, 2] and condition.get("long"):  # Long Only 或 Long & Short
        long_condition = re.sub(pattern, r"dataframe['\1']", condition["long"])
        long_condition = re.sub(r"(dataframe\['\w+'\][^&|()]+)", r"(\1)", long_condition)

        logger.info(f"Processed long {signal_type} condition: {long_condition}")
        try:
            dataframe.loc[eval(long_condition, {}, {"dataframe": dataframe}), long_column] = 1
        except Exception as e:
            raise ValueError(f"Error parsing long {signal_type} condition: {e}")

    # 解析 Short 信号条件
    if trade_style in [1, 2] and condition.get("short"):  # Short Only 或 Long & Short
        short_condition = re.sub(pattern, r"dataframe['\1']", condition["short"])
        short_condition = re.sub(r"(dataframe\['\w+'\][^&|()]+)", r"(\1)", short_condition)

        logger.info(f"Processed short {signal_type} condition: {short_condition}")
        try:
            dataframe.loc[eval(short_condition, {}, {"dataframe": dataframe}), short_column] = 1
        except Exception as e:
            raise ValueError(f"Error parsing short {signal_type} condition: {e}")

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
    corr_data = fetch_and_merge_data(1906724320736051202, "2024-03-29 16:00:00", "2025-03-29 16:00:00", "http://localhost:48080/app-api/coin/freqtrade/getData", "290")

    print(corr_data)

    # dataframe = pd.read_csv("User290Strategy1906724320736051202.csv")
    # conditions = {
    #     "long": "Profit_Sum_v_1d<Profit_Sum_v_1d_lowerband_5_2_2_0",
    #     "short": None
    # }
    # dataframe = parse_condition_and_assign(dataframe, conditions, 0, "entry")
