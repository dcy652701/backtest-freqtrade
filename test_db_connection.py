import os
import socket
import time

import pandas as pd
import pymysql
from sqlalchemy import create_engine


def test_connection_with_retry(max_retries=3, delay=2):
    for attempt in range(max_retries):
        try:
            print(f"\n尝试连接 (第 {attempt + 1} 次):")
            # 添加更多的连接参数
            engine = create_engine(
                "",
                pool_recycle=3600,
                pool_timeout=60,  # 增加超时时间
                pool_pre_ping=True,
                connect_args={"connect_timeout": 30, "read_timeout": 30, "write_timeout": 30},
            )

            # 尝试执行一个简单的查询
            with engine.connect() as connection:
                query = "SELECT 1"
                result = pd.read_sql(query, connection)
                print("数据库连接成功！")
                print("查询结果:", result)
                return True

        except pymysql.Error as e:
            print(f"MySQL错误: {str(e)}")
            if attempt < max_retries - 1:
                print(f"等待 {delay} 秒后重试...")
                time.sleep(delay)
            else:
                print("已达到最大重试次数，连接失败")
        except Exception as e:
            print(f"其他错误: {str(e)}")
            if attempt < max_retries - 1:
                print(f"等待 {delay} 秒后重试...")
                time.sleep(delay)
            else:
                print("已达到最大重试次数，连接失败")
    return False


# 首先测试服务器是否可达
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    result = sock.connect_ex(("194.233.65.183", 39030))
    if result == 0:
        print("服务器端口可以访问")
    else:
        print("无法连接到服务器端口")
    sock.close()
except Exception as e:
    print(f"连接测试失败: {str(e)}")

print("\n测试 SQLAlchemy 连接:")
test_connection_with_retry()

print("\n测试纯 PyMySQL 连接:")
try:
    # 直接使用 PyMySQL 连接
    connection = pymysql.connect(
        host="127.0.0.1",
        port=39030,
        user="test_corr",
        password=os.environ.get("DB_PASSWORD", "default_password"),  # 使用环境变量
        database="coin_db_test",
        connect_timeout=30,
        read_timeout=30,
        write_timeout=30,
    )

    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        print("数据库连接成功！")
        print("查询结果:", result)

except pymysql.Error as e:
    print(f"MySQL错误: {str(e)}")
except Exception as e:
    print(f"其他错误: {str(e)}")
finally:
    if "connection" in locals():
        connection.close()
