import sys

import pymysql
from pathlib import Path

from neo4j import GraphDatabase
from pymysql.cursors import DictCursor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.schema.config import get_config


config = get_config()



def read_user_view_log(config):
    """
    读取用户行为日志（user_view_log 表）
    返回的数据结构：
    [
        {'user_id': 41, 'sku_id': 8, 'view_time': datetime(2011, 2, 8, 20, 28, 6)},
        ...
    ]
    """
    with pymysql.connect(
        host=config.mysql_host,
        user=config.mysql_user,
        password=config.mysql_password,
        database=config.mysql_database,
        charset=config.mysql_charset,
    ) as conn:
        with conn.cursor(DictCursor) as cursor:
            sql = "select user_id, sku_id, view_time from user_view_log order by view_time"
            cursor.execute(sql)
            return cursor.fetchall()



def write_user_view_log(config, user_view_log):
    with GraphDatabase.driver(
        uri=config.neo4j_uri,
        auth=(config.neo4j_user, config.neo4j_password)
    ) as driver:
        for view_log in user_view_log:
            cypher = """
                        MATCH (sku:SKU {sku_id: $sku_id})
                        MERGE (user:User {user_id: $user_id})
                        MERGE (user)-[:View {view_time: $view_time}]->(sku)
                    """
            driver.execute_query(cypher, parameters_=view_log)

if __name__ == "__main__":
    try:
        #1. 读取用户行为日志
        user_view_log = read_user_view_log(config)
        # print(f"读取到{len(user_view_log)}条用户行为日志：{user_view_log[:3]}")
        #2. 将日志写入neo4j中
        write_user_view_log(config, user_view_log)
        print(f"数据迁移成功！")
    except Exception as e:
        print(e)