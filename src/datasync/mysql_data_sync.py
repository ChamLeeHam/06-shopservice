#将业务数据中国的数据迁移到知识图谱中
import sys
from pathlib import Path

from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.schema.config import get_config



import pymysql
from pymysql.cursors import DictCursor



def read_sku_base_info(cursor):
    """从业务库中查询商品相关的数据"""
    pass

def read_sku_attr_info(cursor):
    """从业务库中查询商品属性相关的数据"""
    pass

def write_sku_base_info(driver, sku_base_info):
    """向图谱中添加商品数据"""
    pass

def write_sku_attr_info(driver, sku_attr_info):
    """向图谱中添加商品属性数据"""
    pass



if __name__ == '__main__':
    try:
        config = get_config()
        with pymysql.connect(
            host=config.mysql_host,
            user=config.mysql_user,
            password=config.mysql_password,
            database=config.mysql_database,
            charset=config.mysql_charset
        ) as conn:
            with conn.cursor(DictCursor) as cursor:
                # 1.先读取商品相关的数据
                sku_base_info = read_sku_base_info()

                # 2. 读取商品和平台相关的属性数据
                sku_attr_info = read_sku_attr_info()

        with GraphDatabase.driver(
            uri=config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
        ) as driver:
            # 3. 将商品数据写入到知识图谱中
            write_sku_base_info(driver, sku_base_info)
            # 4. 将商品和平台相关的属性数据写入到知识图谱中
            write_sku_attr_info(driver, sku_attr_info)


        print("数据迁移成功! ")
    except Exception as e:
        print(f"数据迁移失败: {e}")