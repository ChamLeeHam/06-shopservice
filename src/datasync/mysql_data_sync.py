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
    sql = """
    select si.id as sku_id, si.sku_name, 
    sp.spu_name, 
    c3.name as category3_name, c2.name as category2_name, c1.name as category1_name,
    tm.tm_name as trademark_name  
    from sku_info si 
    left join spu_info sp on si.spu_id = sp.id 
    left join base_category3 c3 on si.category3_id = c3.id 
    left join base_category2 c2 on c3.category2_id = c2.id
    left join base_category1 c1 on c2.category1_id = c1.id
    left join base_trademark tm on si.tm_id = tm.id
    order by si.id;
          """
    cursor.execute(sql)
    return cursor.fetchall()


def read_sku_attr_info(cursor):
    """从业务库中查询商品属性相关的数据"""
    sql = """
          select sku_id,
                 attr_name,
                 value_name attr_value
          from sku_attr_value
          union all
          select sku_id,
                 sale_attr_name,
                 sale_attr_value_name
          from sku_sale_attr_value \
          """
    cursor.execute(sql)
    return cursor.fetchall()

def write_sku_base_info(driver, sku_base_info):
    """向图谱中添加商品数据"""
    for sku in sku_base_info:
        cypher = """
            MERGE (sku:SKU {sku_id: $sku_id, sku_name: $sku_name})
            MERGE (spu:SPU {spu_name: $spu_name})
            MERGE (c3:Category3 {category3_name: $category3_name})
            MERGE (c2:Category2 {category2_name: $category2_name})
            MERGE (c1:Category1 {category1_name: $category1_name})
            MERGE (tm:Trademark {trademark_name: $trademark_name})
            MERGE (sku)-[:Belong]->(spu)
            MERGE (spu)-[:Belong]->(c3)
            MERGE (c3)-[:Belong]->(c2)
            MERGE (c2)-[:Belong]->(c1)
            MERGE (spu)-[:Belong]->(tm)
        """

        driver.execute_query(cypher, parameters_=sku)

def write_sku_attr_info(driver, sku_attr_info):
    """向图谱中添加商品属性数据"""
    for attr in sku_attr_info:
        cypher = """
            MATCH (sku:SKU {sku_id: $sku_id})
            MERGE (attr:Attr {attr_name: $attr_name,attr_value:$attr_value })
            MERGE (sku)-[:Have]->(attr)
                    
            """
        driver.execute_query(cypher, parameters_=attr)



if __name__ == '__main__':
    try:
        config = get_config()
        with pymysql.connect(
            host=config.mysql_host,
            user=config.mysql_user,
            password=config.mysql_password,
            database=config.mysql_database,
            charset=config.mysql_charset,
        ) as conn:
            with conn.cursor(DictCursor) as cursor:
                # 1、先读取商品相关的数据
                sku_base_info = read_sku_base_info(cursor)
                #print(sku_base_info)
                # 2、读取商品和平台相关的属性数据
                sku_attr_info = read_sku_attr_info(cursor)

        with GraphDatabase.driver(
            uri=config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        ) as driver:
            # 3、将商品数据写入到知识图谱中
            write_sku_base_info(driver,sku_base_info )
            # 4、将商品和平台相关的属性数据写入到知识图谱中
            write_sku_attr_info(driver, sku_attr_info)

        print("数据迁移成功！")
    except Exception as e:
        print(f"数据迁移失败{e}")