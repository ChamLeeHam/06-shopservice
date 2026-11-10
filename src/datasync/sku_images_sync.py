# 针对非结构化数据进行迁移，主要图商品中的图片内容进行识别。
# 1. 将图片中的文字识别出来
# 2. 从这些文字中提出属性数据（实体，schema）
# 3. 讲这些属性数据迁移到知识图谱中，补充新的Attr节点
import sys

import pymysql
from neo4j import GraphDatabase

from tqdm import tqdm

import torch  # keep this before paddleocr: in your graph env, torch must load before paddle
from paddleocr import PaddleOCR

from pathlib import Path

from pymysql.cursors import DictCursor



PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.schema.config import get_config

from src.nodes.spell_check_node import SpellCheckNode
sys.path.insert(0, str(PROJECT_ROOT/"external_lib"/"uie_pytorch"))
from external_lib.uie_pytorch.uie_predictor import UIEPredictor

config = get_config()
reader = PaddleOCR(
    lang="ch",                          # "ch" model handles both Chinese and English
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
spell_check_node = SpellCheckNode()

uie_predictor = UIEPredictor(
    model='uie-base',
    schema=[],
    task_path=config.checkpoint_dir / "uie" / "model_best",  # 使用微调后的模型位置
    device="gpu",
)

def get_sku_images():
    with pymysql.connect(
        host=config.mysql_host,
        user=config.mysql_user,
        password=config.mysql_password,
        database=config.mysql_database,
        charset=config.mysql_charset
    ) as conn:
        with conn.cursor(DictCursor) as cursor:
            sql = "select sku_id, img_url from sku_image where img_url like '/data%'"
            cursor.execute(sql)
            return cursor.fetchall()

def recognize_text_from_images(sku_images_url):
    """
    sku_images_url的数据结构：
    [
        {
            'sku_id': 36,
            'img_url': '/data/images/36/1.jpg'
        },
        ...
   ]
    识别出图片后返回的结构为：
    {
        'sku_id':[36, 37...],
        'image_text':['xxxx', 'yyyyy'...]
    }
    """
    i = 0
    sku_images_text = {'sku_id':[], 'image_text':[]}

    for sku_image in tqdm(sku_images_url, f"图片识别中。。。"):
        i += 1
        # 注意提供图片的绝对路径
        # [ OCRResult({
        #     'rec_texts':  ['E-SPORTS', 'GAME BOOK', 'Intel', '11代i7'],   # the text lines
        #     'rec_scores': [1.00, 1.00, 1.00, 0.99],                      # confidence
        #     'rec_polys':  [...],                                         # box positions
        #     ...
        # }) ]
        result = reader.predict(str(PROJECT_ROOT) + sku_image["img_url"])
        image_text = "".join(result[0]["rec_texts"])
        sku_images_text['sku_id'].append(sku_image["sku_id"])
        sku_images_text['image_text'].append(image_text)

        if i >= 3:
            break
    return sku_images_text


def correct_text(sku_images_text):
    """
        sku_images_text:{"sku_id":[], "image_text":[]}
        对image_text中的文字进行纠错

        :param sku_images_text:
        :return:
    """
    for i, text in enumerate(sku_images_text["image_text"]):
        state = {
            "question": text
        }
        result = spell_check_node.process(state)
        sku_images_text["image_text"][i] = result["corrected_text"]
    return sku_images_text

def get_product_description():
    sku_desc_text = {"sku_id":[], "sku_desc":[]}
    with pymysql.connect(
        host=config.mysql_host,
        user=config.mysql_user,
        password=config.mysql_password,
        database=config.mysql_database,
        charset=config.mysql_charset
    ) as conn:
        with conn.cursor(DictCursor) as cursor:
            sql = "select id as sku_id, sku_desc from sku_info where sku_desc is not null limit 2"
            cursor.execute(sql)
            result = cursor.fetchall()
    for row in result:
        sku_desc_text["sku_id"].append(row["sku_id"])
        sku_desc_text["sku_desc"].append(row["sku_desc"])
    return sku_desc_text


def extract_entity_from_merged_content(merged_sku_info, schema):
    """merge_sku_info = {
            "sku_id":[1,2,3,...],
            "sku_text":["xxxx", "yyyy", "zzzz"....]
        }

        merge_sku_info["sku_text"]进行实体抽取，抽取属性，写入neo4j中

        """
    uie_predictor.set_schema(schema)
    result = uie_predictor.predict(merged_sku_info['sku_text'])
    """
       [{'商品': [{'end': 11,
             'probability': np.float32(0.99792856),
             'start': 0,
             'text': '小米12S Ultra'}],
     '颜色': [{'end': 16,
             'probability': np.float32(0.9713122),
             'start': 12,
             'text': '黑色陶瓷'}]}]

       返回的数据结果:
       [
           {
               "sku_id":1,
               "attr_name":"商品",
               "attr_value":"小米12S Ultra"
           },
           {
               "sku_id":1,
               "attr_name":"颜色",
               "attr_value":"黑色陶瓷"
           }
       ]
       """
    sku_ids = merged_sku_info["sku_id"]
    sku_entities = []
    for i, item in enumerate(result):
        for key, value in item.items():
            sku_id = sku_ids[i]
            attr_name = key
            attr_value = value[0]["text"]
            sku_entities.append({
                "sku_id": sku_id,
                "attr_name": attr_name,
                "attr_value": attr_value
            })
    return sku_entities

def migrate_attributes_to_knowledge_graph(sku_entites):
    with GraphDatabase.driver(
        uri=config.neo4j_uri,
        auth=(config.neo4j_user, config.neo4j_password)
    ) as driver:
        for sku_entity in sku_entites:
            cypher="""
                MATCH (sku:SKU{sku_id:$sku_id})
                OPTIONAL MATCH (sku)-[:Have]->(attr_exists:Attr{attr_name:$attr_name, attr_value:$attr_value})
                WITH sku, attr_exists
                WHERE attr_exists IS NULL
                MERGE (attr:Attr{attr_name:$attr_name, attr_value:$attr_value})
                MERGE (sku)-[:Have]->(attr)
            """
            driver.execute_query(cypher, parameters_=sku_entity)






if __name__ == '__main__':
    try:
        # 1. 获取商品图片
        sku_images = get_sku_images()
        #print(sku_images)
        # 2. 识别图片中的文字
        sku_images_text = recognize_text_from_images(sku_images)
        print(sku_images_text)

        # 3. 对文字进行纠错
        sku_images_text = correct_text(sku_images_text)
        print(sku_images_text)

        # 4. 获取商品描述信息
        sku_desc_text = get_product_description()
        # 5. 将图片内容和商品描述内容进行合并
        merge_sku_info = {
            "sku_id": sku_images_text["sku_id"] + sku_desc_text["sku_id"],
            "sku_text": sku_images_text["image_text"] + sku_desc_text["sku_desc"]
        }
        # 6. 从合并后的内容中提取属性数据
        sku_entities = extract_entity_from_merged_content(merge_sku_info, config.schema)
        # 7. 将属性数据迁移到知识图谱中
        migrate_attributes_to_knowledge_graph(sku_entities)

        print("非结构化数据迁移成功！")
    except Exception as e:
        print(e)