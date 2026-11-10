# 针对非结构化数据进行迁移，主要图商品中的图片内容进行识别。
# 1. 将图片中的文字识别出来
# 2. 从这些文字中提出属性数据（实体，schema）
# 3. 讲这些属性数据迁移到知识图谱中，补充新的Attr节点
import sys

import pymysql

from tqdm import tqdm

import torch  # keep this before paddleocr: in your graph env, torch must load before paddle
from paddleocr import PaddleOCR

from pathlib import Path

from pymysql.cursors import DictCursor



PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.schema.config import get_config

from src.nodes.spell_check_node import SpellCheckNode

config = get_config()
reader = PaddleOCR(
    lang="ch",                          # "ch" model handles both Chinese and English
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
spell_check_node = SpellCheckNode()

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
    pass

def extract_entity_from_merged_content():
    pass

def migrate_attributes_to_knowledge_graph():
    pass






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

        # 5. 将图片内容和商品描述内容进行合并

        # 6. 从合并后的内容中提取属性数据

        # 7. 将属性数据迁移到知识图谱中
    except Exception as e:
        print(e)