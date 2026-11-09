import sys
from pathlib import Path
PROJECT_ROOT= Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT/"external_lib"/"uie_pytorch"))
from uie_predictor import UIEPredictor
from src.schema.config import get_config

from pprint import pprint

# schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
# ie = UIEPredictor(
#     model='uie-base',
#     schema=schema,
#     task_path=get_config().pre_trained_dir/"uie_base_pytorch" # 设置模型位置
# )
# pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"))

"""
[{'时间': [{'end': 6,
          'probability': np.float32(0.985738),
          'start': 0,
          'text': '2月8日上午'}],
  '赛事名称': [{'end': 23,
            'probability': np.float32(0.85030854),
            'start': 6,
            'text': '北京冬奥会自由式滑雪女子大跳台决赛'}],
  '选手': [{'end': 31,
          'probability': np.float32(0.89815384),
          'start': 28,
          'text': '谷爱凌'}]}]

"""

# schema = {'竞赛名称': ['主办方', '承办方', '已举办次数']}
# ie.set_schema(schema)
# pprint(ie('2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。'))

"""
[{'竞赛名称': [{'end': 13,
            'probability': np.float32(0.7825388),
            'relations': {'主办方': [{'end': 22,
                                   'probability': np.float32(0.8421713),
                                   'start': 14,
                                   'text': '中国中文信息学会'},
                                  {'end': 30,
                                   'probability': np.float32(0.7580802),
                                   'start': 23,
                                   'text': '中国计算机学会'}],
                          '已举办次数': [{'end': 82,
                                     'probability': np.float32(0.46712977),
                                     'start': 80,
                                     'text': '4届'}],
                          '承办方': [{'end': 55,
                                   'probability': np.float32(0.70004934),
                                   'start': 40,
                                   'text': '中国中文信息学会评测工作委员会'},
                                  {'end': 39,
                                   'probability': np.float32(0.82927024),
                                   'start': 35,
                                   'text': '百度公司'},
                                  {'end': 72,
                                   'probability': np.float32(0.61934745),
                                   'start': 56,
                                   'text': '中国计算机学会自然语言处理专委会'}]},
            'start': 0,
            'text': '2022语言与智能技术竞赛'}]}]
"""

schema = ["颜色", "商品"]
ie = UIEPredictor(
    model='uie-base',
    schema=schema,
    task_path=PROJECT_ROOT/"checkpoints"/"uie"/"model_best" # 使用微调后的模型位置
)
pprint(ie("小米12S Ultra 黑色陶瓷12GB+256GB"))
"""
使用原始的模型：
[{'颜色': [{'end': 14,
          'probability': np.float32(0.7972948),
          'start': 12,
          'text': '黑色'}]}]

使用微调后的模型：
[{'商品': [{'end': 11,
          'probability': np.float32(0.99792856),
          'start': 0,
          'text': '小米12S Ultra'}],
  '颜色': [{'end': 16,
          'probability': np.float32(0.9713122),
          'start': 12,
          'text': '黑色陶瓷'}]}]

"""


