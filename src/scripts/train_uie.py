from pprint import pprint
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT/"external_lib"/"uie_pytorch"))
# 将项目根目录加入 sys.path
from uie_predictor import UIEPredictor
from src.schema.config import get_config

#
# schema = ["时间", "选手", "赛事名称"]
# ie = UIEPredictor( model="uie-base", schema=schema)
# pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"))

schema = ["时间", "选手", "赛事名称"]
ie = UIEPredictor(
    model='uie-base',
    schema=schema,
    task_path=get_config().pre_train_dir/"uie_base_pytorch"
)
pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"))