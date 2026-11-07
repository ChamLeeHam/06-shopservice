"""
流程状态类型定义
定义完整的状态结构和辅助函数
"""
from typing import TypedDict, List
import copy

class GraphState(TypedDict, total=False):
    """
    流程图状态
    包含整个流程中传递的所有数据
    """
    # ==================== 任务标识 ====================
    session_id: str # 任务 ID，用于任务追踪
    # ==================== 用户输入问题 ====================
    question: str
    require_high_confidence: bool # 是否要求高置信度
    corrected_text: str # 纠错后的文本
    errors: List # 错误列表
    explanation: str # 修正说明
    confidence: float # 置信度
    # ==================== 实体抽取 ====================
    entity: List # 实体列表
    # ==================== 意图分类 ====================
    predicted_intent: str # 意图
    label_id: int # 意图 ID
    confidence: float # 意图置信度
    is_other: bool # 是否为其他意图
    recommend_action: str # 推荐操作

GRPAH_DEFAULT_STATE: GraphState = {
    'session_id': '',
    'question': '',
    'require_high_confidence': False,
    'corrected_text': '',
    'errors': [],
    'explanation': '',
    'confidence': 1.0,
    'entity': [],
    'predicted_intent': '',
    'label_id': -1,
    "is_other": False,
    "recommend_action": "",
}

def get_default_state() -> GraphState:
    """
    获取默认状态副本
    :return:
    """
    return copy.deepcopy(GRPAH_DEFAULT_STATE)

def create_default_state(**overrides) -> GraphState:
    """
    create_default_state(label_id=10,is_other=True)
    :param overrides:
    :return:
    """
    state = copy.deepcopy(GRPAH_DEFAULT_STATE)
    state.update(overrides)   #更新拷贝过来的值
    return state