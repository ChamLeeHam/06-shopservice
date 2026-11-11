"""
定义工作流中的数据状态：包括完整的状态结构和辅助函数
"""
import copy
from typing import TypedDict, List


class GraphState(TypedDict, total=False):
    """
    记录工作流完整的状态数据
    total=False:字段是可选的
    """
    # ==================== 任务标识 ====================
    session_id: str  # 任务 ID，用于任务追踪

    # ==================== 用户输入问题 ====================
    question: str
    require_high_confidence: bool  # 是否要求高置信度
    corrected_text: str  # 纠错后的文本
    errors: List  # 错误列表
    explanation: str  # 修正说明
    spell_confidence: float  # 置信度

    # ==================== 实体抽取 ====================
    entity: List  # 实体列表

    # ==================== 意图分类 ====================
    intent_domain: str  # 意图领域
    intent_confidence: float  # 意图置信度
    human_requested:bool # 是否需要人工干预
    predicted_intent: str  # 意图
    label_id: int  # 意图 ID
    is_other: bool  # 是否为其他意图
    recommend_action: str  # 推荐操作

    # ==================== 其他状态 ====================

GRAPH_DEFAULT_STATE: GraphState = {
    "session_id": "",
    "question": "",
    "require_high_confidence": False,
    "corrected_text": "",
    "errors": [],
    "explanation": "",
    "spell_confidence": 1.0,
    "entity": [],
    "predicted_intent": "",
    "intent_domain": "",
    "intent_confidence": 1.0,
    "human_requested": False,
    "label_id": -1,
    "is_other": False,
    "recommend_action": "",
}

def get_default_state()->GraphState:
    """
    获取默认状态副本
    :return: 状态副本（避免全局污染）
    """
    return copy.deepcopy(GRAPH_DEFAULT_STATE)

def create_default_state(**overrides)->GraphState:
    """
    创建默认状态，并根据需要修改字段
    create_default_state(label_id=10,is_other=True )
    :param overrides:
    :return:
    """
    state = copy.deepcopy(GRAPH_DEFAULT_STATE)
    state.update(overrides) #跟新拷贝过来的值
    return state