"""
构建整个流程图的
"""
import sys
from pathlib import Path



PROJECT_ROOT= Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_lib"/"uie_pytorch"))

from src.nodes.entity_extraction_node import EntityExtractionNode
from src.nodes import entry_node
from src.nodes.intent_classify_node import IntentClassifyNode


from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, END

from src.nodes.entry_node import EntryNode
from src.schema.state import GraphState, create_default_state
from src.schema.base import setup_logging
from src.nodes.spell_check_node import SpellCheckNode

from src.nodes.presale_node import PresaleNode
from src.nodes.aftersale_node import AftersaleNode
from src.nodes.logistics_node import LogisticsNode
from src.nodes.handoff_node import HandoffNode
from src.nodes.order_node import OrderNode
from src.nodes.other_node import OtherNode

def route_by_intent(state:GraphState) -> str:
    """
    根据意图分类进行路由
    :param state:
    :return:
    """
    intent = state.get("predicted_intent") or ""
    if intent.startswith("presale."):
        return "presale_node"
    elif intent.startswith("after_sale."):
        return "aftersale_node"
    elif intent.startswith("logistics."):
        return "logistics_node"
    elif intent.startswith("order."):
        return "order_node"
    elif intent.startswith("out_of_scope"):
        return "other_node"

    if state.get("human_requested"):
        return "handoff_node"


def create_query_graph()->CompiledStateGraph:
    # 1.定义状态图
    workflow = StateGraph(GraphState)

    # 定义当前存在各种节点
    nodes = {
        "entry_node": EntryNode(),
        "spell_check_node": SpellCheckNode(),
        "entity_extraction_node": EntityExtractionNode(),
        "intent_classify_node": IntentClassifyNode(),
        "presale_node": PresaleNode(),  # 接下来完成售前咨询问题
        "aftersale_node": AftersaleNode(),
        "logistics_node": LogisticsNode,
        "other_node": OtherNode,
        "order_node": OrderNode,
        "handoff_node": HandoffNode,
    }
    # 2。 添加节点
    for name, node in nodes.items():
        workflow.add_node(name, node)

    # 3。添加边
    workflow.set_entry_point("entry_node"),
    workflow.add_edge("entry_node", "intent_classify_node"),
    workflow.add_edge("intent_classify_node", "spell_check_node")

    # 根据意图分类进行路由：售前咨询、售后服务、物流服务、订单管理、兜底、人工介入
    workflow.add_conditional_edges(
        "spell_check_node",
        route_by_intent,
        ["presale_node","aftersale_node","logistics_node","order_node","other_node","handoff_node"])

    workflow.add_edge("presale_node", "entity_extraction_node")
    workflow.add_edge("entity_extraction_node", END)
    workflow.add_edge("aftersale_node", END)
    workflow.add_edge("logistics_node", END)
    workflow.add_edge("order_node", END)
    workflow.add_edge("other_node", END)
    workflow.add_edge("handoff_node", END)
    workflow.add_edge("entity_extraction_node", END)

    # 4。构建图
    return workflow.compile()

_app = create_query_graph()

def run_query(session_id, question:str):
    # 1.创建初始状态
    initial_state = create_default_state(session_id=session_id, question=question)

    final_state = None
    # 2.运行图
    for event in _app.stream(initial_state):
        for key, value in event.items():
            print(f"节点：{key}")
            final_state = value

    # 3。返回图执行完成后的状态
    return final_state or initial_state

if __name__ == '__main__':
    setup_logging()
    # 执行工作流
    question = "小米12S Ultra 都有哪些版本。"
    #question = "请帮我转人工处理！"
    result = run_query(session_id="123", question=question)
    print(result)

    # 显示整个图
    _app.get_graph().print_ascii() #在控制台中显示图结构