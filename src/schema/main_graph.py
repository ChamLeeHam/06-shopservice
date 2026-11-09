"""
构建整个流程图的
"""
import sys
from pathlib import Path



PROJECT_ROOT= Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_lib"/"uie_pytorch"))

from src.nodes.entity_extraction_node import EntityExtractionNode

from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, END

from src.nodes.entry_node import EntryNode
from src.schema.state import GraphState, create_default_state
from src.schema.base import setup_logging
from src.nodes.spell_check_node import SpellCheckNode


def create_query_graph()->CompiledStateGraph:
    # 1.定义状态图
    workflow = StateGraph(GraphState)

    # 定义当前存在各种节点
    nodes = {
        "entry_node": EntryNode(),
        "spell_check_node": SpellCheckNode(),
        "entity_extraction_node": EntityExtractionNode()
    }
    # 2。 添加节点
    for name, node in nodes.items():
        workflow.add_node(name, node)

    # 3。添加边
    workflow.set_entry_point("entry_node")
    workflow.add_edge("entry_node", "spell_check_node")
    workflow.add_edge("spell_check_node", "entity_extraction_node")
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
    result = run_query(session_id="123", question=question)
    print(result)

    # 显示整个图
    _app.get_graph().print_ascii() #在控制台中显示图结构