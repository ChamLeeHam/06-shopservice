"""
售后服务
"""
from src.schema.base import BaseNode, T
from src.schema.exceptions import ValidationError
from src.schema.state import GraphState

class AftersaleNode(BaseNode):
    name = "aftersale_node"
    def process(self, state:GraphState) ->GraphState:
        """
        处理入口节点
        :param state:
        :return:
        """
        self.log_step(self.name, "售后服务")
        question = state.get("question") or ""

        if not question:
            return ValidationError("用户问题为空", self.name)

        # 添加入口节点的处理逻辑

        return state