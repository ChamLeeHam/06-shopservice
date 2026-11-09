"""
实体抽取节点：根据用户的需求抽取问题中的实体
"""
from external_lib.uie_pytorch.uie_predictor import UIEPredictor
from src.schema.base import BaseNode, T
from src.schema.exceptions import ValidationError
from src.schema.state import GraphState

class EntityExtractionNode(BaseNode):
    name = "entity_extraction_node"

    def __init__(self):
        super().__init__()
        self.uie_predictor = UIEPredictor(
            model='uie-base',
            schema=[],
            task_path= self.config.checkpoint_dir / "uie" / "model_best"  # 使用微调后的模型位置
        )
    def _extract_entity(self, question: str, schema):
        self.uie_predictor.set_schema(schema)   # 设置实体抽取的schema
        result = self.uie_predictor(question)[0]
        """
                [
                    {'商品': [{'end': 11,
                      'probability': np.float32(0.99792856),
                      'start': 0,
                      'text': '小米12S Ultra'}],
                      '颜色': [{'end': 16,
                              'probability': np.float32(0.9713122),
                              'start': 12,
                              'text': '黑色陶瓷'}]
                    }
                  ]

                  返回的数据结构应该为：
                  {
                      "商品": "小米12S Ultra",
                      "颜色": "黑色陶瓷"
                  }
        """

        for key in result.keys():
            result[key] = [val['text'] for val in result[key] ]
        return result

    def process(self, state:GraphState) ->GraphState:
        """
        处理入口节点
        :param state:
        :return:
        """
        self.log_step(self.name, "开始执行实体抽取节点")
        question = state.get("corrected_text") or state.get("question")

        if not question:
            return ValidationError("用户问题为空", self.name)

            # 添加添加实体处理逻辑
            # 根据意图来确定具体的实体，就是这个schema

        schema = ["商品"]
        result = self._extract_entity(question, schema)
        state["entity"] = result

        self.log_step(self.name, f"实体抽取的结果为：{result}")

        return state


