"""意图识别"""
import json

import torch
from transformers import BertTokenizer

from src.models.intent_classify_bert import BertIntentModel
from src.schema.base import BaseNode, T
from src.schema.exceptions import ValidationError
from src.schema.state import GraphState

# 二级意图 -> 一级域（由前缀推导，与训练侧 INTENT_DOMAIN 保持一致）
PREFIX_DOMAIN = {
    "logistics": "物流服务",
    "order": "订单管理",
    "aftersale": "售后服务",
    "presale": "售前咨询",
}
OOD_LABEL = "out_of_scope"


class IntentClassifyNode(BaseNode):
    name = "intent_classify_node"

    # 关键词规则：只放「几乎不可能误判」的高确定性场景，命中即短路省一次前向。
    # 注意：返回的意图名必须是标签空间内的真实标签。此处仅示例，上线前需逐条评测召回/误伤。
    INTENT_RULES = [
        ("logistics.track", ["快递到哪了", "物流到哪了", "我的快递到哪","物流", "快递"]),
        ("order.cancel", ["取消订单", "帮我把订单取消"]),
        ("aftersale.apply", ["我要退货", "申请退货退款"]),
        #("presale.attr", ["产品", "版本"]),
    ]

    def __init__(self):
        super().__init__()
        model_path = self.config.checkpoint_dir / "intent_classify" / "best_model"
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        # 关键：用训练时的自定义双头模型类加载，权重名才能一一对应
        self.model = BertIntentModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # type: ignore

        mapping_path = self.config.checkpoint_dir / "intent_classify" / "label_mapping.json"
        with open(mapping_path, "r", encoding="utf-8") as f:
            self.label_mapping = json.load(f)  # {"0": "logistics.track", ...}

        # 注意：这是「未校准」的 softmax 阈值，仅作占位；上线前应做温度缩放后按意图分别定阈值。
        self.threshold = self.config.intent_threshold


    def _match_intent(self, question: str):
        """关键词规则意图识别：命中返回标签空间内的意图，未命中返回 None 交给模型。"""
        for intent, keywords in self.INTENT_RULES:
            if any(kw in question for kw in keywords):
                return intent
        return None

    def _predict_intent(self, question:str):
        """基于bert模型做意图识别"""
        inputs = self.tokenizer(
            question,
            max_length=self.config.max_length,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # type: ignore

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)  # [batch_size, num_labels]
            intent_probs = torch.softmax(outputs["logits"], dim=-1)
            human_probs = torch.softmax(outputs["human_logits"], dim=-1)

            label_id = int(torch.argmax(intent_probs, dim=-1).item())
            confidence = float(intent_probs[0][label_id].item())

            human_label_id = int(torch.argmax(human_probs, dim=-1).item())
            human_requested = human_label_id == 1
            human_confidence = float(human_probs[0][human_label_id].item())

        pred_label = self.label_mapping.get(str(label_id), OOD_LABEL)
        domain = PREFIX_DOMAIN.get(pred_label.split(".")[0], OOD_LABEL)
        is_out_of_scope = pred_label == OOD_LABEL
        is_low_confidence = (not is_out_of_scope) and confidence < self.threshold

        # 路由建议（与路由优先级一致：P0 人工 > 兜底 > 低置信澄清 > 直接响应）
        if human_requested:
            recommend_action = "转人工"  # P0：用户明确要人
        elif is_out_of_scope:
            recommend_action = "兜底回复或转人工"  # 超出业务范围
        elif is_low_confidence:
            recommend_action = "建议澄清或转人工"  # 置信度不足
        else:
            recommend_action = "直接响应"

        return {
            "question": question,
            "predicted_intent": pred_label,
            "domain": domain,
            "confidence": round(confidence, 4),
            "label_id": label_id,
            "is_out_of_scope": is_out_of_scope,
            "is_low_confidence": is_low_confidence,
            "human_requested": human_requested,
            "human_confidence": round(human_confidence, 4),
            "recommend_action": recommend_action,
        }

    def process(self, state:GraphState) ->GraphState:
        self.log_step(self.name, "意图识别业务逻辑了。。。。")
        question = state.get("question") or ""

        if not question:
            return ValidationError("用户问题为空", self.name)

        # 1、基于规则做意图识别
        intent = self._match_intent(question)
        if intent is not None:
            self.log_step(self.name, f"基于规则的意图识别结果：{intent}")

            state["predicted_intent"] = intent
            state["intent_confidence"] = 1.0
            state["intent_domain"] = PREFIX_DOMAIN.get(intent.split(".")[0], OOD_LABEL)
            state["label_id"] = -1
            state["is_other"] = False
            state["human_requested"] = False
            state["recommend_action"] = "直接响应"
        else:
            # 2、基于bert模型做意图识别
            result =  self._predict_intent(question)
            state["predicted_intent"] = result["predicted_intent"]
            state["intent_domain"] = result["domain"]
            state["intent_confidence"] = result["confidence"]
            state["label_id"] = result["label_id"]
            state["is_other"] = result["is_out_of_scope"]  # 兼容旧字段
            state["human_requested"] = result["human_requested"]

            state["recommend_action"] = result["recommend_action"]

        # 3、基于大模型做意图（选装）

        self.log_step(
            self.name,
            f"意图识别结果：{state['predicted_intent']}, 置信度：{state['intent_confidence']}, 域：{state['intent_domain']},人工介入：{state["human_requested"]}"
        )
        return state