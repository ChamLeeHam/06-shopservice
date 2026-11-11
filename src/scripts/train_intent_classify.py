# 微调macbert模型
#基于Huggingface的Trainer类进行模型微调
#基于Huggingface的Trainer类进行模型微调
import sys
from pathlib import Path

import torch

PROJECT_ROOT= Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os, json
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from src.schema.config import get_config
from src.models.intent_classify_bert import BertIntentModel
from transformers import TrainingArguments, Trainer, BertTokenizer, DataCollatorWithPadding, EarlyStoppingCallback

config = get_config()

# ==================== 1. 基础配置 ====================
CONFIG = {
    "data_dir": config.data_dir / "intent_classify", # 数据文件
    "model_name": config.pre_train_dir / 'chinese-macbert-base',# 本地模型路径
    "output_dir": config.checkpoint_dir / 'intent_classify', # 输出目录
    "max_length": config.max_length,
    "batch_size": config.batch_size,
    "epochs": config.epochs,
    "learning_rate": config.learning_rate,
}

INTENT_LABELS=config.intent_labels
LABEL2ID = {label: i for i, label in enumerate(INTENT_LABELS)}
INTENT_DOMAIN = {
    "logistics.track": "物流服务",
    "logistics.complaint": "物流服务",
    "order.query": "订单管理",
    "order.modify": "订单管理",
    "order.cancel": "订单管理",
    "aftersale.apply": "售后服务",
    "aftersale.track": "售后服务",
    "presale.attr": "售前咨询",
    "presale.recommend": "售前咨询",
    "presale.compare": "售前咨询",
    "presale.stock": "售前咨询",
    "out_of_scope": "兜底",
}
DOMAIN_LABELS = ["物流服务", "订单管理", "售后服务", "售前咨询", "兜底"]
# 二级 id -> 一级 id 的映射数组，评测时直接按数组索引转换，零额外开销
DOMAIN_ID = np.array([DOMAIN_LABELS.index(INTENT_DOMAIN[label]) for label in INTENT_LABELS])

# ==================== 3. 数据加载 ====================
def load_and_prepare_data(tokenizer):
    splits = {}
    # 极简版只加载 train 和 dev
    for name in ("train", "dev"):
        path = os.path.join(CONFIG["data_dir"], f"{name}.jsonl")
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

        # 核心修复：列名必须与模型 forward 参数名完全一致 (labels, human_labels)
        df = pd.DataFrame({
            "text": [r["text"] for r in rows],
            "labels": [LABEL2ID[r["intent"]] for r in rows],
            "human_labels": [int(bool(r["human_requested"])) for r in rows],
        })

        ds = Dataset.from_pandas(df, preserve_index=False)
        ds = ds.map(
            lambda e: tokenizer(e["text"], truncation=True, max_length=CONFIG["max_length"]),
            batched=True, remove_columns=["text"]
        )
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels", "human_labels"])
        splits[name] = ds

    # 保存标签映射（推理时用）
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    with open(os.path.join(CONFIG["output_dir"], "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({str(i): l for i, l in enumerate(INTENT_LABELS)}, f, ensure_ascii=False)

    return splits["train"], splits["dev"]

# ==================== 4. 评估与 Trainer ====================
def compute_metrics(eval_pred):
    # 只评估主意图 (intent)；防御性解包：嵌套结构可能是 tuple/list 或 object ndarray
    def _first(x):
        if isinstance(x, (tuple, list)):
            return x[0]
        if isinstance(x, np.ndarray) and x.dtype == object:
            return x[0]
        return x
    logits = _first(eval_pred.predictions)
    preds = np.argmax(logits, axis=-1)
    labels = _first(eval_pred.label_ids)

    # 一级指标：由二级预测结果推导（不是另一个模型输出的），用于业务汇报
    domain_preds = DOMAIN_ID[preds]
    domain_labels = DOMAIN_ID[labels]

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "domain_accuracy": accuracy_score(domain_labels, domain_preds),
        "domain_f1_macro": f1_score(domain_labels, domain_preds, average="macro", zero_division=0),
    }

def main():
    print("开始微调...")
    tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])
    train_dataset, dev_dataset = load_and_prepare_data(tokenizer)

    model = BertIntentModel.from_pretrained(CONFIG["model_name"])

    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        eval_strategy="epoch",  # 新版 API
        save_strategy="epoch",
        load_best_model_at_end=True,  # 自动保存 dev 上 f1_macro 最高的模型
        metric_for_best_model="f1_macro",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,  # 关闭锁页内存
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,  # 新版 API
        data_collator=DataCollatorWithPadding(tokenizer),  # 动态 Padding。每个 batch 只按该 batch 内最长文本补齐，大幅减少无效计算和显存浪费。
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        # 早停机制。由于 eval_strategy='epoch'，此处表示：若连续 3 个 epoch 的 f1_macro 未创新高，则提前终止训练。节省算力且防    trainer.train()过拟合。
    )

    # 开始训练
    trainer.train()

    # 保存最终最优模型
    final_path = os.path.join(CONFIG["output_dir"], "best_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✅ 微调完成！最优模型已保存至: {final_path}")

if __name__ == "__main__":
    main()