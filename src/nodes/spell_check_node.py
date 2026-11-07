"""
定义拼写纠错节点， 对用户输入的问题使用Agent进行纠错
"""
from typing import List

from langchain.agents import create_agent
from langchain.tools import  tool
import re

from src.models.spell_models import CorrectionResult
from src.schema.base import BaseNode, T
from src.schema.exceptions import ValidationError
from src.schema.state import GraphState
from src.utils.llm_client_util import get_llm_client


system_prompt = """你是一个专业的中文文本纠错专家，擅长识别和修正各种类型的中文错误。
    【可用工具】
     1. check_common_errors(text): 检查文本中常见的中文输入错误模式（如拼音相似错误、形近字错误）。
     2. query_dictionary(word): 查询中文词典，验证疑似错误词汇是否为真实存在的词语。

    【错误类型】
    1. 拼写错误：拼音输入错误、形近字错误
    2. 语法错误：的/地/得混用、语序错误、成分残缺
    3. 用词不当：词语搭配错误、专业术语误用
    4. 标点错误：标点缺失、误用或冗余
    5. 语义错误：逻辑矛盾、事实错误
    【输出要求】
    1. 严格按 JSON Schema 输出，不要添加额外说明
    2. 错误位置使用字符索引（从0开始） 
    3. 修正理由需结合上下文解释 
    4. 置信度计算：高确定性错误(0.9+)，需推断错误(0.7-0.9)，不确定(<0.7) 
    型自我评估
    【特别注意】
    - 保留原文风格和语义，只修正明确错误 
    - 专有名词（人名、地名、企业名）需谨慎修正 
    - 不确定时宁可不改，置信度设为低值
"""

class SpellCheckNode(BaseNode):

    name = "spell_check_node"

    def __init__(self):
        super().__init__()
        tools = self._common_tools()
        self.agent = create_agent(
            model=get_llm_client(),
            tools=tools,
            system_prompt=system_prompt,
            response_format=CorrectionResult
        )

    # 切块，根据符号分割长段落，然后控制每个块的大小接近300
    def _split_long_text(self, text: str, max_length: int = 300) -> List[str]:
        """将长文本智能分割为语义连贯的块"""
        # 优先按句子分割（保留标点完整性）
        sentences = re.split(r'(?<=[。！？；])', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_length and current_chunk:
                chunks.append(current_chunk.strip())  # 保存当前块
                current_chunk = sentence  # 新块从当前句开始
            else:
                current_chunk += sentence  # 继续累加

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if len(chunks) > 1 else [text]

    def _merge_corrections(self, chunks, results):
        pass
    #定义智能需要的外部工具
    def _common_tools(self):
        # 专有名词验证器
        @tool
        def query_dictionary(word: str) -> str:
            """
                    查询中文词典获取词语的正确用法、近义词和常见搭配。
                    适用于验证疑似错误词汇是否为真实存在的词语。
                    """
            # 模拟词典查询（实际应用中可对接权威词典 API）
            dictionary = {
                "万豪": ("品牌名", "万豪国际酒店集团，非房企名称"),
                "万好万家": ("企业名", "浙江万好万家文化股份有限公司"),
                "东日": ("企业名", "浙江东日股份有限公司"),
                "冬日": ("普通词汇", "冬季的白天，非企业名称")
            }
            result = dictionary.get(word, ("未知词汇", "建议结合上下文判断"))
            return f"词条: {word}\n类型: {result[0]}\n说明: {result[1]}"

        # 高频错误快速拦截器
        @tool
        def check_common_errors(text: str) -> str:
            """检查文本中常见的中文输入错误模式（如拼音相似错误、形近字错误）。"""
            # 常见错误模式库：硬编码模式库（维护成本高）
            error_patterns = [
                (r"始曰苯", "使日本", "拼音输入错误"),
                (r"万豪万家", "万好万家", "企业名称混淆"),
                (r"冬日", "东日", "形近字错误（房企名称）"),
                (r"的\b", "地|得", "的/地/得混用（需结合语法判断）"),
                (r"在\b.*?在\b", "", "重复使用'在'字")
            ]
            # 改进方案：外部配置化
            # import json
            # with open("error_patterns.json") as f:
            #     error_patterns = json.load(f)  # 支持热更新

            matches = []
            for pattern, correction, error_type in error_patterns:
                if re.search(pattern, text):
                    matches.append(f"- '{pattern}' → '{correction}' ({error_type})")
                    print("检测到潜在错误模式:\n" + "\n".join(matches))

            if matches:
                return "检测到潜在错误模式:\n" + "\n".join(matches)
            return "未检测到常见错误模式"
        return [check_common_errors, query_dictionary]


    def process(self, state:GraphState) ->GraphState:
        """
        拼写纠错
        :param state:
        :return:
        """
        self.log_step(self.name, "开始进行拼写纠错")
        question = state.get("question") or ""

        if not question:
            return ValidationError("用户问题为空", self.name)

        # 考虑长文本的情况
        chunks = self._split_long_text(question)

        # 分块纠错
        results = []
        for chunk in chunks:
            try:
                response = self.agent.invoke({
                    "messages":[
                        {
                            "role":"user", "content": chunk,
                        }
                    ]
                })

                result = response.get("structured_response")
                # 设置了一个新的评分标准，用来衡量当前纠错的结果是否真的靠谱

                if result is None:
                    continue

                results.append(result)
            except Exception as e:
                self.log_step(self.name, f"当前块{chunk}纠错失败！")
                continue

        # 合并每个纠错之后的内容
        if len(results) > 1:
            #多块情况
            merged_result = self._merge_corrections(chunks, results)
        else:
            #单块情况
            merged_result = results[0]

        return state
