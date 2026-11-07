######################## 后续不需要测试该脚本可以删除 ##########
import sys
from pathlib import Path
PROJECT_ROOT= Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
#########################################################


import os, logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from langchain_openai import ChatOpenAI
from src.schema.config import get_config
import re
cache_llm_client = {}

config = get_config()

def get_llm_client(mode_name: str = None, temperature: float = 0.0, response_format: bool = False):
    """
    Returns: 返回LLM客户端对象
    # 自己加的提供:dashscope
    # 兼容OpenAI: LangChain LangGraph(集成OpenAI)
    缓存的对象是：client
    缓存的key: 不同的节点用不同的模型以及同一个节点用不同响应格式
    """

    # 1. 获取模型的名字
    model_name = mode_name or config.default_model
    api_key = config.openai_api_key
    api_base = config.openai_api_base

    cache_key = (mode_name, response_format)  # 复合缓存key(a,b)

    # 2. 缓存命中 直接返回
    if cache_key in cache_llm_client:
        return cache_llm_client[cache_key]

    # 3. 返回的内容格式
    model_kwargs = {}
    if response_format:
        model_kwargs['response_format'] = {"type": "json_object"}
    try:
        # 4. 定义模型实例
        client = ChatOpenAI(
            model_name=model_name,
            openai_api_key=api_key,
            openai_api_base=api_base,
            temperature=temperature,
            extra_body={"enable_thinking": False},
            model_kwargs=model_kwargs
        )

        # 5. 同步数据
        cache_llm_client[cache_key] = client

        # 6. 返回
        return client
    except Exception as e:
        logger.error(f"LLM客户端创建失败,原因:{str(e)}")
        return None


def extract_json_from_markdown(content: str) -> str:
    """
    从可能包含 Markdown 代码块标记的字符串中提取纯 JSON 内容
    """
    # 使用正则表达式匹配第一个 '{' 到最后一个 '}' 之间的所有内容
    # re.DOTALL 确保 '.' 可以匹配换行符
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        return match.group(0)

    # 如果没有找到 {}，尝试清理 markdown 标记
    clean_content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content.strip(), flags=re.MULTILINE)
    return clean_content


if __name__ == '__main__':
    llm_client = get_llm_client(mode_name="gpt-4o", response_format=True)

    import json

    try:
        # ai_message = llm_client.invoke("你好，请问您是谁?")
        # print(ai_message.content)
        # 使用本质发送请求（底层将model_kwargs的所有参数都在发送请求之前拼接到请求体身上）
        ai_message = llm_client.invoke("您好，请给我讲一个故事，返回json格式：{\"key\":\"value\"}")

        # 先清理 Markdown 标记，再解析 JSON
        clean_json_str = extract_json_from_markdown(ai_message.content)
        json_object = json.loads(clean_json_str)
        print(json_object)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败: {e}")
        logger.error(f"清理后的字符串内容为: {clean_json_str}")