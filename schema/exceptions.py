"""
1.定义统一的异常错误逻辑，便于后续维护
2.定义各种需要的议程类，便于后续扩展

"""


class ProcessError(Exception):
    """提供统一的异常处理逻辑"""
    def __init__(self, message: str, node_name: str="", casue: Exception=None) -> None:
        super().__init__(message)
        self.node_name = node_name
        self.casue = casue

    def __str__(self) -> str:
        parts = []
        if self.node_name:
            parts.append(f"Node:{self.node_name}")      #节点信息
        parts.append(super().__str__())
        if self.casue:
            parts.append(f"Cause:{self.casue}")      # 错误原因
        return " ".join(parts)   #将所有的内容统一返回



class LLMError(ProcessError):
    pass


class EmbeddingError(ProcessError):
    pass

class StorageError(ProcessError):
    """存储异常"""
    pass

class MilvusError(StorageError):
    pass

class MySQLError(StorageError):
    pass

class RedisError(StorageError):
    pass

class Neo4jError(StorageError):
    pass


if __name__ == '__main__':
    try:
        raise MilvusError("milvus连接错误", node_name="spell_check_node")
    except MilvusError as e:
        print(e)