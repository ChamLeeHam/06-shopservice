"""
1。定义统一的异常错误逻辑，便于后续维护
2。定义各种需要的异常类，便于后续扩展
"""
class ProcessError(Exception):
    """提供统一的异常处理逻辑"""
    def __init__(self, message: str,node_name:str="", cause: Exception=None) -> None:
        super().__init__(message)
        self.node_name = node_name
        self.cause = cause

    def __str__(self) -> str:
        parts = []
        if self.node_name:
            parts.append(f"Node:{self.node_name}") # 节点信息
        parts.append(super().__str__()) # message信息
        if self.cause:
            parts.append(f"Cause:{self.cause}") # 错误原因
        return " ".join(parts) # 将所有的内容统一返回

class LLMError(ProcessError):
    pass

class EmbeddingError(ProcessError):
    pass

class StorageError(ProcessError):
    """存储异常"""
    pass

class MilvusError(StorageError):
    """milvus向量数据库的异常"""
    pass

class MySQLError(StorageError):
    pass

class RedisError(StorageError):
    pass

class Neo4jError(StorageError):
    """图谱的存储异常"""
    pass

class ValidationError(StorageError):
    """数据验证异常"""
    pass

if __name__ == "__main__":
    try:
        raise MilvusError("milvus连接错误", "spell_check_node")
    except MilvusError as e:
        print(e)