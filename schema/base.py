"""
定义各个节点中都存在的一些共性能力
1、日志的输出
2、配置信息
3、异常处理
"""
import logging
from abc import ABC, abstractmethod
from typing import TypeVar, Optional

from schema.config import Config, get_config
from schema.exceptions import ProcessError

#泛型
T = TypeVar('T')   #类似Any

class BaseNode(ABC):

    name: str ="base_node"  #节点名称，子类可以覆盖

    def __init__(self, config: Optional[Config]=None) -> None:
        self.config = config or get_config()   #配置对象
        self.logger = logging.getLogger(f"{self.name}")  #日志对象

    def __call__(self, state: T) -> T:    #共性问题处理
        try:
            self.logger.info(f"--------{self.name} 开始执行--------")

            result = self.process(state)

            self.logger.info(f"--------{self.name} 执行完成--------")

            return result
        except ProcessError as e:
            raise e
        except Exception as e:
            self.logger.error(f"节点执行失败，错误信息为：{e}")
            raise ProcessError(
                message=str(e),
                node_name=self.name,
                cause=e
            )

    def log_step(self, step_name: str, message: str):
        """日志输出"""
        msg = f"{step_name}"
        if message:
            msg += f":{message}"
        self.logger.info(msg)


    # 要求每个节点自己来完成特性能力的
    @abstractmethod
    def process(self, state: T) -> T:
        """每个子节点需要处理的特定业务逻辑"""
        pass



def setup_logging(level:Optional[int]=logging.INFO) -> None:
    """配置日志的级别"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )




