import logging
# 这里导入你上面发的那个全局配置
from infrastructure.config import config  

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """
    专门负责数据库连接参数的解析、构建与调优配置
    """
    def __init__(self):
        # 继承从 setting.txt 里读到的 URL
        self.raw_url = getattr(config, 'DATABASE_URL', 'sqlite:///:memory:')
        
        # --- 企业级连接池调优参数 ---
        # 应对 Agent 并发请求和防止长时间闲置导致的数据库断连
        self.POOL_SIZE = 10
        self.MAX_OVERFLOW = 20
        self.POOL_RECYCLE = 3600  # 核心机制：1小时主动回收一次连接，防止 MySQL 8小时断开报错
        self.POOL_PRE_PING = True # 核心机制：每次向数据库发请求前，先 ping 一下确认连接存活

    def get_connection_url(self) -> str:
        """
        获取数据库连接字符串
        """
        if not self.raw_url or "sqlite:///:memory:" in self.raw_url:
            logger.warning("TabClaw Agent is using an in-memory SQLite database. Data will not be persisted.")
        return self.raw_url

    def get_engine_kwargs(self) -> dict:
        """
        动态构造 SQLAlchemy Engine 的初始化参数。
        """
        kwargs = {
            "pool_pre_ping": self.POOL_PRE_PING,
        }
        
        # SQLite 内存库比较特殊，它不支持连接池大小配置，所以需要做分支隔离
        if "sqlite" not in self.raw_url.lower():
            kwargs["pool_size"] = self.POOL_SIZE
            kwargs["max_overflow"] = self.MAX_OVERFLOW
            kwargs["pool_recycle"] = self.POOL_RECYCLE
            
        return kwargs

# 实例化一个供其他模块直接使用的对象
db_config = DatabaseConfig()