import logging
from typing import Dict, List, Optional
from sqlalchemy import create_engine, inspect, MetaData
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.exc import SQLAlchemyError

# 配置简单的日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_url: str):
        """
        初始化数据库管理器
        :param db_url: 数据库连接字符串，例如：
                       MySQL: mysql+pymysql://user:password@host:port/dbname
                       PostgreSQL: postgresql+psycopg2://user:password@host:port/dbname
        """
        self.db_url = db_url
        try:
            # pool_pre_ping=True 会在每次从连接池获取连接时测试连接是否存活，防止 MySQL 8 小时断连问题
            self.engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=3600)
            logger.info("Database engine initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise

    def test_connection(self) -> bool:
        """
        测试数据库连接是否连通
        """
        try:
            with self.engine.connect() as conn:
                logger.info("Database connection test passed.")
                return True
        except SQLAlchemyError as e:
            logger.error(f"Database connection failed: {e}")
            return False

    def get_schema_info(self, include_tables: Optional[List[str]] = None) -> str:
        """
        提取数据库的 DDL/Schema 元数据（包含表名、列名、数据类型和外键关联），
        并格式化为对 LLM 非常友好的纯文本。
        
        :param include_tables: 允许指定只提取特定的表。如果为 None，则提取所有表。
        """
        if not self.test_connection():
            return "Error: Unable to connect to the database."

        inspector: Inspector = inspect(self.engine)
        all_table_names = inspector.get_table_names()
        
        # 过滤需要查询的表
        tables_to_inspect = include_tables if include_tables else all_table_names
        schema_text_blocks = []

        for table_name in tables_to_inspect:
            if table_name not in all_table_names:
                continue

            table_desc = [f"Table: {table_name}"]
            
            # 1. 提取列信息 (列名 + 数据类型)
            try:
                columns = inspector.get_columns(table_name)
                for col in columns:
                    col_name = col['name']
                    col_type = str(col['type'])
                    table_desc.append(f"  - Column: {col_name} ({col_type})")
            except Exception as e:
                logger.warning(f"Could not get columns for table {table_name}: {e}")

            # 2. 提取外键信息 (用于多表 Join 的推理)
            try:
                fks = inspector.get_foreign_keys(table_name)
                for fk in fks:
                    # 格式: Foreign Key: 当前表的列 -> 目标表.目标列
                    constrained_cols = ", ".join(fk['constrained_columns'])
                    referred_table = fk['referred_table']
                    referred_cols = ", ".join(fk['referred_columns'])
                    table_desc.append(f"  * Foreign Key: [{constrained_cols}] references [{referred_table}.{referred_cols}]")
            except Exception as e:
                logger.warning(f"Could not get foreign keys for table {table_name}: {e}")

            schema_text_blocks.append("\n".join(table_desc))

        # 将所有表结构拼接为一个完整的字符串
        final_schema_prompt = "\n\n".join(schema_text_blocks)
        return final_schema_prompt

# ==========================================
# 本地测试代码 (仅测试时运行)
# ==========================================
if __name__ == "__main__":
    # 替换为企业测试数据库 URL
    TEST_URL = "sqlite:///:memory:"  # 仅作占位符，实际请换成 MySQL/PG 链接
    
    db_manager = DatabaseManager(TEST_URL)
    if db_manager.test_connection():
        schema = db_manager.get_schema_info()
        print("\n--- Extracted Database Schema for LLM ---\n")
        print(schema)