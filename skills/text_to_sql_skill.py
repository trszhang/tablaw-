import logging
import pandas as pd
from typing import Dict, Any
from sqlalchemy import text
from infrastructure.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class TextToSQLSkill:
    def __init__(self, db_manager: DatabaseManager, llm_client):
        """
        初始化 Text-to-SQL 技能组件
        :param db_manager: 之前写好的数据库管理器实例
        :param llm_client: 你的大模型调用客户端 (例如 OpenAI client 或自定义的封装)
        """
        self.db = db_manager
        self.llm_client = llm_client

    def execute(self, user_query: str) -> Dict[str, Any]:
        """
        核心执行逻辑：提取 Schema -> 组装 Prompt -> 生成 SQL -> 安全执行 -> 返回结果
        """
        # 1. 提取带外键的 Schema
        schema_info = self.db.get_schema_info()
        if "Error" in schema_info:
            return {"status": "error", "message": schema_info}

        # 2. 组装强约束 Prompt (内置业务字段过滤逻辑)
        system_prompt = f"""
        你是一个资深的 PostgreSQL/MySQL 数据分析专家。你的任务是将用户的自然语言问题转化为准确的 SQL 查询语句。
        
        【数据库结构 (包含表、字段、外键)】:
        {schema_info}
        
        【核心业务约束 (严格遵守)】:
        1. 字段过滤：请仔细分辨时间字段。如果遇到类似 "导出时间" 或 "数据更新时间" 的系统自动生成字段，请务必忽略！进行时序分析或条件筛选时，必须且只能关注真实的业务 "账期" 字段。
        2. 只读操作：只能使用 SELECT 语句。绝对禁止使用 INSERT, UPDATE, DELETE, DROP 等修改数据的操作。
        3. 结果优化：如果查询可能返回大量数据，请默认加上 LIMIT 100 限制。
        
        请直接输出 SQL 语句，不要包含任何 markdown 格式符（如 ```sql ）或多余的解释说明。
        """

        # 3. 调用 LLM 生成 SQL 
        try:
            generated_sql = self.llm_client.generate_text(
                system_prompt=system_prompt, 
                user_prompt=user_query
            ).strip()
            
            # 清理可能的 markdown 格式残留
            generated_sql = generated_sql.replace("```sql", "").replace("```", "").strip()
            logger.info(f"Generated SQL: {generated_sql}")
            
        except Exception as e:
            return {"status": "error", "message": f"LLM SQL Generation failed: {e}"}

        # 4. 在只读沙盒中执行 SQL
        return self._execute_readonly_sql(generated_sql)

    def _execute_readonly_sql(self, sql_query: str) -> Dict[str, Any]:
        """安全执行生成的 SQL，并转为 Pandas DataFrame 处理"""
        # 简单安全校验：拦截明显的写操作词汇
        forbidden_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'TRUNCATE']
        if any(keyword in sql_query.upper() for keyword in forbidden_keywords):
            return {"status": "error", "message": "Security Alert: Non-SELECT queries are forbidden."}

        try:
            # 使用只读事务执行
            with self.db.engine.connect() as conn:
                # 显式开启事务，并在结束后回滚，确保绝对的只读安全
                with conn.begin():
                    # 使用 pandas 直接读取 SQL 结果为 DataFrame
                    df = pd.read_sql(text(sql_query), conn)
                    # 强制回滚，防止任何意外的修改
                    conn.rollback()

            return {
                "status": "success",
                "sql_used": sql_query,
                "data_preview": df.head(10).to_dict(orient="records"), # 仅返回前10行给大模型用于下一步总结
                "data_summary": f"Query returned {len(df)} rows and {len(df.columns)} columns."
            }

        except Exception as e:
            logger.error(f"SQL Execution Error: {e}")
            return {"status": "error", "message": f"Database execution failed: {e}"}