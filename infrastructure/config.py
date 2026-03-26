import os
import logging

# 配置全局日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        setting_path = os.path.join(base_dir, "setting.txt")

        self.settings = {}
        
        # 解析 setting.txt
        if os.path.exists(setting_path):
            with open(setting_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # 忽略空行和以 # 开头的注释
                    if not line or line.startswith("#"):
                        continue
                    # 只分割第一个等号，防止 value 中本身包含等号
                    if "=" in line:
                        key, value = line.split("=", 1)
                        self.settings[key.strip()] = value.strip()
            logger.info("Successfully loaded configuration from setting.txt")
        else:
            logger.warning(f"Configuration file not found at {setting_path}. Using empty defaults.")

        # --- LLM 模型配置 (严格对齐你的 setting.txt 字段) ---
        self.LLM_API_KEY = self.settings.get("API_KEY", "")
        self.LLM_BASE_URL = self.settings.get("BASE_URL", "https://api.openai.com/v1")
        self.LLM_MODEL_NAME = self.settings.get("MODEL", "gpt-4o")
        
        # 针对 Text-to-SQL 任务强制设定低温度，保证 SQL 生成的稳定性
        self.LLM_TEMPERATURE = 0.1 

        # --- 数据库配置 (建议你也加到 setting.txt 里) ---
        # 如果 setting.txt 里没写 DATABASE_URL，默认用内存 SQLite 防报错
        self.DATABASE_URL = self.settings.get("DATABASE_URL", "sqlite:///:memory:")

config = Config()