CRITICAL_ZH_CN_OUTPUT_RULE = (
    "CRITICAL: 你的所有思考过程 (Thought) 和最终回复 (Response) 必须完全使用简体中文 "
    "(zh-CN)。严禁中英文夹杂。"
)


def with_zh_cn_rule(prompt: str) -> str:
    """Append the highest-priority zh-CN output rule to prompt templates."""
    return f"{prompt.rstrip()}\n\n{CRITICAL_ZH_CN_OUTPUT_RULE}"
