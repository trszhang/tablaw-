import json
import re
from typing import Dict, List

from agent.prompt_locale import with_zh_cn_rule


class Planner:
    def __init__(self, llm, memory_manager):
        self.llm = llm
        self.memory = memory_manager

    async def generate(self, message: str, tables: Dict) -> Dict:
        """Generate a structured execution plan for the user's request."""
        table_lines = []
        for tid, t in tables.items():
            df = t["df"]
            cols = ", ".join(df.columns[:8].tolist())
            if len(df.columns) > 8:
                cols += f" (+{len(df.columns) - 8} more)"
            table_lines.append(f"- ID: `{tid}` | Name: '{t['name']}' | {len(df)} rows | Columns: {cols}")
        tables_text = "\n".join(table_lines) if table_lines else "No tables uploaded."

        mem_text = self.memory.get_relevant(message)

        prompt = with_zh_cn_rule(f"""You are TabClaw, an AI assistant for table analysis and data manipulation.

## Available Tables
{tables_text}

## User Memory
{mem_text}

## User Request
{message}

Generate a step-by-step execution plan. Return ONLY valid JSON with this exact structure:
{{
  "title": "Brief plan title",
  "steps": [
    {{"id": 1, "description": "Clear description of what to do in this step"}},
    {{"id": 2, "description": "..."}},
    ...
  ]
}}

Rules:
- Each step should be one concrete action (filter, aggregate, sort, merge, analyze, etc.)
- 2-8 steps is ideal
- Steps should be in logical execution order
- Descriptions should be clear enough for an AI to execute without ambiguity
- Output ONLY the JSON, no markdown fences, no extra text.""")

        try:
            resp = await self.llm.chat([{"role": "user", "content": prompt}])
            content = (resp.content or "").strip()
            # Strip markdown code fences if present
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            plan = json.loads(content)
            if "steps" not in plan:
                raise ValueError("Missing steps")
            return plan
        except Exception as e:
            # Fallback plan
            return {
                "title": "Execute request",
                "steps": [
                    {"id": 1, "description": f"Analyse the available tables and understand the data structure"},
                    {"id": 2, "description": f"Execute the user request: {message}"},
                    {"id": 3, "description": "Summarise and present the results clearly"},
                ],
            }

    async def check_clarification(self, message: str, tables: Dict) -> Dict:
        """Return clarification options if the request is ambiguous, else needs_clarification=false."""
        if not tables:
            return {"needs_clarification": False}

        table_lines = []
        for tid, t in tables.items():
            df = t["df"]
            cols = ", ".join(df.columns[:6].tolist())
            table_lines.append(f"- '{t['name']}' | {len(df)} rows | Columns: {cols}")
        tables_text = "\n".join(table_lines)

        prompt = with_zh_cn_rule(f"""A user sent this data analysis request:

"{message}"

Available tables:
{tables_text}

Decide if this request is genuinely ambiguous — i.e. could lead to meaningfully DIFFERENT analyses depending on intent.

Return ONLY valid JSON in one of these two forms:

{{"needs_clarification": false}}

OR:

{{"needs_clarification": true, "question": "Short clarifying question (match user's language)", "options": ["Option A", "Option B", "Option C"]}}

Rules:
- Only return true if ambiguity would lead to VERY different outputs
- 2–4 options, each concrete and actionable (under 20 words)
- Match the language of the user's request (Chinese/English)
- For specific, unambiguous requests always return false
- Output ONLY valid JSON, no other text""")

        try:
            resp = await self.llm.chat([{"role": "user", "content": prompt}])
            content = re.sub(
                r"^```(?:json)?\s*|\s*```$", "",
                (resp.content or "").strip(),
            )
            result = json.loads(content)
            if not isinstance(result.get("needs_clarification"), bool):
                return {"needs_clarification": False}
            if result["needs_clarification"] and (
                not result.get("options") or len(result["options"]) < 2
            ):
                return {"needs_clarification": False}
            return result
        except Exception:
            return {"needs_clarification": False}
