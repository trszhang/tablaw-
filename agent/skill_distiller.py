"""
SkillDistiller: after a task completes, review the tool call log and
try to extract a reusable custom skill.

Triggered only when >= MIN_TOOL_CALLS tool calls were made — trivial
one-shot queries are skipped entirely.
"""
import json
import re
import uuid
from typing import Dict, List, Optional

from agent.prompt_locale import with_zh_cn_rule

MIN_TOOL_CALLS = 3  # minimum tool calls before attempting distillation

_BUILTIN_NAMES = {
    "table_info", "filter_rows", "select_columns", "aggregate", "sort_table",
    "merge_tables", "pivot_table", "add_column", "describe_stats", "find_values",
    "drop_duplicates", "rename_columns", "sample_rows", "value_counts",
    "correlation_matrix", "head_rows", "execute_python",
}


class SkillDistiller:
    def __init__(self, llm, skill_registry):
        self.llm = llm
        self.skills = skill_registry

    async def try_distill(
        self, message: str, tool_calls_log: List[Dict]
    ) -> Optional[Dict]:
        """
        Analyse the tool call log for a completed task and create a new
        custom skill if a reusable pattern is found.
        Returns the created skill dict, or None.
        """
        if len(tool_calls_log) < MIN_TOOL_CALLS:
            return None

        existing_names = {s["name"] for s in self.skills.list_custom()}
        summary = self._format_tool_log(tool_calls_log)
        has_python = any(t["name"] == "execute_python" for t in tool_calls_log)
        python_hint = (
            "If execute_python was used with non-trivial code, prefer a "
            "code-mode skill — generalise the code so it works on any table."
            if has_python else ""
        )

        builtin_str = ", ".join(sorted(_BUILTIN_NAMES))
        existing_str = ", ".join(sorted(existing_names)) if existing_names else "none"

        prompt = with_zh_cn_rule(f"""You are a skill-extraction assistant for a data analysis tool.

A user just completed this task:
"{message}"

Tool calls made (in order):
{summary}

Already available built-in skills (do NOT recreate): {builtin_str}
Already saved custom skills (do NOT duplicate): {existing_str}

{python_hint}

Decide: is there a REUSABLE, GENERALIZABLE skill worth saving?

GOOD candidate:
- Addresses a recurring data-analysis pattern (e.g. profit margin ranking,
  top-N per category, cohort retention, KPI report with multiple metrics)
- Can be parameterised — not hard-coded to one specific dataset or column name
- Adds meaningful value beyond a single built-in skill

BAD candidate:
- One-off task specific to this exact dataset / columns
- Duplicate or near-duplicate of an existing skill
- Trivially simple (single filter, sort, or lookup)

If a good candidate exists, return:
{{
  "create": true,
  "name": "descriptive_snake_case_name",
  "description": "One sentence: what it does and when to use it.",
  "mode": "code",
  "code": "# Generalised Python code.\\n# Access tables via the 'tables' dict: tables[tid]['df']\\n# Assign final DataFrame to 'result'\\n..."
}}

OR for a prompt-based skill:
{{
  "create": true,
  "name": "descriptive_snake_case_name",
  "description": "One sentence: what it does and when to use it.",
  "mode": "prompt",
  "prompt": "Detailed reusable prompt template. Use {{{{table_name}}}} and {{{{user_request}}}} as placeholders."
}}

If no good candidate: {{"create": false}}

Output ONLY valid JSON, no other text.""")

        try:
            resp = await self.llm.chat([{"role": "user", "content": prompt}])
            raw = re.sub(
                r"^```(?:json)?\s*|\s*```$", "",
                (resp.content or "").strip(),
            )
            result = json.loads(raw)

            if not result.get("create"):
                return None

            name = (result.get("name") or "").strip()
            description = (result.get("description") or "").strip()
            if not name or not description:
                return None
            if name in existing_names or name in _BUILTIN_NAMES:
                return None

            mode = result.get("mode", "prompt")
            code = result.get("code") if mode == "code" else None
            prompt_text = result.get("prompt") if mode == "prompt" else None

            skill_id = uuid.uuid4().hex[:8]
            skill_data = {
                "name": name,
                "description": description,
                "prompt": prompt_text or "",
                "code": code,
                "parameters": {},
            }
            return self.skills.add_custom(skill_id, skill_data)
        except Exception:
            return None

    def _format_tool_log(self, tool_calls_log: List[Dict]) -> str:
        lines = []
        for entry in tool_calls_log[:25]:
            name = entry.get("name", "?")
            params = entry.get("params", {})
            result_preview = (entry.get("result") or "")[:150]
            compact = {
                k: (v[:80] + "…" if isinstance(v, str) and len(v) > 80 else v)
                for k, v in params.items()
            }
            lines.append(f"  [{name}] {json.dumps(compact, ensure_ascii=False)}")
            if result_preview:
                lines.append(f"    → {result_preview}")
        return "\n".join(lines)
