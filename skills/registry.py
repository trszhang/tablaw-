"""Skill registry — manages built-in and custom skills."""
import json
from pathlib import Path
from typing import Dict, List, Any

from skills.builtin import BUILTIN_SKILLS
from skills.code_skill import execute_python

# --- [新增导入] ---
from infrastructure.database_manager import DatabaseManager
from infrastructure.config import config
from skills.text_to_sql_skill import TextToSQLSkill
# ------------------

DATA_PATH = Path(__file__).parent.parent / "data" / "custom_skills.json"


class _TextToSQLLLMAdapter:
    """Sync adapter that exposes generate_text for TextToSQLSkill."""

    def __init__(self):
        from openai import OpenAI

        self._client = OpenAI(
            api_key=getattr(config, "LLM_API_KEY", ""),
            base_url=getattr(config, "LLM_BASE_URL", "https://api.openai.com/v1"),
        )
        self._model = getattr(config, "LLM_MODEL_NAME", "gpt-4o")
        self._temperature = float(getattr(config, "LLM_TEMPERATURE", 0.1))

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

# 初始化全局的 Text-to-SQL 技能实例
try:
    _db_manager = DatabaseManager(getattr(config, 'DATABASE_URL', 'sqlite:///:memory:'))
    _text_to_sql_skill = TextToSQLSkill(_db_manager, _TextToSQLLLMAdapter())
    _SQL_SKILL_READY = True
except Exception as e:
    print(f"Warning: Failed to initialize TextToSQLSkill in registry: {e}")
    _SQL_SKILL_READY = False

# OpenAI-format tool definitions for every built-in skill
BUILTIN_TOOL_DEFS = [
    # --- [新增：Text-to-SQL 技能定义 (必须加在最前面，确保它被识别)] ---
    {
        "type": "function",
        "function": {
            "name": "text_to_sql",
            "description": "Convert natural language to SQL and execute it against the enterprise database to retrieve macro business facts, trends, or when the user mentions databases/systems. Returns aggregate data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_query": {
                        "type": "string",
                        "description": "The natural language query translated into the intent for the database search."
                    }
                },
                "required": ["user_query"],
            },
        },
    },
    # -------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "table_info",
            "description": "Get metadata (shape, columns, dtypes, missing values, sample rows) for a table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string", "description": "ID of the table to inspect"},
                },
                "required": ["table_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_rows",
            "description": "Filter rows using a pandas query string (e.g. 'age > 30 and city == \"NYC\"').",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string"},
                    "condition": {"type": "string", "description": "Pandas query expression"},
                    "result_name": {"type": "string", "description": "Name for the resulting table"},
                },
                "required": ["table_id", "condition"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_columns",
            "description": "Select a subset of columns from a table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}, "description": "List of column names"},
                    "result_name": {"type": "string"},
                },
                "required": ["table_id", "columns"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate",
            "description": "Group by columns and aggregate with functions like sum, mean, count, max, min.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string"},
                    "group_by": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to group by",
                    },
                    "agg_config": {
                        "type": "object",
                        "description": "Dict of {column_name: agg_function} e.g. {\"sales\": \"sum\", \"qty\": \"mean\"}",
                    },
                    "result_name": {"type": "string"},
                },
                "required": ["table_id", "group_by", "agg_config"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sort_table",
            "description": "Sort a table by one or more columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string"},
                    "by": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                        "description": "Column name or list of column names to sort by",
                    },
                    "ascending": {"type": "boolean", "default": True},
                    "result_name": {"type": "string"},
                },
                "required": ["table_id", "by"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "merge_tables",
            "description": "Merge/join two tables (inner, left, right, outer join).",
            "parameters": {
                "type": "object",
                "properties": {
                    "left_table_id": {"type": "string"},
                    "right_table_id": {"type": "string"},
                    "on": {"type": "string", "description": "Column to join on (if same name in both)"},
                    "left_on": {"type": "string"},
                    "right_on": {"type": "string"},
                    "how": {"type": "string", "enum": ["inner", "left", "right", "outer"], "default": "inner"},
                    "result_name": {"type": "string"},
                },
                "required": ["left_table_id", "right_table_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pivot_table",
            "description": "Create a pivot table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string"},
                    "index": {
                        "oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]
                    },
                    "columns": {"type": "string"},
                    "values": {
                        "oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]
                    },
                    "aggfunc": {"type": "string", "default": "sum"},
                    "result_name": {"type": "string"},
                },
                "required": ["table_id", "index", "values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_column",
            "description": "Add a new computed column using a pandas eval expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string"},
                    "column_name": {"type": "string", "description": "Name for the new column"},
                    "expression": {
                        "type": "string",
                        "description": "Pandas eval expression, e.g. 'price * quantity'",
                    },
                    "result_name": {"type": "string"},
                },
                "required": ["table_id", "column_name", "expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_stats",
            "description": "Return descriptive statistics (count, mean, std, min, max, quartiles) for a table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string"},
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: only include these columns",
                    },
                },
                "required": ["table_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_values",
            "description": "Find rows where a column equals a value or matches a regex pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string"},
                    "column": {"type": "string"},
                    "value": {"description": "Exact value to search for"},
                    "pattern": {"type": "string", "description": "Regex pattern (case-insensitive)"},
                    "result_name": {"type": "string"},
                },
                "required": ["table_id", "column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drop_duplicates",
            "description": "Remove duplicate rows from a table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string"},
                    "subset": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: only consider these columns for duplicates",
                    },
                    "result_name": {"type": "string"},
                },
                "required": ["table_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rename_columns",
            "description": "Rename one or more columns in a table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string"},
                    "rename_map": {
                        "type": "object",
                        "description": "Dict of {old_name: new_name}",
                    },
                    "result_name": {"type": "string"},
                },
                "required": ["table_id", "rename_map"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sample_rows",
            "description": "Get a random sample of N rows from a table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string"},
                    "n": {"type": "integer", "default": 10},
                    "result_name": {"type": "string"},
                },
                "required": ["table_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "value_counts",
            "description": "Count occurrences of each unique value in a column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string"},
                    "column": {"type": "string"},
                    "result_name": {"type": "string"},
                },
                "required": ["table_id", "column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "correlation_matrix",
            "description": "Compute a Pearson correlation matrix for numeric columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string"},
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: only include these columns",
                    },
                    "result_name": {"type": "string"},
                },
                "required": ["table_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "head_rows",
            "description": "Get the first N rows of a table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string"},
                    "n": {"type": "integer", "default": 10},
                    "result_name": {"type": "string"},
                },
                "required": ["table_id"],
            },
        },
    },
]

BUILTIN_META = {
    # --- [新增 META 记录] ---
    "text_to_sql": {"description": "Query enterprise database directly via natural language.", "category": "database"},
    # ------------------------
    "table_info": {"description": "Get table metadata (shape, columns, dtypes, sample)", "category": "inspection"},
    "filter_rows": {"description": "Filter rows using a query condition", "category": "transformation"},
    "select_columns": {"description": "Select a subset of columns", "category": "transformation"},
    "aggregate": {"description": "Group by and aggregate (sum, mean, count…)", "category": "analysis"},
    "sort_table": {"description": "Sort rows by one or more columns", "category": "transformation"},
    "merge_tables": {"description": "Join/merge two tables", "category": "transformation"},
    "pivot_table": {"description": "Create a pivot table", "category": "analysis"},
    "add_column": {"description": "Add a computed column via expression", "category": "transformation"},
    "describe_stats": {"description": "Descriptive statistics (mean, std, quartiles…)", "category": "analysis"},
    "find_values": {"description": "Find rows matching a value or regex pattern", "category": "inspection"},
    "drop_duplicates": {"description": "Remove duplicate rows", "category": "cleaning"},
    "rename_columns": {"description": "Rename columns", "category": "cleaning"},
    "sample_rows": {"description": "Get a random sample of rows", "category": "inspection"},
    "value_counts": {"description": "Count unique values in a column", "category": "analysis"},
    "correlation_matrix": {"description": "Pearson correlation matrix for numeric columns", "category": "analysis"},
    "head_rows": {"description": "Get the first N rows", "category": "inspection"},
}


# Tool definition for the optional sandboxed code execution skill
CODE_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": (
            "Execute sandboxed Python code for advanced table analysis or operations. "
            "Available libraries: pd (pandas), np (numpy), math, re, json, "
            "collections, itertools, datetime, statistics. "
            "Each uploaded table is pre-loaded as a variable: accessible by its ID "
            "(e.g. 'r_abc123') AND by its sanitised name (e.g. 'sales_2023'). "
            "Before coding, inspect schema and handle dirty data defensively. "
            "For Object columns with mixed numeric/null values, use "
            "pd.to_numeric(errors='coerce') + fillna(). "
            "For date-like columns, normalize with pd.to_datetime(errors='coerce'). "
            "To produce a new result table, assign a DataFrame to the variable 'result'. "
            "Use print() for intermediate output. Do NOT import os, sys, subprocess, "
            "or any network/file-system libraries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                },
                "result_name": {
                    "type": "string",
                    "description": "Descriptive name for the result table (if 'result' is assigned)",
                },
            },
            "required": ["code"],
        },
    },
}


class SkillRegistry:
    def __init__(self):
        self._custom: List[Dict] = self._load_custom()

    def _load_custom(self) -> List[Dict]:
        if DATA_PATH.exists():
            with open(DATA_PATH) as f:
                return json.load(f)
        return []

    def _save_custom(self):
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA_PATH, "w") as f:
            json.dump(self._custom, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_all(self) -> Dict:
        # Build a lookup of parameters from the tool definitions
        params_lookup = {
            td["function"]["name"]: td["function"].get("parameters", {})
            for td in BUILTIN_TOOL_DEFS
        }
        builtin = [
            {
                "id": name,
                "name": name,
                "type": "builtin",
                **BUILTIN_META[name],
                "parameters": params_lookup.get(name, {}),
            }
            for name in BUILTIN_META
        ]
        return {"builtin": builtin, "custom": self._custom}

    def list_custom(self) -> List[Dict]:
        return self._custom

    def get_tool_definitions(self, code_tool: bool = False) -> List[Dict]:
        """Return OpenAI-format tool definitions for all enabled skills."""
        if code_tool:
            # When in code_tool mode, we inject text_to_sql alongside table_info and execute_python
            table_info_def = next(d for d in BUILTIN_TOOL_DEFS if d["function"]["name"] == "table_info")
            sql_def = next(d for d in BUILTIN_TOOL_DEFS if d["function"]["name"] == "text_to_sql")
            defs = [sql_def, table_info_def, CODE_TOOL_DEF]
        else:
            defs = list(BUILTIN_TOOL_DEFS)

        # Always register custom skills as callable tools
        for s in self._custom:
            mode_hint = "Executes Python code." if s.get("code") else "Uses an LLM sub-call guided by a custom prompt."
            defs.append({
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": f"{s['description']} ({mode_hint})",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table_id": {"type": "string", "description": "ID of the table to work with"},
                            "user_request": {"type": "string", "description": "Specific instructions or context for this invocation"},
                        },
                        "required": [],
                    },
                },
            })
        return defs

    def add_custom(self, skill_id: str, skill: Dict) -> Dict:
        entry = {"id": skill_id, "type": "custom", **skill}
        self._custom.append(entry)
        self._save_custom()
        return entry

    def update_custom(self, skill_id: str, skill: Dict) -> Dict:
        for i, s in enumerate(self._custom):
            if s["id"] == skill_id:
                self._custom[i] = {"id": skill_id, "type": "custom", **skill}
                self._save_custom()
                return self._custom[i]
        raise ValueError(f"Custom skill '{skill_id}' not found")

    def delete_custom(self, skill_id: str) -> Dict:
        before = len(self._custom)
        self._custom = [s for s in self._custom if s["id"] != skill_id]
        if len(self._custom) == before:
            raise ValueError(f"Custom skill '{skill_id}' not found")
        self._save_custom()
        return {"status": "deleted"}

    def clear_custom(self) -> Dict:
        count = len(self._custom)
        self._custom = []
        self._save_custom()
        return {"cleared": count}

    def execute_sync(self, skill_name: str, params: Dict, tables: Dict) -> Any:
        """Execute a built-in or code skill synchronously."""
        
        # --- [新增执行路由：拦截并执行 text_to_sql] ---
        if skill_name == "text_to_sql":
            if not _SQL_SKILL_READY:
                return {"text": "Error: Text-to-SQL Skill is not initialized properly. Check database connection."}
            # 调用我们在 text_to_sql_skill.py 里写的核心逻辑
            # 注意：如果你的 _text_to_sql_skill.execute 需要 llm，我们在那里面应该通过导入 llm_client 解决了
            query = params.get("user_query", "")
            return _text_to_sql_skill.execute(query)
        # ---------------------------------------------
        
        if skill_name == "execute_python":
            return execute_python(params, tables)
            
        if skill_name not in BUILTIN_SKILLS:
            raise ValueError(f"Unknown skill '{skill_name}'")
        return BUILTIN_SKILLS[skill_name](params, tables)