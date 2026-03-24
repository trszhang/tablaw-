"""Multi-agent parallel execution for multi-table analysis.

Each table gets a specialist agent that runs independently (in parallel).
An aggregator then synthesises all findings and marks:
  [CONSENSUS]  — conclusions most/all agents agree on  (high confidence)
  [UNCERTAIN]  — conflicting or caveat-heavy findings  (low confidence)
"""
import asyncio
from typing import AsyncGenerator, Dict, List, Optional

from agent.executor import AgentExecutor
from agent.prompt_locale import with_zh_cn_rule

_MULTI_AGENT_KEYWORDS = [
    # Chinese
    "对比", "比较", "分别", "各表", "综合", "横向", "差异", "相同", "不同",
    "两张", "三张", "多张", "所有表", "每张", "各个",
    # English
    "compare", "contrast", "overview", "across", "each table", "all tables",
    "differences", "similarities", "side by side",
]


class MultiAgentExecutor(AgentExecutor):
    """Specialist-per-table agents running in parallel, with an aggregator."""

    def should_activate(self, message: str, tables: Dict) -> bool:
        if len(tables) < 2:
            return False
        msg_lower = message.lower()
        return any(kw in msg_lower for kw in _MULTI_AGENT_KEYWORDS)

    async def execute_multi(
        self,
        message: str,
        tables: Dict,
        history: List,
        result_tables_store: Dict,
        persisted_tables: Optional[Dict] = None,
        code_tool: bool = False,
    ) -> AsyncGenerator:
        tools = self.skills.get_tool_definitions(code_tool=code_tool)

        # Announce the agent pool
        agents_info = [
            {"id": tid, "table_name": t["name"]}
            for tid, t in tables.items()
        ]
        yield {"type": "agent_pool_start", "agents": agents_info}

        # ── Parallel per-table agents ─────────────────────────────────────────
        _DONE = object()
        queue: asyncio.Queue = asyncio.Queue()

        async def run_agent(tid: str, table: Dict) -> str:
            try:
                await queue.put({
                    "type": "agent_start",
                    "agent_id": tid,
                    "table_name": table["name"],
                })
                # Scoped system prompt — shows only this table
                scoped_system = self._system_prompt({tid: table}, code_tool=code_tool)
                msgs = [{"role": "system", "content": scoped_system}]
                msgs.extend(history[-6:])
                msgs.append({
                    "role": "user",
                    "content": with_zh_cn_rule(
                        f"You are a specialist analyst assigned to the table "
                        f"'{table['name']}'.\n"
                        f"User request: {message}\n\n"
                        "Analyse this table thoroughly. Always call table_info "
                        "first. Use specific numbers and column names."
                    ),
                })

                conclusion = ""
                # Use shared result_tables_store so created tables persist
                async for event in self._agent_stream(
                    msgs,
                    tools,
                    result_tables_store,
                    result_tables_store,
                    persisted_tables=persisted_tables,
                    generated_tables=[],
                    commit_final_table=False,
                    user_message=message,
                ):
                    await queue.put({**event, "agent_id": tid})
                    if event["type"] == "final_text":
                        conclusion = event["content"]

                await queue.put({
                    "type": "agent_done",
                    "agent_id": tid,
                    "conclusion": conclusion,
                })
                return conclusion
            except Exception as exc:
                conclusion = f"(Agent error: {exc})"
                await queue.put({
                    "type": "agent_done",
                    "agent_id": tid,
                    "conclusion": conclusion,
                    "error": True,
                })
                return conclusion
            finally:
                await queue.put(_DONE)

        tasks = [
            asyncio.create_task(run_agent(tid, t))
            for tid, t in tables.items()
        ]

        done_count = 0
        n = len(tables)
        while done_count < n:
            event = await queue.get()
            if event is _DONE:
                done_count += 1
            else:
                yield event

        await asyncio.gather(*tasks, return_exceptions=True)
        conclusions: Dict[str, str] = {}
        for tid, task in zip(tables.keys(), tasks):
            try:
                conclusions[tid] = task.result() or ""
            except Exception:
                conclusions[tid] = ""

        # ── Aggregation with uncertainty markers ──────────────────────────────
        yield {"type": "aggregate_start"}
        async for event in self._run_aggregator(message, conclusions, tables):
            yield event

        await self._try_update_memory(message, tables)

    # ------------------------------------------------------------------
    # Aggregator
    # ------------------------------------------------------------------

    async def _run_aggregator(
        self,
        message: str,
        conclusions: Dict[str, str],
        tables: Dict,
    ) -> AsyncGenerator:
        clean_facts = []
        for tid, conclusion in conclusions.items():
            if not conclusion:
                continue
            clean_facts.append(
                {
                    "table_id": tid,
                    "table_name": tables.get(tid, {}).get("name", tid),
                    "analyst_fact": conclusion,
                }
            )

        if not clean_facts:
            yield {"type": "final_text", "content": "未提取到可用的客观数据事实，无法生成最终洞察。"}
            return

        insight = await self.generate_business_insight(message, clean_facts)
        yield {"type": "final_text", "content": insight}
