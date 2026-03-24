"""Built-in table skills implemented with pandas."""
import json
import re
import pandas as pd
import numpy as np
from typing import Dict, Any


def _get_table(tables: Dict, table_id: str) -> pd.DataFrame:
    if table_id not in tables:
        avail = list(tables.keys())
        raise ValueError(f"Table '{table_id}' not found. Available: {avail}")
    return tables[table_id]["df"].copy()


def _safe_name(tables: Dict, result_name: str, default: str) -> str:
    return result_name if result_name else default


# -----------------------------------------------------------------------
# Skill implementations
# -----------------------------------------------------------------------

def table_info(params: Dict, tables: Dict) -> str:
    """Return metadata and sample rows for a table."""
    tid = params["table_id"]
    df = _get_table(tables, tid)
    name = tables[tid].get("name", tid)
    info = {
        "name": name,
        "table_id": tid,
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "sample_rows": df.head(5).fillna("").to_dict("records"),
    }
    return json.dumps(info, default=str, ensure_ascii=False)


def filter_rows(params: Dict, tables: Dict) -> Dict:
    """Filter rows using a pandas query string."""
    tid = params["table_id"]
    condition = params["condition"]
    result_name = params.get("result_name", f"filtered_{tables[tid]['name']}")
    df = _get_table(tables, tid)
    result = df.query(condition, engine="python")
    return {"df": result, "name": result_name}


def select_columns(params: Dict, tables: Dict) -> Dict:
    """Select a subset of columns."""
    tid = params["table_id"]
    columns = params["columns"]
    result_name = params.get("result_name", f"selected_{tables[tid]['name']}")
    df = _get_table(tables, tid)
    result = df[columns]
    return {"df": result, "name": result_name}


def aggregate(params: Dict, tables: Dict) -> Dict:
    """Group by columns and aggregate."""
    tid = params["table_id"]
    group_by = params["group_by"]
    agg_config = params["agg_config"]   # e.g. {"sales": "sum", "qty": "mean"}
    result_name = params.get("result_name", f"agg_{tables[tid]['name']}")
    df = _get_table(tables, tid)
    result = df.groupby(group_by).agg(agg_config).reset_index()
    result.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in result.columns]
    return {"df": result, "name": result_name}


def sort_table(params: Dict, tables: Dict) -> Dict:
    """Sort a table by one or more columns."""
    tid = params["table_id"]
    by = params["by"]
    ascending = params.get("ascending", True)
    result_name = params.get("result_name", f"sorted_{tables[tid]['name']}")
    df = _get_table(tables, tid)
    result = df.sort_values(by=by, ascending=ascending)
    return {"df": result, "name": result_name}


def merge_tables(params: Dict, tables: Dict) -> Dict:
    """Merge/join two tables."""
    left_id = params["left_table_id"]
    right_id = params["right_table_id"]
    on = params.get("on")
    left_on = params.get("left_on")
    right_on = params.get("right_on")
    how = params.get("how", "inner")
    result_name = params.get("result_name", "merged_table")

    left_df = _get_table(tables, left_id)
    right_df = _get_table(tables, right_id)

    if on:
        result = pd.merge(left_df, right_df, on=on, how=how)
    elif left_on and right_on:
        result = pd.merge(left_df, right_df, left_on=left_on, right_on=right_on, how=how)
    else:
        raise ValueError("Provide 'on' or both 'left_on' and 'right_on'")
    return {"df": result, "name": result_name}


def pivot_table(params: Dict, tables: Dict) -> Dict:
    """Create a pivot table."""
    tid = params["table_id"]
    index = params["index"]
    columns = params.get("columns")
    values = params["values"]
    aggfunc = params.get("aggfunc", "sum")
    result_name = params.get("result_name", f"pivot_{tables[tid]['name']}")

    df = _get_table(tables, tid)
    result = pd.pivot_table(
        df, index=index, columns=columns, values=values, aggfunc=aggfunc
    ).reset_index()
    result.columns = [str(c) for c in result.columns]
    return {"df": result, "name": result_name}


def add_column(params: Dict, tables: Dict) -> Dict:
    """Add a computed column using a pandas eval expression."""
    tid = params["table_id"]
    col_name = params["column_name"]
    expression = params["expression"]   # e.g. "price * quantity"
    result_name = params.get("result_name", tables[tid]["name"])

    df = _get_table(tables, tid)
    df[col_name] = df.eval(expression)
    return {"df": df, "name": result_name}


def describe_stats(params: Dict, tables: Dict) -> str:
    """Return descriptive statistics for a table."""
    tid = params["table_id"]
    columns = params.get("columns")
    df = _get_table(tables, tid)
    if columns:
        df = df[columns]
    stats = df.describe(include="all").fillna("").to_dict()
    return json.dumps(stats, default=str, ensure_ascii=False)


def find_values(params: Dict, tables: Dict) -> Dict:
    """Find rows where a column contains a specific value or matches a pattern."""
    tid = params["table_id"]
    column = params["column"]
    value = params.get("value")
    pattern = params.get("pattern")
    result_name = params.get("result_name", f"found_{tables[tid]['name']}")

    df = _get_table(tables, tid)
    if pattern:
        mask = df[column].astype(str).str.contains(pattern, case=False, na=False)
    else:
        mask = df[column] == value
    result = df[mask]
    return {"df": result, "name": result_name}


def drop_duplicates(params: Dict, tables: Dict) -> Dict:
    """Remove duplicate rows."""
    tid = params["table_id"]
    subset = params.get("subset")
    result_name = params.get("result_name", f"dedup_{tables[tid]['name']}")
    df = _get_table(tables, tid)
    result = df.drop_duplicates(subset=subset)
    return {"df": result, "name": result_name}


def rename_columns(params: Dict, tables: Dict) -> Dict:
    """Rename columns in a table."""
    tid = params["table_id"]
    rename_map = params["rename_map"]  # {"old_name": "new_name", ...}
    result_name = params.get("result_name", tables[tid]["name"])
    df = _get_table(tables, tid)
    result = df.rename(columns=rename_map)
    return {"df": result, "name": result_name}


def sample_rows(params: Dict, tables: Dict) -> Dict:
    """Get a random sample of rows."""
    tid = params["table_id"]
    n = params.get("n", 10)
    result_name = params.get("result_name", f"sample_{tables[tid]['name']}")
    df = _get_table(tables, tid)
    result = df.sample(min(n, len(df)))
    return {"df": result, "name": result_name}


def value_counts(params: Dict, tables: Dict) -> Dict:
    """Count occurrences of each unique value in a column."""
    tid = params["table_id"]
    column = params["column"]
    result_name = params.get("result_name", f"counts_{column}")
    df = _get_table(tables, tid)
    result = df[column].value_counts().reset_index()
    result.columns = [column, "count"]
    return {"df": result, "name": result_name}


def correlation_matrix(params: Dict, tables: Dict) -> Dict:
    """Compute correlation matrix for numeric columns."""
    tid = params["table_id"]
    columns = params.get("columns")
    result_name = params.get("result_name", "correlation_matrix")
    df = _get_table(tables, tid)
    if columns:
        df = df[columns]
    result = df.select_dtypes(include=[np.number]).corr().reset_index()
    return {"df": result, "name": result_name}


def head_rows(params: Dict, tables: Dict) -> Dict:
    """Get the first N rows of a table."""
    tid = params["table_id"]
    n = params.get("n", 10)
    result_name = params.get("result_name", f"head_{tables[tid]['name']}")
    df = _get_table(tables, tid)
    return {"df": df.head(n), "name": result_name}


# -----------------------------------------------------------------------
# Registry helper
# -----------------------------------------------------------------------

BUILTIN_SKILLS = {
    "table_info": table_info,
    "filter_rows": filter_rows,
    "select_columns": select_columns,
    "aggregate": aggregate,
    "sort_table": sort_table,
    "merge_tables": merge_tables,
    "pivot_table": pivot_table,
    "add_column": add_column,
    "describe_stats": describe_stats,
    "find_values": find_values,
    "drop_duplicates": drop_duplicates,
    "rename_columns": rename_columns,
    "sample_rows": sample_rows,
    "value_counts": value_counts,
    "correlation_matrix": correlation_matrix,
    "head_rows": head_rows,
}
