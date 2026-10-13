"""Data processing and logic tools."""
import collections
import itertools
import logging
import json
import io
import math
import re
import statistics
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional

from config.settings import settings
from toolkit.base import BaseTool, enrich_tool_result

logger = logging.getLogger(__name__)


def _coerce_comparable(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return value
    try:
        return float(text)
    except (TypeError, ValueError):
        return value


def _matches_constraint(actual: Any, expected: Any) -> bool:
    """Match exact values plus explicit deterministic comparison operators."""
    if isinstance(expected, dict):
        for operator, operand in expected.items():
            left = _coerce_comparable(actual)
            right = _coerce_comparable(operand)
            if operator in {"$eq", "eq"} and left != right:
                return False
            if operator in {"$ne", "ne"} and left == right:
                return False
            if operator in {"$in", "in"} and actual not in (operand or []):
                return False
            if operator in {"$contains", "contains"} and str(operand).lower() not in str(actual).lower():
                return False
            if operator in {"$gte", "gte"} and left < right:
                return False
            if operator in {"$gt", "gt"} and left <= right:
                return False
            if operator in {"$lte", "lte"} and left > right:
                return False
            if operator in {"$lt", "lt"} and left >= right:
                return False
        return True
    if isinstance(expected, (list, tuple, set)):
        return actual in expected
    left = _coerce_comparable(actual)
    right = _coerce_comparable(expected)
    return left == right


def _temporal_bounds(item: Any) -> Optional[tuple[float, float]]:
    if not isinstance(item, dict):
        return None
    start = item.get("start", item.get("t_start"))
    end = item.get("end", item.get("t_end"))
    if start is None or end is None:
        value = item.get("time_range", item.get("time"))
        text = str(value or "").strip()
        if "-" not in text:
            return None
        start, end = text.split("-", 1)
    try:
        left = float(start)
        right = float(end)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(left) or not math.isfinite(right):
        return None
    return (min(left, right), max(left, right))


class Filtering(BaseTool):
    """Filter results based on metadata constraints."""
    
    name: str = "Filtering"
    description: str = (
        "Filter retrieval results based on structured metadata. Supports exact values and explicit "
        "operators ($eq/$ne/$in/$contains/$gte/$gt/$lte/$lt); it does not search the video corpus."
    )
    tags: list = ["base", "filter", "metadata", "selection"]
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "description": "Data to filter"
            },
            "metadata_constraints": {
                "type": "object",
                "description": "Metadata constraints as key-value pairs"
            }
        },
        "required": ["data", "metadata_constraints"]
    }
    
    def run(self, data: List[Dict[str, Any]], metadata_constraints: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """Execute filtering.
        
        Args:
            data: Data to filter
            metadata_constraints: Filtering criteria
            **kwargs: Additional arguments
            
        Returns:
            Filtered data
        """
        filtered = []
        
        for item in data:
            matches = True
            for key, value in metadata_constraints.items():
                if key not in item or not _matches_constraint(item[key], value):
                    matches = False
                    break
            
            if matches:
                filtered.append(item)
        
        logger.info(f"Filtering: {len(data)} -> {len(filtered)} items with constraints {metadata_constraints}")
        return filtered


class Aggregation(BaseTool):
    """Aggregate and summarize retrieval results."""
    
    name: str = "Aggregation"
    description: str = (
        "Perform statistical aggregation on retrieval results. "
        "Supports counting, sorting by confidence/time, finding extremes (e.g., longest segment), "
        "and temporal fusion (merging overlapping or adjacent time ranges). Invalid operations are "
        "reported instead of silently returning the unmodified input."
    )
    tags: list = ["base", "aggregation", "composite"]
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "description": "Data to aggregate"
            },
            "operation": {
                "type": "string",
                "enum": ["count", "sort", "max", "min", "merge_temporal"],
                "description": "Aggregation operation"
            },
            "target_field": {
                "type": "string",
                "description": "Field to operate on",
                "default": None
            }
        },
        "required": ["data", "operation"]
    }
    
    def run(self, data: List[Any], operation: str, target_field: Optional[str] = None, **kwargs) -> Any:
        """Execute aggregation.
        
        Args:
            data: Data to aggregate
            operation: Type of aggregation
            target_field: Field to aggregate on
            **kwargs: Additional arguments
            
        Returns:
            Aggregation result
        """
        if operation == "count":
            result = len(data)
        
        elif operation in {"sort", "max", "min"}:
            if not target_field:
                return enrich_tool_result(
                    {
                        "status": "no_valid_result",
                        "operation": operation,
                        "result": f"Aggregation operation={operation!r} requires target_field.",
                    },
                    tool_name=self.name,
                )
            if not data:
                return enrich_tool_result(
                    {
                        "status": "no_valid_result",
                        "operation": operation,
                        "target_field": target_field,
                        "result": "Aggregation received no rows to process.",
                    },
                    tool_name=self.name,
                )
            if not all(isinstance(item, dict) and target_field in item for item in data):
                return enrich_tool_result(
                    {
                        "status": "no_valid_result",
                        "operation": operation,
                        "target_field": target_field,
                        "result": (
                            f"Aggregation cannot {operation} because target_field={target_field!r} "
                            "is absent from one or more input rows."
                        ),
                    },
                    tool_name=self.name,
                )
            try:
                if operation == "sort":
                    result = sorted(data, key=lambda x: _coerce_comparable(x[target_field]), reverse=True)
                elif operation == "max":
                    result = max(data, key=lambda x: _coerce_comparable(x[target_field]))
                else:
                    result = min(data, key=lambda x: _coerce_comparable(x[target_field]))
            except (TypeError, ValueError) as exc:
                return enrich_tool_result(
                    {
                        "status": "no_valid_result",
                        "operation": operation,
                        "target_field": target_field,
                        "error": str(exc),
                        "result": "Aggregation could not compare the requested target_field values.",
                    },
                    tool_name=self.name,
                )
        
        elif operation == "merge_temporal":
            try:
                tolerance = float(kwargs.get("tolerance", 0.0) or 0.0)
            except (TypeError, ValueError):
                tolerance = -1.0
            if not math.isfinite(tolerance) or tolerance < 0:
                return enrich_tool_result(
                    {
                        "status": "no_valid_result",
                        "operation": operation,
                        "result": "Aggregation rejected tolerance; it must be a finite non-negative number.",
                    },
                    tool_name=self.name,
                )
            normalized = []
            ignored_count = 0
            for item in data:
                bounds = _temporal_bounds(item)
                if bounds is None:
                    ignored_count += 1
                    continue
                normalized.append({"start": bounds[0], "end": bounds[1], "source": item})
            normalized.sort(key=lambda item: (item["start"], item["end"]))
            merged = []
            for item in normalized:
                if not merged or item["start"] > merged[-1]["end"] + tolerance:
                    merged.append(
                        {
                            "start": item["start"],
                            "end": item["end"],
                            "source_segments": [item["source"]],
                        }
                    )
                else:
                    merged[-1]["end"] = max(merged[-1]["end"], item["end"])
                    merged[-1]["source_segments"].append(item["source"])
            result = {
                "status": "success" if normalized else "no_valid_result",
                "merged_segments": merged,
                "input_count": len(data),
                "ignored_count": ignored_count,
                "tolerance": tolerance,
            }

        else:
            return enrich_tool_result(
                {
                    "status": "no_valid_result",
                    "operation": operation,
                    "result": (
                        f"Aggregation does not support operation={operation!r}. "
                        "Use count, sort, max, min, or merge_temporal."
                    ),
                },
                tool_name=self.name,
            )
        
        logger.info(f"Aggregation ({operation}): {len(data)} items on field '{target_field}'")
        return result


class CountItems(BaseTool):
    """Count items in a collection."""
    
    name: str = "Count_Items"
    description: str = "Return the count of items in the given collection."
    tags: list = ["meta", "aggregation", "count", "metric"]
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "description": "Collection to count"
            }
        },
        "required": ["data"]
    }
    
    def run(self, data: List[Any], **kwargs) -> int:
        result = len(data)
        logger.info(f"Count_Items: counted {result} items")
        return result


class SortItems(BaseTool):
    """Sort items by a target field."""
    
    name: str = "Sort_Items"
    description: str = "Sort items by a numeric or comparable target field."
    tags: list = ["meta", "aggregation", "sort", "ranking"]
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "description": "Items to sort"
            },
            "target_field": {
                "type": "string",
                "description": "Field to sort by"
            },
            "order": {
                "type": "string",
                "enum": ["asc", "desc"],
                "description": "Sort order",
                "default": "desc"
            }
        },
        "required": ["data", "target_field"]
    }
    
    def run(self, data: List[Dict[str, Any]], target_field: str, order: str = "desc", **kwargs) -> List[Dict[str, Any]]:
        try:
            reverse = order != "asc"
            result = sorted(data, key=lambda x: x.get(target_field, 0), reverse=reverse)
        except Exception as e:
            logger.error(f"Sort_Items failed: {e}")
            result = data
        logger.info(f"Sort_Items: sorted {len(data)} items by '{target_field}' order={order}")
        return result


class MaxItem(BaseTool):
    """Find item with maximum target field."""
    
    name: str = "Max_Item"
    description: str = "Return the item with the maximum value on target_field."
    tags: list = ["meta", "aggregation", "max", "extreme"]
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "description": "Items to search"
            },
            "target_field": {
                "type": "string",
                "description": "Field to maximize"
            }
        },
        "required": ["data", "target_field"]
    }
    
    def run(self, data: List[Dict[str, Any]], target_field: str, **kwargs) -> Optional[Dict[str, Any]]:
        try:
            result = max(data, key=lambda x: x.get(target_field, float("-inf")))
        except ValueError:
            result = None
        logger.info(f"Max_Item: selected item by '{target_field}'")
        return result


class MinItem(BaseTool):
    """Find item with minimum target field."""
    
    name: str = "Min_Item"
    description: str = "Return the item with the minimum value on target_field."
    tags: list = ["meta", "aggregation", "min", "extreme"]
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "description": "Items to search"
            },
            "target_field": {
                "type": "string",
                "description": "Field to minimize"
            }
        },
        "required": ["data", "target_field"]
    }
    
    def run(self, data: List[Dict[str, Any]], target_field: str, **kwargs) -> Optional[Dict[str, Any]]:
        try:
            result = min(data, key=lambda x: x.get(target_field, float("inf")))
        except ValueError:
            result = None
        logger.info(f"Min_Item: selected item by '{target_field}'")
        return result


class MergeTemporalSegments(BaseTool):
    """Merge overlapping temporal segments."""
    
    name: str = "Merge_Temporal_Segments"
    description: str = "Merge overlapping or adjacent temporal segments into consolidated ranges."
    tags: list = ["meta", "aggregation", "temporal", "merge"]
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "description": "List of segments with start/end keys, e.g., [{'start': 0, 'end': 10}]"
            },
            "tolerance": {
                "type": "number",
                "description": "Merge tolerance in the same units as start/end",
                "default": 0
            }
        },
        "required": ["segments"]
    }
    
    def run(self, segments: List[Dict[str, Any]], tolerance: float = 0, **kwargs) -> Dict[str, Any]:
        if not segments:
            return {"merged_segments": [], "result": "No temporal segments were provided."}
        
        try:
            sorted_segments = sorted(segments, key=lambda s: s.get("start", 0))
            merged = []
            current = dict(sorted_segments[0])
            
            for seg in sorted_segments[1:]:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                if start <= current.get("end", 0) + tolerance:
                    current["end"] = max(current.get("end", 0), end)
                    current.setdefault("sources", []).extend(seg.get("sources", []))  # optional provenance merge
                else:
                    merged.append(current)
                    current = dict(seg)
            merged.append(current)
        except Exception as e:
            logger.error(f"Merge_Temporal_Segments failed: {e}")
            merged = segments
        
        logger.info(f"Merge_Temporal_Segments: merged {len(segments)} -> {len(merged)}")
        return enrich_tool_result(
            {
                "merged_segments": merged,
                "result": f"Merged {len(segments)} temporal segments into {len(merged)} consolidated segments.",
            },
            tool_name=self.name,
        )


class PythonExecutor(BaseTool):
    """Execute Python code for complex logic."""
    
    name: str = "Python_Executor"
    description: str = (
        "Execute generated Python code to handle complex logic. "
        "Used for complex mathematical calculations, text-first aggregation, "
        "or processing structured evidence with custom logic after evidence has already been extracted. "
        "It must not infer image/video content from file paths alone."
    )
    tags: list = ["base", "execution", "python", "custom_logic"]
    
    @property
    def model_path(self) -> str:
        return settings.python_coder_model_path
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "code_script": {
                "type": "string",
                "description": "Python code to execute"
            },
            "context_vars": {
                "type": "object",
                "description": "Variables to make available in execution context",
                "default": {}
            }
        },
        "required": ["code_script"]
    }
    
    def run(self, code_script: str, context_vars: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """Execute Python code.
        
        Args:
            code_script: Code to execute
            context_vars: Variables for execution context
            **kwargs: Additional arguments
            
        Returns:
            Execution result
        """
        context_vars = context_vars or {}
        
        # Create safe execution environment
        local_vars = dict(context_vars)
        global_vars = {
            "__builtins__": __builtins__,
            "collections": collections,
            "Counter": collections.Counter,
            "defaultdict": collections.defaultdict,
            "itertools": itertools,
            "json": json,
            "math": math,
            "re": re,
            "sum": sum,
            "len": len,
            "max": max,
            "min": min,
            "abs": abs,
            "round": round,
            "statistics": statistics,
        }
        
        try:
            lowered_code = str(code_script or "").lower()
            if any(
                marker in lowered_code
                for marker in (
                    "import seaborn",
                    "from seaborn",
                    "import matplotlib",
                    "from matplotlib",
                    "plt.",
                    "sns.",
                )
            ):
                error_msg = (
                    "Execution error: plotting libraries are unavailable in Python_Executor. "
                    "Rewrite the code to return numeric/text summaries only; do not import seaborn/matplotlib."
                )
                logger.error(f"Python_Executor: {error_msg}")
                return {"error": error_msg}

            try:
                compile(str(code_script or ""), "<python_executor>", "exec")
            except SyntaxError as exc:
                detail = str(exc.msg or exc).strip()
                if "invalid decimal literal" in detail.lower():
                    detail = (
                        "percent-style text such as `12%` is not valid Python syntax; "
                        "convert it to a decimal number or quoted string first"
                    )
                error_msg = (
                    "Execution error: Python_Executor requires runnable Python syntax. "
                    f"Rewrite the code_script so it is valid Python only ({detail})."
                )
                logger.error(f"Python_Executor: {error_msg}")
                return {"error": error_msg}

            stdout_buffer = io.StringIO()
            with redirect_stdout(stdout_buffer):
                exec(code_script, global_vars, local_vars)

            # Priority: explicit result variable > printed stdout > success sentinel
            if "result" in local_vars:
                result = local_vars.get("result")
            else:
                captured_stdout = stdout_buffer.getvalue().strip()
                if captured_stdout:
                    result = captured_stdout
                else:
                    result = "Code executed successfully"

            logger.info("Python_Executor: Executed code successfully")
            return result
            
        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            logger.error(f"Python_Executor: {error_msg}")
            return {"error": error_msg}


class SQLExecutor(BaseTool):
    """Execute generated SQL for structured/tabular queries."""
    
    name: str = "SQL_Executor"
    description: str = (
        "Execute generated SQL for complex structured queries. "
        "Supports running ad-hoc analytics against in-memory tables provided at call time. "
        "When duckdb/pandas is unavailable, returns an informative error."
    )
    tags: list = ["base", "sql", "analytics", "table"]
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "sql_query": {
                "type": "string",
                "description": "SQL to execute"
            },
            "tables": {
                "type": "object",
                "description": "Mapping of table_name -> list[dict] rows to register in-memory"
            },
            "limit": {
                "type": "integer",
                "description": "Optional row limit applied after execution",
                "default": None
            }
        },
        "required": ["sql_query"]
    }
    
    def run(self, sql_query: str, tables: Optional[Dict[str, List[Dict[str, Any]]]] = None, limit: Optional[int] = None, **kwargs) -> Any:
        tables = tables or {}
        try:
            import duckdb  # type: ignore
            import pandas as pd  # type: ignore
        except Exception as e:
            msg = f"SQL execution unavailable (missing duckdb/pandas): {e}"
            logger.error(msg)
            return {"error": msg}
        
        try:
            con = duckdb.connect(database=":memory:")
            for name, rows in tables.items():
                df = pd.DataFrame(rows)
                con.register(name, df)
            query = sql_query if limit is None else f"SELECT * FROM ({sql_query}) t LIMIT {int(limit)}"
            result_df = con.execute(query).fetchdf()
            con.close()
            return result_df.to_dict(orient="records")
        except Exception as e:
            msg = f"SQL execution failed: {e}"
            logger.error(msg)
            return {"error": msg}


class DebtInterestAnalyzer(BaseTool):
    """Compare debt interest under minimum and accelerated repayments."""

    name: str = "Debt_Interest_Analyzer"
    description: str = (
        "Compute amortization for a debt and compare total interest between "
        "minimum payment and accelerated repayment scenarios."
    )
    tags: list = ["base", "finance", "loan"]
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "principal": {"type": "number", "description": "Initial debt amount"},
            "annual_rate": {"type": "number", "description": "Annual interest rate (e.g., 0.13 for 13%)"},
            "min_rate": {
                "type": "number",
                "description": "Minimum payment as a fraction of current balance (e.g., 0.02 for 2%)",
                "default": 0.02,
            },
            "min_payment_floor": {
                "type": "number",
                "description": "Absolute floor for minimum payment per period",
                "default": 25.0,
            },
            "accelerated_extra": {
                "type": "number",
                "description": "Extra amount added on top of minimum payment for accelerated scenario",
                "default": 0.0,
            },
            "max_periods": {
                "type": "integer",
                "description": "Safety cap on number of periods to avoid infinite loops",
                "default": 600,
            },
        },
        "required": ["principal", "annual_rate"],
    }

    def _simulate(
        self,
        principal: float,
        annual_rate: float,
        min_rate: float,
        min_payment_floor: float,
        extra: float,
        max_periods: int,
    ) -> Dict[str, Any]:
        balance = float(principal)
        rate_per_period = float(annual_rate) / 12.0
        min_rate = float(min_rate)
        min_payment_floor = float(min_payment_floor)
        extra = float(extra)
        max_periods = int(max_periods)

        total_interest = 0.0
        periods = 0
        first_payment = None
        last_payment = None

        for _ in range(max_periods):
            if balance <= 1e-8:
                break
            interest = balance * rate_per_period
            base_payment = max(balance * min_rate, min_payment_floor)
            payment = base_payment + extra
            if payment <= interest:
                raise ValueError("Payment does not cover interest; increase minimums or extra.")
            payment = min(payment, balance + interest)
            principal_paid = payment - interest
            balance = max(0.0, balance - principal_paid)
            total_interest += interest
            periods += 1
            
            # Only keep first and last payment for summary
            if periods == 1:
                first_payment = {
                    "period": periods,
                    "payment": round(payment, 2),
                    "interest": round(interest, 2),
                    "principal": round(principal_paid, 2),
                    "balance": round(balance, 2),
                }
            last_payment = {
                "period": periods,
                "payment": round(payment, 2),
                "interest": round(interest, 2),
                "principal": round(principal_paid, 2),
                "balance": round(balance, 2),
            }
        else:
            raise ValueError("Max periods exceeded; payments too small to amortize the debt.")

        return {
            "periods": periods,
            "total_interest": round(total_interest, 2),
            "total_paid": round(principal + total_interest, 2),
            "first_payment": first_payment,
            "last_payment": last_payment,
        }

    def run(
        self,
        principal: float,
        annual_rate: float,
        min_rate: float = 0.02,
        min_payment_floor: float = 25.0,
        accelerated_extra: float = 0.0,
        max_periods: int = 600,
        **kwargs,
    ) -> Dict[str, Any]:
        candidate_periods = []
        for candidate in (int(max_periods), 240, 600, 1200):
            if candidate > 0 and candidate not in candidate_periods:
                candidate_periods.append(candidate)

        last_error: Optional[Exception] = None
        baseline: Optional[Dict[str, Any]] = None
        accelerated: Optional[Dict[str, Any]] = None
        used_max_periods = candidate_periods[0]

        for period_cap in candidate_periods:
            try:
                baseline = self._simulate(principal, annual_rate, min_rate, min_payment_floor, 0.0, period_cap)
                accelerated = self._simulate(
                    principal, annual_rate, min_rate, min_payment_floor, accelerated_extra, period_cap
                )
                used_max_periods = period_cap
                break
            except ValueError as exc:
                if "Max periods exceeded" in str(exc):
                    last_error = exc
                    continue
                raise

        if baseline is None or accelerated is None:
            raise ValueError(
                f"Max periods exceeded; payments too small to amortize the debt within cap={candidate_periods[-1]}."
            ) from last_error

        return {
            "minimum_payment": baseline,
            "accelerated_payment": accelerated,
            "interest_saved": round(baseline["total_interest"] - accelerated["total_interest"], 2),
            "periods_saved": baseline["periods"] - accelerated["periods"],
            "used_max_periods": used_max_periods,
            "result": (
                f"Minimum-payment interest is {baseline['total_interest']}, "
                f"accelerated-payment interest is {accelerated['total_interest']}, "
                f"saving {round(baseline['total_interest'] - accelerated['total_interest'], 2)} in interest."
            ),
        }


# ToolName catch-all removed: the SYSTEM_PROMPT format example was changed to use a
# concrete tool name, so 'ToolName' can no longer appear as a legitimate call.
# It is now blocked in engine.py's invalid-action guard ('toolname' in the reject list).
