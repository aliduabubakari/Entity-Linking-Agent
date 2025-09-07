from typing import Dict, Any
from domain.entities import TableColumn, ColumnType
import logging

logger = logging.getLogger(__name__)

async def infer_column_type(column: TableColumn, llm) -> str:
    """Infer column type using LLM"""
    try:
        system_prompt = """You are an expert data analyst. Analyze the column values and determine the most likely semantic type.
        Return ONLY the type name from this list: PERSON, ORGANIZATION, LOCATION, EVENT, WORK, DATE, NUMERIC, LITERAL, MIXED, UNKNOWN"""
        
        sample_values = column.values[:10] if len(column.values) > 10 else column.values
        human_prompt = f"""Analyze this column named '{column.name}' with values: {sample_values}
        What type of entities does this column contain? Return only the type name."""
        
        response = await llm.generate(system_prompt, human_prompt)
        return response.strip().upper()
        
    except Exception as e:
        logger.error(f"Column type inference failed: {e}")
        return "UNKNOWN"

async def infer_header(column: TableColumn, llm) -> str:
    """Infer column header using LLM"""
    try:
        system_prompt = """You are an expert data analyst. Infer a meaningful header name for a column based on its values."""
        
        sample_values = column.values[:10] if len(column.values) > 10 else column.values
        human_prompt = f"""Column values: {sample_values}
        Suggest an appropriate header name. Return only the name."""
        
        response = await llm.generate(system_prompt, human_prompt)
        return response.strip()
        
    except Exception as e:
        logger.error(f"Header inference failed: {e}")
        return "unknown_column"

async def extract_table_context(target_column: TableColumn, other_columns: list = None) -> Dict[str, Any]:
    """Extract table context"""
    try:
        context = {
            "target_column_name": target_column.name,
            "other_columns": [col.name for col in (other_columns or [])],
            "sample_data": {}
        }
        
        for col in (other_columns or [])[:3]:
            context["sample_data"][col.name] = col.values[:5]
        
        return context
        
    except Exception as e:
        logger.error(f"Context extraction failed: {e}")
        return {}