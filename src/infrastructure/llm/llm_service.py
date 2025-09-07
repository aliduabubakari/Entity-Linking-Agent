from typing import List, Dict, Optional, Any
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from domain.interfaces import DisambiguationService, ValidationService, ColumnAnalysisService
from domain.entities import EntityCandidate, TableColumn, LinkingResult, ColumnType
from domain.value_objects import ValidationResult, ConfidenceScoreBreakdown
from config.settings import settings
import logging
import json
import asyncio

logger = logging.getLogger(__name__)

class AzureOpenAILanguageModel:
    """Wrapper for Azure OpenAI LLM interactions"""
    
    def __init__(self, model: str = None, temperature: float = 0.1):
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_deployment=model or settings.LLM_MODEL,
            temperature=temperature
        )
    
    async def generate(self, system_prompt: str, human_prompt: str) -> str:
        """Generate response from LLM"""
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].text
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

class LLMColumnAnalysisService(ColumnAnalysisService):
    """LLM-based implementation of column analysis"""
    
    def __init__(self, llm: AzureOpenAILanguageModel):
        self.llm = llm
    
    async def analyze_column_type(self, column: TableColumn) -> ColumnType:
        system_prompt = """You are an expert data analyst. Analyze the column values and determine the most likely semantic type.
        Return ONLY the type name from this list: PERSON, ORGANIZATION, LOCATION, EVENT, WORK, DATE, NUMERIC, LITERAL, MIXED, UNKNOWN"""
        
        sample_values = column.values[:10] if len(column.values) > 10 else column.values
        human_prompt = f"""Analyze this column named '{column.name}' with values: {sample_values}
        What type of entities does this column contain? Return only the type name."""
        
        response = await self.llm.generate(system_prompt, human_prompt)
        try:
            return ColumnType(response.strip().upper())
        except ValueError:
            return ColumnType.UNKNOWN
    
    async def infer_header(self, column: TableColumn, other_columns: List[TableColumn] = None) -> str:
        system_prompt = """You are an expert data analyst. Infer a meaningful header name for a column based on its values."""
        
        sample_values = column.values[:10] if len(column.values) > 10 else column.values
        human_prompt = f"""Column values: {sample_values}
        Suggest an appropriate header name. Return only the name."""
        
        response = await self.llm.generate(system_prompt, human_prompt)
        return response.strip()
    
    async def extract_table_context(self, target_column: TableColumn, other_columns: List[TableColumn]) -> Dict[str, Any]:
        context = {
            "target_column_name": target_column.name,
            "other_columns": [col.name for col in other_columns],
            "sample_data": {}
        }
        
        for col in other_columns[:3]:
            context["sample_data"][col.name] = col.values[:5]
        
        return context

class LLMDisambiguationService(DisambiguationService):
    """LLM-based disambiguation service"""
    
    def __init__(self, llm: AzureOpenAILanguageModel):
        self.llm = llm
    
    async def disambiguate(
        self,
        candidates: List[EntityCandidate],
        mention: str,
        column: TableColumn,
        table_context: Dict[str, Any]
    ) -> List[EntityCandidate]:
        if len(candidates) <= 1:
            return candidates
        
        system_prompt = """You are an expert at entity disambiguation. Given a mention and its candidates, 
        rank them by relevance and assign confidence scores. Consider context carefully."""
        
        candidates_text = "\n".join([
            f"ID: {c.id}, Name: {c.name}, Description: {c.description}, Types: {[t.name for t in c.types]}"
            for c in candidates[:5]  # Limit to top 5 to avoid token limits
        ])
        
        human_prompt = f"""
        Mention: "{mention}"
        Column Type: {column.type}
        Column Name: {column.name}
        Context: {table_context}
        
        Candidates:
        {candidates_text}
        
        Rank these candidates by relevance (1 = most relevant) and provide confidence scores (0-1).
        Return as JSON: {{"rankings": [{{"id": "...", "rank": 1, "confidence": 0.9, "reasoning": "..."}}, ...]}}
        """
        
        try:
            response = await self.llm.generate(system_prompt, human_prompt)
            rankings = json.loads(response)
            
            # Update candidate confidences based on LLM rankings
            id_to_ranking = {r["id"]: r for r in rankings.get("rankings", [])}
            
            for candidate in candidates:
                if candidate.id in id_to_ranking:
                    ranking_info = id_to_ranking[candidate.id]
                    candidate.confidence = ranking_info.get("confidence", 0.5)
                    candidate.metadata["llm_reasoning"] = ranking_info.get("reasoning", "")
            
            # Sort by confidence (descending)
            return sorted(candidates, key=lambda c: c.confidence, reverse=True)
            
        except Exception as e:
            logger.error(f"LLM disambiguation failed: {e}")
            # Fallback: return candidates sorted by existing scores
            return sorted(candidates, key=lambda c: c.ed_score or 0, reverse=True)
    
    async def calculate_confidence_breakdown(
        self,
        candidate: EntityCandidate,
        context: Dict[str, Any]
    ) -> ConfidenceScoreBreakdown:
        # Simplified implementation
        return ConfidenceScoreBreakdown(
            semantic_similarity=candidate.ed_score or 0.5,
            type_compatibility=0.8 if candidate.types else 0.3,
            popularity=candidate.popularity or 0.5,
            context_relevance=0.7,
            llm_confidence=candidate.confidence,
            total=candidate.confidence
        )

class LLMValidationService(ValidationService):
    """LLM-based validation service"""
    
    def __init__(self, llm: AzureOpenAILanguageModel):
        self.llm = llm
    
    async def validate_linking_result(
        self,
        result: LinkingResult,
        context: Dict[str, Any]
    ) -> ValidationResult:
        if not result.selected_candidate:
            return ValidationResult(
                is_valid=False,
                score=0.0,
                reasons=["No candidate selected"],
                suggestions=["Consider relaxing matching criteria"]
            )
        
        # Simple validation based on confidence
        is_valid = result.confidence > 0.6
        return ValidationResult(
            is_valid=is_valid,
            score=result.confidence,
            reasons=["High confidence match" if is_valid else "Low confidence match"],
            suggestions=[] if is_valid else ["Consider manual verification"]
        )
    
    async def validate_llm_reasoning(
        self,
        reasoning: str,
        evidence: Dict[str, Any]
    ) -> ValidationResult:
        # Simplified validation
        return ValidationResult(
            is_valid=bool(reasoning and len(reasoning) > 10),
            score=0.8 if reasoning else 0.2,
            reasons=["Valid reasoning provided" if reasoning else "No reasoning provided"]
        )