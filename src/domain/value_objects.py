from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class ProcessingStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"

class ValidationResult(BaseModel):
    is_valid: bool
    score: float = 0.0
    reasons: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ConfidenceScoreBreakdown(BaseModel):
    semantic_similarity: float = 0.0
    type_compatibility: float = 0.0
    popularity: float = 0.0
    context_relevance: float = 0.0
    llm_confidence: float = 0.0
    total: float = 0.0

class TableContext(BaseModel):
    headers: List[str] = Field(default_factory=list)
    sample_rows: List[Dict[str, Any]] = Field(default_factory=list)
    inferred_semantics: Dict[str, Any] = Field(default_factory=dict)
    relationships: List[Dict[str, Any]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ProcessingMetrics(BaseModel):
    total_mentions: int = 0
    processed_mentions: int = 0
    successful_links: int = 0
    failed_links: int = 0
    average_confidence: float = 0.0
    total_processing_time: float = 0.0
    kb_usage: Dict[str, int] = Field(default_factory=dict)