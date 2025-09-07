from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from datetime import datetime

class ColumnType(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    WORK = "WORK"  # Books, movies, artworks, etc.
    DATE = "DATE"
    NUMERIC = "NUMERIC"
    LITERAL = "LITERAL"  # Other text values
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"

class EntityType(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    source: Optional[str] = None

class EntityCandidate(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    types: List[EntityType] = Field(default_factory=list)
    ed_score: Optional[float] = None  # Edit distance score
    popularity: Optional[float] = None
    source_kb: str
    kb_specific_data: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('confidence')
    def validate_confidence(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v

class TableColumn(BaseModel):
    name: str
    values: List[str]
    type: ColumnType = ColumnType.UNKNOWN
    index: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('values')
    def validate_values(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError('Column values cannot be empty')
        return v

class KnowledgeBaseConfig(BaseModel):
    name: str
    url: str
    credentials: Dict[str, Any] = Field(default_factory=dict)
    type: str  # "lamapi", "geonames", "sparql", "alligator", etc.
    supported_column_types: List[ColumnType] = Field(default_factory=list)
    enabled: bool = True
    priority: int = 99  # Lower number = higher priority
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class LinkingResult(BaseModel):
    mention: str
    selected_candidate: Optional[EntityCandidate] = None
    all_candidates: List[EntityCandidate] = Field(default_factory=list)
    confidence: float = 0.0
    is_ambiguous: bool = False
    processing_time: float = 0.0
    used_knowledge_bases: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ColumnLinkingResult(BaseModel):
    column_name: str
    column_type: ColumnType
    results: List[LinkingResult] = Field(default_factory=list)
    success_rate: float = 0.0
    average_confidence: float = 0.0
    start_time: datetime
    end_time: Optional[datetime] = None
    total_processing_time: float = 0.0

    @field_validator('success_rate', 'average_confidence')
    def validate_percentages(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError('Must be between 0 and 1')
        return v