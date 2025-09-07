from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class AgentState(BaseModel):
    """State object for LangGraph agent workflow"""
    column_name: str
    column_values: List[str]
    column_type: Optional[str] = None
    table_context: Dict[str, Any] = Field(default_factory=dict)
    candidates: Dict[str, List[Dict]] = Field(default_factory=dict)  # mention -> list of candidates
    disambiguated_results: Dict[str, Dict] = Field(default_factory=dict)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    validation_results: Dict[str, Dict] = Field(default_factory=dict)
    current_mention: Optional[str] = None
    processing_errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EntityLinkingRequest(BaseModel):
    column_name: str
    column_values: List[str]
    table_context: Optional[Dict[str, Any]] = None
    knowledge_bases: Optional[List[str]] = None  # Specific KBs to use
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)

class EntityLinkingResponse(BaseModel):
    request_id: str
    status: str  # processing, completed, failed
    results: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class AgentToolRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]
    agent_id: str

class AgentToolResponse(BaseModel):
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float

class KnowledgeBaseStatus(BaseModel):
    name: str
    type: str
    enabled: bool
    priority: int
    health_status: bool
    last_checked: datetime

class SystemHealth(BaseModel):
    status: str
    knowledge_bases: List[KnowledgeBaseStatus]
    llm_available: bool
    cache_enabled: bool
    total_processed: int
    uptime: float

# Add to existing SystemHealth class
class SystemHealth(BaseModel):
    status: str
    knowledge_bases: List[KnowledgeBaseStatus]
    llm_available: bool
    cache_enabled: bool
    total_processed: int
    uptime: float
    agent_performance: Optional[Dict[str, Any]] = None  # Add this line

# Add to existing EntityLinkingResponse class  
class EntityLinkingResponse(BaseModel):
    request_id: str
    status: str
    current_phase: Optional[str] = None  # Add this line
    results: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    execution_timeline: Optional[List[Dict[str, Any]]] = None  # Add this line