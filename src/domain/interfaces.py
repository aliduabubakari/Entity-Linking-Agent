from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from domain.entities import EntityCandidate, TableColumn, KnowledgeBaseConfig, LinkingResult
from domain.value_objects import ValidationResult, ConfidenceScoreBreakdown

class KnowledgeBaseGateway(ABC):
    """Abstract interface for knowledge base integrations"""
    
    @abstractmethod
    async def get_candidates(self, mention: str, context: Optional[Dict] = None) -> List[EntityCandidate]:
        pass
    
    @abstractmethod
    def get_config(self) -> KnowledgeBaseConfig:
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        pass

class DisambiguationService(ABC):
    """Abstract interface for disambiguation services"""
    
    @abstractmethod
    async def disambiguate(
        self,
        candidates: List[EntityCandidate],
        mention: str,
        column: TableColumn,
        table_context: Dict[str, Any]
    ) -> List[EntityCandidate]:
        pass
    
    @abstractmethod
    async def calculate_confidence_breakdown(
        self,
        candidate: EntityCandidate,
        context: Dict[str, Any]
    ) -> ConfidenceScoreBreakdown:
        pass

class ValidationService(ABC):
    """Abstract interface for validation services"""
    
    @abstractmethod
    async def validate_linking_result(
        self,
        result: LinkingResult,
        context: Dict[str, Any]
    ) -> ValidationResult:
        pass
    
    @abstractmethod
    async def validate_llm_reasoning(
        self,
        reasoning: str,
        evidence: Dict[str, Any]
    ) -> ValidationResult:
        pass

class ColumnAnalysisService(ABC):
    """Abstract interface for column analysis"""
    
    @abstractmethod
    async def analyze_column_type(self, column: TableColumn) -> str:
        pass
    
    @abstractmethod
    async def infer_header(self, column: TableColumn, other_columns: List[TableColumn] = None) -> str:
        pass
    
    @abstractmethod
    async def extract_table_context(
        self,
        target_column: TableColumn,
        other_columns: List[TableColumn]
    ) -> Dict[str, Any]:
        pass

class CacheRepository(ABC):
    """Abstract interface for caching"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass