from typing import List, Dict, Optional
from dataclasses import dataclass
from domain.entities import TableColumn, LinkingResult, ColumnLinkingResult
from domain.value_objects import ProcessingStatus, ProcessingMetrics
from domain.interfaces import (
    KnowledgeBaseGateway, 
    DisambiguationService, 
    ValidationService,
    ColumnAnalysisService
)
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class EntityLinkingUseCase:
    """Orchestrates the entity linking process for a table column"""
    
    knowledge_base_gateways: List[KnowledgeBaseGateway]
    disambiguation_service: DisambiguationService
    validation_service: ValidationService
    column_analysis_service: ColumnAnalysisService
    batch_size: int = 50
    
    async def execute(self, column: TableColumn, table_context: Dict = None) -> ColumnLinkingResult:
        """Main entry point for entity linking"""
        start_time = datetime.now()
        
        try:
            # Step 1: Analyze column if type is unknown
            if column.type.value == "UNKNOWN":
                column.type = await self.column_analysis_service.analyze_column_type(column)
                logger.info(f"Inferred column type: {column.type}")
            
            # Step 2: Process column in batches
            results = await self._process_column_batches(column, table_context or {})
            
            # Step 3: Calculate overall metrics
            metrics = self._calculate_metrics(results)
            
            return ColumnLinkingResult(
                column_name=column.name,
                column_type=column.type,
                results=results,
                success_rate=metrics.successful_links / metrics.total_mentions if metrics.total_mentions > 0 else 0,
                average_confidence=metrics.average_confidence,
                start_time=start_time,
                end_time=datetime.now(),
                total_processing_time=(datetime.now() - start_time).total_seconds()
            )
            
        except Exception as e:
            logger.error(f"Entity linking failed: {e}")
            return ColumnLinkingResult(
                column_name=column.name,
                column_type=column.type,
                results=[],
                success_rate=0.0,
                average_confidence=0.0,
                start_time=start_time,
                end_time=datetime.now(),
                total_processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def _process_column_batches(self, column: TableColumn, table_context: Dict) -> List[LinkingResult]:
        """Process column values in batches"""
        results = []
        
        for i in range(0, len(column.values), self.batch_size):
            batch = column.values[i:i + self.batch_size]
            batch_results = await asyncio.gather(
                *[self._process_single_mention(mention, column, table_context) for mention in batch]
            )
            results.extend(batch_results)
            
        return results
    
    async def _process_single_mention(self, mention: str, column: TableColumn, table_context: Dict) -> LinkingResult:
        """Process a single cell value through the entire pipeline"""
        start_time = datetime.now()
        used_kbs = []
        all_candidates = []
        
        try:
            # Step 1: Retrieve candidates from knowledge bases
            for kb_gateway in self.knowledge_base_gateways:
                if column.type in kb_gateway.get_config().supported_column_types:
                    candidates = await kb_gateway.get_candidates(mention, table_context)
                    all_candidates.extend(candidates)
                    used_kbs.append(kb_gateway.get_config().name)
            
            if not all_candidates:
                return LinkingResult(
                    mention=mention,
                    all_candidates=[],
                    confidence=0.0,
                    is_ambiguous=False,
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    used_knowledge_bases=used_kbs,
                    metadata={"error": "No candidates found"}
                )
            
            # Step 2: Disambiguate candidates
            disambiguated_candidates = await self.disambiguation_service.disambiguate(
                all_candidates, mention, column, table_context
            )
            
            # Step 3: Select best candidate
            selected_candidate = disambiguated_candidates[0] if disambiguated_candidates else None
            
            # Step 4: Validate result
            validation_result = await self.validation_service.validate_linking_result(
                LinkingResult(
                    mention=mention,
                    selected_candidate=selected_candidate,
                    all_candidates=disambiguated_candidates,
                    confidence=selected_candidate.confidence if selected_candidate else 0.0,
                    is_ambiguous=len(disambiguated_candidates) > 1 and 
                               selected_candidate and 
                               selected_candidate.confidence < 0.8,
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    used_knowledge_bases=used_kbs
                ),
                {"column": column.dict(), "table_context": table_context}
            )
            
            result = LinkingResult(
                mention=mention,
                selected_candidate=selected_candidate,
                all_candidates=disambiguated_candidates,
                confidence=selected_candidate.confidence if selected_candidate else 0.0,
                is_ambiguous=len(disambiguated_candidates) > 1 and 
                           selected_candidate and 
                           selected_candidate.confidence < 0.8,
                processing_time=(datetime.now() - start_time).total_seconds(),
                used_knowledge_bases=used_kbs,
                metadata={"validation": validation_result.dict()}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process mention '{mention}': {e}")
            return LinkingResult(
                mention=mention,
                all_candidates=[],
                confidence=0.0,
                is_ambiguous=False,
                processing_time=(datetime.now() - start_time).total_seconds(),
                used_knowledge_bases=used_kbs,
                metadata={"error": str(e)}
            )
    
    def _calculate_metrics(self, results: List[LinkingResult]) -> ProcessingMetrics:
        """Calculate processing metrics from results"""
        successful_links = sum(1 for r in results if r.selected_candidate and r.confidence > 0.6)
        total_confidences = sum(r.confidence for r in results if r.selected_candidate)
        total_with_candidates = sum(1 for r in results if r.selected_candidate)
        
        kb_usage = {}
        for result in results:
            for kb in result.used_knowledge_bases:
                kb_usage[kb] = kb_usage.get(kb, 0) + 1
        
        return ProcessingMetrics(
            total_mentions=len(results),
            processed_mentions=len(results),
            successful_links=successful_links,
            failed_links=len(results) - successful_links,
            average_confidence=total_confidences / total_with_candidates if total_with_candidates > 0 else 0,
            kb_usage=kb_usage
        )