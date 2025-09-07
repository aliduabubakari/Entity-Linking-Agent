from typing import Dict, Any, List
from datetime import datetime
from langgraph.graph import StateGraph, END  # Add END import
from interface.schemas import AgentState
from domain.entities import EntityCandidate
from infrastructure.knowledge_bases import KnowledgeBaseFactory
from config.settings import knowledge_bases_config
from domain.entities import KnowledgeBaseConfig
import asyncio
import logging

logger = logging.getLogger(__name__)

class CandidateRetrieverAgent:
    """Agent responsible for retrieving candidates from knowledge bases"""
    
    def __init__(self):
        self.knowledge_bases = self._initialize_knowledge_bases()
        self.graph = self._build_graph()
    
    def _initialize_knowledge_bases(self) -> List:
        """Initialize knowledge base gateways"""
        kb_configs = [KnowledgeBaseConfig(**kb) for kb in knowledge_bases_config["knowledge_bases"]]
        return [KnowledgeBaseFactory.create_gateway(config) for config in kb_configs if config.enabled]
    
    def _build_graph(self):
        """Build the LangGraph for candidate retrieval"""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("select_knowledge_bases", self.select_knowledge_bases)
        workflow.add_node("retrieve_candidates", self.retrieve_candidates)
        workflow.add_node("filter_candidates", self.filter_candidates)
        
        workflow.set_entry_point("select_knowledge_bases")
        workflow.add_edge("select_knowledge_bases", "retrieve_candidates")
        workflow.add_edge("retrieve_candidates", "filter_candidates")
        workflow.add_edge("filter_candidates", END)  # Now END is imported
        
        return workflow.compile()
    
    async def select_knowledge_bases(self, state: AgentState) -> Dict[str, Any]:
        """Select appropriate knowledge bases based on column type"""
        try:
            suitable_kbs = [
                kb for kb in self.knowledge_bases
                if state.column_type in [ct.value for ct in kb.get_config().supported_column_types]
            ]
            
            return {
                "metadata": {
                    **state.metadata,
                    "selected_knowledge_bases": [kb.get_config().name for kb in suitable_kbs],
                    "selection_criteria": f"column_type_{state.column_type}"
                }
            }
            
        except Exception as e:
            logger.error(f"KB selection failed: {e}")
            return {
                "processing_errors": [f"KB selection error: {str(e)}"],
                "metadata": {**state.metadata, "kb_selection_failed": True}
            }
    
    async def retrieve_candidates(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve candidates for all mentions"""
        try:
            candidates = {}
            
            for mention in state.column_values:
                mention_candidates = []
                
                # Query all suitable knowledge bases for this mention
                for kb in self.knowledge_bases:
                    if state.column_type in [ct.value for ct in kb.get_config().supported_column_types]:
                        kb_candidates = await kb.get_candidates(mention, state.table_context)
                        mention_candidates.extend([cand.model_dump() for cand in kb_candidates])  # Use model_dump() instead of dict()
                
                candidates[mention] = mention_candidates
            
            return {
                "candidates": candidates,
                "metadata": {
                    **state.metadata,
                    "total_candidates": sum(len(cands) for cands in candidates.values()),
                    "retrieval_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Candidate retrieval failed: {e}")
            return {
                "candidates": {},
                "processing_errors": [f"Candidate retrieval error: {str(e)}"]
            }
    
    async def filter_candidates(self, state: AgentState) -> Dict[str, Any]:
        """Filter and pre-process candidates"""
        try:
            filtered_candidates = {}
            
            for mention, cands in state.candidates.items():
                # Basic filtering: remove duplicates and low-confidence candidates
                unique_candidates = self._remove_duplicates(cands)
                filtered = [cand for cand in unique_candidates if cand.get('ed_score', 0) > 0.1]  # Lower threshold
                filtered_candidates[mention] = filtered
            
            return {
                "candidates": filtered_candidates,
                "metadata": {
                    **state.metadata,
                    "filtering_applied": True,
                    "pre_filter_count": sum(len(cands) for cands in state.candidates.items()),
                    "post_filter_count": sum(len(cands) for cands in filtered_candidates.values())
                }
            }
            
        except Exception as e:
            logger.error(f"Candidate filtering failed: {e}")
            return {
                "candidates": state.candidates,
                "processing_errors": [f"Candidate filtering error: {str(e)}"]
            }
    
    def _remove_duplicates(self, candidates: List[Dict]) -> List[Dict]:
        """Remove duplicate candidates based on ID"""
        seen_ids = set()
        unique_candidates = []
        
        for cand in candidates:
            if cand['id'] not in seen_ids:
                seen_ids.add(cand['id'])
                unique_candidates.append(cand)
        
        return unique_candidates
    
    async def execute(self, state: AgentState) -> AgentState:
        """Execute the candidate retrieval workflow"""
        return await self.graph.ainvoke(state)