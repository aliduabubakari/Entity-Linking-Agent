from typing import Dict, Any, List
from datetime import datetime
from langgraph.graph import StateGraph, END  # Add END import
from interface.schemas import AgentState
import logging

logger = logging.getLogger(__name__)

class DisambiguationAgent:
    """Agent responsible for disambiguating between candidates"""
    
    def __init__(self, llm_service):
        self.llm = llm_service
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph for disambiguation"""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("calculate_similarity", self.calculate_similarity)
        workflow.add_node("check_type_compatibility", self.check_type_compatibility)
        workflow.add_node("contextual_disambiguation", self.contextual_disambiguation)
        workflow.add_node("rank_candidates", self.rank_candidates)
        
        workflow.set_entry_point("calculate_similarity")
        workflow.add_edge("calculate_similarity", "check_type_compatibility")
        workflow.add_edge("check_type_compatibility", "contextual_disambiguation")
        workflow.add_edge("contextual_disambiguation", "rank_candidates")
        workflow.add_edge("rank_candidates", END)  # Now END is imported
        
        return workflow.compile()
    
    async def calculate_similarity(self, state: AgentState) -> Dict[str, Any]:
        """Calculate semantic similarity for candidates"""
        try:
            similarity_scores = {}
            
            for mention, candidates in state.candidates.items():
                mention_scores = {}
                for cand in candidates:
                    # Simple similarity calculation based on name matching
                    name_similarity = self._calculate_name_similarity(mention, cand.get('name', ''))
                    mention_scores[cand['id']] = name_similarity
                similarity_scores[mention] = mention_scores
            
            return {
                "metadata": {
                    **state.metadata,
                    "similarity_scores": similarity_scores,
                    "similarity_calculated": True
                }
            }
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return {
                "processing_errors": [f"Similarity calculation error: {str(e)}"]
            }
    
    def _calculate_name_similarity(self, mention: str, candidate_name: str) -> float:
        """Simple name similarity calculation"""
        mention_lower = mention.lower().strip()
        candidate_lower = candidate_name.lower().strip()
        
        if mention_lower == candidate_lower:
            return 1.0
        elif mention_lower in candidate_lower or candidate_lower in mention_lower:
            return 0.8
        else:
            # Simple character overlap
            common_chars = set(mention_lower) & set(candidate_lower)
            return len(common_chars) / max(len(set(mention_lower)), len(set(candidate_lower)), 1)
    
    async def check_type_compatibility(self, state: AgentState) -> Dict[str, Any]:
        """Check type compatibility for candidates"""
        try:
            type_scores = {}
            
            for mention, candidates in state.candidates.items():
                mention_scores = {}
                for cand in candidates:
                    # Simple type compatibility score
                    score = 0.8  # Default compatibility score
                    mention_scores[cand['id']] = score
                type_scores[mention] = mention_scores
            
            return {
                "metadata": {
                    **state.metadata,
                    "type_scores": type_scores,
                    "type_checked": True
                }
            }
            
        except Exception as e:
            logger.error(f"Type compatibility check failed: {e}")
            return {
                "processing_errors": [f"Type compatibility error: {str(e)}"]
            }
    
    async def contextual_disambiguation(self, state: AgentState) -> Dict[str, Any]:
        """Perform contextual disambiguation"""
        try:
            disambiguation_results = {}
            
            for mention, candidates in state.candidates.items():
                if candidates:
                    # Simple disambiguation: pick the one with highest combined score
                    similarity_scores = state.metadata.get("similarity_scores", {}).get(mention, {})
                    type_scores = state.metadata.get("type_scores", {}).get(mention, {})
                    
                    best_candidate = None
                    best_score = 0.0
                    
                    for cand in candidates:
                        cand_id = cand['id']
                        similarity = similarity_scores.get(cand_id, 0.5)
                        type_compat = type_scores.get(cand_id, 0.5)
                        ed_score = cand.get('ed_score', 0.5)
                        popularity = cand.get('popularity', 0.5)
                        
                        # Combined score
                        combined_score = (
                            similarity * 0.3 +
                            type_compat * 0.2 +
                            ed_score * 0.3 +
                            popularity * 0.2
                        )
                        
                        if combined_score > best_score:
                            best_score = combined_score
                            best_candidate = cand
                    
                    disambiguation_results[mention] = {
                        "selected_candidate": best_candidate,
                        "confidence": best_score,
                        "reasoning": f"Selected based on combined score: {best_score:.3f}"
                    }
                else:
                    disambiguation_results[mention] = {
                        "selected_candidate": None,
                        "confidence": 0.0,
                        "reasoning": "No candidates found"
                    }
            
            return {
                "disambiguated_results": disambiguation_results,
                "metadata": {
                    **state.metadata,
                    "disambiguation_completed": True,
                    "disambiguation_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Contextual disambiguation failed: {e}")
            return {
                "processing_errors": [f"Disambiguation error: {str(e)}"]
            }
    
    async def rank_candidates(self, state: AgentState) -> Dict[str, Any]:
        """Rank candidates and calculate final confidence scores"""
        try:
            confidence_scores = {}
            
            for mention, result in state.disambiguated_results.items():
                if result.get('selected_candidate'):
                    confidence = result.get('confidence', 0.0)
                    confidence_scores[mention] = confidence
                else:
                    confidence_scores[mention] = 0.0
            
            return {
                "confidence_scores": confidence_scores,
                "metadata": {
                    **state.metadata,
                    "ranking_completed": True,
                    "final_confidence_calculated": True
                }
            }
            
        except Exception as e:
            logger.error(f"Ranking failed: {e}")
            return {
                "confidence_scores": {},
                "processing_errors": [f"Ranking error: {str(e)}"]
            }
    
    async def execute(self, state: AgentState) -> AgentState:
        """Execute the disambiguation workflow"""
        return await self.graph.ainvoke(state)