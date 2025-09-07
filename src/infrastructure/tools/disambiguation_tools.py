from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

async def calculate_semantic_similarity(candidate: Dict, context: Dict, mention: str) -> float:
    """Calculate semantic similarity between mention and candidate"""
    try:
        # Simple string similarity - in production, use proper embedding models
        candidate_name = candidate.get('name', '').lower()
        mention_lower = mention.lower()
        
        if candidate_name == mention_lower:
            return 1.0
        elif mention_lower in candidate_name or candidate_name in mention_lower:
            return 0.8
        else:
            # Use edit distance or other similarity metrics
            return candidate.get('ed_score', 0.5)
    except Exception as e:
        logger.error(f"Similarity calculation failed: {e}")
        return 0.5

async def analyze_type_compatibility(candidate: Dict, column_type: str, context: Dict) -> float:
    """Analyze type compatibility between candidate and column"""
    try:
        candidate_types = [t.get('name', '') for t in candidate.get('types', [])]
        if not candidate_types:
            return 0.3
        
        # Simple type matching
        if column_type.upper() in [t.upper() for t in candidate_types]:
            return 1.0
        
        # Partial matching for related types
        related_types = {
            'PERSON': ['HUMAN', 'INDIVIDUAL', 'PEOPLE'],
            'LOCATION': ['PLACE', 'GEOGRAPHICAL', 'CITY', 'COUNTRY'],
            'ORGANIZATION': ['COMPANY', 'INSTITUTION', 'CORP']
        }
        
        for cand_type in candidate_types:
            if column_type in related_types.get(cand_type.upper(), []):
                return 0.8
        
        return 0.5
    except Exception as e:
        logger.error(f"Type compatibility analysis failed: {e}")
        return 0.5

async def llm_contextual_disambiguation(
    candidates: List[Dict], 
    mention: str, 
    column_type: str, 
    table_context: Dict, 
    llm
) -> Dict:
    """Use LLM for contextual disambiguation"""
    try:
        if not candidates:
            return {"selected_candidate": None, "confidence": 0.0, "reasoning": "No candidates available"}
        
        if len(candidates) == 1:
            return {
                "selected_candidate": candidates[0],
                "confidence": 0.9,
                "reasoning": "Single candidate found"
            }
        
        # Simple fallback disambiguation based on edit distance
        best_candidate = max(candidates, key=lambda c: c.get('ed_score', 0))
        
        return {
            "selected_candidate": best_candidate,
            "confidence": 0.7,
            "reasoning": f"Selected based on highest similarity score: {best_candidate.get('ed_score', 0)}"
        }
        
    except Exception as e:
        logger.error(f"LLM disambiguation failed: {e}")
        return {
            "selected_candidate": candidates[0] if candidates else None,
            "confidence": 0.5,
            "reasoning": f"Fallback selection due to error: {str(e)}"
        }