import requests
import asyncio
from typing import List, Optional, Dict
from domain.entities import EntityCandidate, EntityType, KnowledgeBaseConfig
from domain.interfaces import KnowledgeBaseGateway
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class GeoNamesGateway(KnowledgeBaseGateway):
    """Implementation for GeoNames knowledge base"""
    
    def __init__(self, config: KnowledgeBaseConfig):
        self.config = config
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_candidates(self, mention: str, context: Optional[Dict] = None) -> List[EntityCandidate]:
        """Get location candidates from GeoNames"""
        try:
            params = {
                "q": mention,
                "maxRows": self.config.parameters.get("maxRows", 10),
                "username": self.config.credentials.get("username", "demo"),
                "style": self.config.parameters.get("style", "FULL")
            }
            
            # Use asyncio to run the synchronous request
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.get(self.config.url, params=params, timeout=30)
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Check for error in response
            if "status" in data:
                logger.error(f"GeoNames API error: {data.get('status', {}).get('message', 'Unknown error')}")
                return []
            
            candidates = []
            
            for place in data.get("geonames", []):
                entity_type = EntityType(
                    id=place.get("fcode", ""),
                    name=place.get("fcodeName", "Location"),
                    source="geonames"
                )
                
                # Calculate similarity score
                similarity_score = self._calculate_similarity(mention, place.get("name", ""))
                
                candidates.append(EntityCandidate(
                    id=f"geoname:{place['geonameId']}",
                    name=place.get("name", ""),
                    description=self._create_description(place),
                    types=[entity_type],
                    ed_score=similarity_score,
                    popularity=float(place.get("population", 0)) / 1000000 if place.get("population") else 0.1,
                    source_kb=self.config.name,
                    kb_specific_data=place
                ))
            
            logger.debug(f"Retrieved {len(candidates)} candidates from {self.config.name} for '{mention}'")
            return candidates
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to {self.config.name} failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error processing response from {self.config.name}: {e}")
            return []
    
    def _calculate_similarity(self, mention: str, candidate_name: str) -> float:
        """Calculate simple string similarity"""
        mention_lower = mention.lower().strip()
        name_lower = candidate_name.lower().strip()
        
        if mention_lower == name_lower:
            return 1.0
        elif mention_lower in name_lower or name_lower in mention_lower:
            return 0.8
        else:
            # Simple word overlap
            mention_words = set(mention_lower.split())
            name_words = set(name_lower.split())
            
            if not mention_words or not name_words:
                return 0.1
            
            intersection = len(mention_words.intersection(name_words))
            union = len(mention_words.union(name_words))
            
            return intersection / union if union > 0 else 0.1
    
    def _create_description(self, place: Dict) -> str:
        """Create a descriptive string from GeoNames data"""
        parts = []
        if place.get("adminName1"):
            parts.append(place["adminName1"])
        if place.get("countryName"):
            parts.append(place["countryName"])
        if place.get("fcodeName"):
            parts.append(place["fcodeName"])
        return ", ".join(parts) if parts else place.get("name", "")
    
    def get_config(self) -> KnowledgeBaseConfig:
        return self.config
    
    async def health_check(self) -> bool:
        try:
            test_params = {
                "q": "test",
                "maxRows": 1,
                "username": self.config.credentials.get("username", "demo")
            }
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(self.config.url, params=test_params, timeout=10)
            )
            return response.status_code == 200
        except:
            return False