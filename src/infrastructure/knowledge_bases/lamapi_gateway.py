import requests
import asyncio
from typing import List, Optional, Dict
from domain.entities import EntityCandidate, EntityType, KnowledgeBaseConfig
from domain.interfaces import KnowledgeBaseGateway
from domain.entities import ColumnType
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class LamAPIGateway(KnowledgeBaseGateway):
    """Implementation for LAMAPI knowledge bases (Wikidata)"""
    
    def __init__(self, config: KnowledgeBaseConfig):
        self.config = config
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_candidates(self, mention: str, context: Optional[Dict] = None) -> List[EntityCandidate]:
        """Get entity candidates from LAMAPI endpoint"""
        try:
            params = {
                "name": mention,
                "limit": 10,
                "token": self.config.credentials.get("token", "")
            }
            
            # Add optional parameters
            if "kind" in self.config.parameters:
                params["kind"] = self.config.parameters["kind"]
            if "kg" in self.config.parameters:
                params["kg"] = self.config.parameters["kg"]
            if "fuzzy" in self.config.parameters:
                params["fuzzy"] = str(self.config.parameters["fuzzy"]).lower()
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(self.config.url, params=params, timeout=30)
            )
            response.raise_for_status()
            
            raw_candidates = response.json()
            candidates = []
            
            # Handle the actual LamAPI response format
            for cand in raw_candidates:
                # Map types to EntityType objects
                types = []
                for type_info in cand.get("types", []):
                    types.append(EntityType(
                        id=type_info.get("id", ""),
                        name=type_info.get("name", ""),
                        description="",
                        source=self.config.name
                    ))
                
                candidates.append(EntityCandidate(
                    id=cand.get("id", ""),
                    name=cand.get("name", ""),
                    description=cand.get("description", ""),
                    types=types,
                    ed_score=cand.get("ed_score", 0.5),
                    popularity=cand.get("popularity", 0.1),
                    source_kb=self.config.name,
                    kb_specific_data=cand
                ))
            
            logger.debug(f"Retrieved {len(candidates)} candidates from {self.config.name} for '{mention}'")
            return candidates
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to {self.config.name} failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error processing response from {self.config.name}: {e}")
            return []
    
    def get_config(self) -> KnowledgeBaseConfig:
        return self.config
    
    async def health_check(self) -> bool:
        """Check if the knowledge base is accessible"""
        try:
            test_params = {
                "name": "test",
                "limit": 1,
                "token": self.config.credentials.get("token", "")
            }
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(self.config.url, params=test_params, timeout=10)
            )
            return response.status_code == 200
        except:
            return False