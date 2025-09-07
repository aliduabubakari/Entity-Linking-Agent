import requests
import asyncio
import json
import time
import uuid
from typing import List, Optional, Dict
from domain.entities import EntityCandidate, EntityType, KnowledgeBaseConfig
from domain.interfaces import KnowledgeBaseGateway
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class AlligatorGateway(KnowledgeBaseGateway):
    """Optimized Alligator gateway with efficient polling"""
    
    def __init__(self, config: KnowledgeBaseConfig):
        self.config = config
        self.base_url = config.url.rstrip('/')
        self.token = config.credentials.get("token", "")
    
    async def get_candidates(self, mention: str, context: Optional[Dict] = None) -> List[EntityCandidate]:
        """Get entity candidates using optimized Alligator workflow"""
        try:
            logger.info(f"Starting Alligator search for: '{mention}'")
            
            # Step 1: Create the dataset
            dataset_info = await self._create_simple_dataset(mention, context)
            if not dataset_info:
                return []
            
            # Step 2: Poll for results with shorter intervals
            candidates = await self._poll_for_results(dataset_info, max_attempts=8, initial_delay=3)
            
            logger.info(f"Alligator returned {len(candidates)} candidates for '{mention}'")
            return candidates
            
        except Exception as e:
            logger.error(f"Alligator gateway error for '{mention}': {e}")
            return []
    
    async def _create_simple_dataset(self, mention: str, context: Optional[Dict] = None) -> Optional[Dict]:
        """Create a minimal dataset optimized for single entity lookup"""
        try:
            # Generate unique identifiers
            timestamp = int(time.time() * 1000)
            dataset_name = f"EL-{timestamp}"
            table_name = f"T-{timestamp}"
            
            # Create minimal table structure
            table_data = {
                "datasetName": dataset_name,
                "tableName": table_name,
                "header": ["Entity", "Type", "Context"],
                "rows": [
                    {
                        "idRow": 1,
                        "data": [
                            mention,
                            "Person",  # Default assumption
                            context.get("table_context", {}).get("domain", "general") if context else "general"
                        ]
                    }
                ],
                "semanticAnnotations": {"cea": [], "cta": [], "cpa": []},
                "metadata": {
                    "column": [
                        {"idColumn": 0, "tag": "NE"},  # Named Entity
                        {"idColumn": 1, "tag": "LIT"}, # Literal
                        {"idColumn": 2, "tag": "LIT"}  # Literal
                    ]
                },
                "kgReference": "wikidata"
            }
            
            # Make the POST request
            create_url = f"{self.base_url}/dataset/createWithArray"
            params = {"token": self.token}
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            
            logger.debug(f"Creating dataset: {dataset_name}")
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(create_url, params=params, json=[table_data], headers=headers, timeout=15)
            )
            
            logger.debug(f"Dataset creation response: {response.status_code}")
            
            if response.status_code in [200, 201, 202]:
                return {
                    "dataset_name": dataset_name,
                    "table_name": table_name,
                    "timestamp": timestamp
                }
            else:
                logger.error(f"Failed to create dataset: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            return None
    
    async def _poll_for_results(self, dataset_info: Dict, max_attempts: int = 8, initial_delay: int = 3) -> List[EntityCandidate]:
        """Poll for results with optimized timing"""
        
        dataset_patterns = [
            "EMD-BC",  # From your logs - seems to be the main pattern
            f"EMD-{dataset_info['dataset_name']}",
            dataset_info['dataset_name'],
        ]
        
        table_name = dataset_info['table_name']
        
        # Try immediate check first (sometimes it's ready quickly)
        await asyncio.sleep(1)
        
        for attempt in range(max_attempts):
            current_delay = initial_delay + (attempt * 2)  # 3, 5, 7, 9, 11, 13, 15, 17 seconds
            
            logger.debug(f"Polling attempt {attempt + 1}/{max_attempts}")
            
            for pattern in dataset_patterns:
                try:
                    results_url = f"{self.base_url}/dataset/{pattern}/table/{table_name}"
                    params = {"page": 1, "per_page": 20, "token": self.token}
                    
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: requests.get(results_url, params=params, timeout=10)
                    )
                    
                    if response.status_code == 200:
                        try:
                            results = response.json()
                            candidates = self._parse_results(results)
                            if candidates:
                                logger.info(f"Success! Got results from {pattern} after {attempt + 1} attempts")
                                return candidates
                        except json.JSONDecodeError:
                            logger.debug(f"Invalid JSON response from {pattern}")
                            continue
                    elif response.status_code == 404:
                        logger.debug(f"Table not found in {pattern} (still processing?)")
                    else:
                        logger.debug(f"Unexpected status {response.status_code} from {pattern}")
                        
                except Exception as e:
                    logger.debug(f"Request failed for {pattern}: {e}")
                    continue
            
            # Wait before next attempt, but don't wait after the last attempt
            if attempt < max_attempts - 1:
                logger.debug(f"Waiting {current_delay}s before next attempt...")
                await asyncio.sleep(current_delay)
        
        logger.warning(f"Failed to get results after {max_attempts} attempts")
        return []
    
    def _parse_results(self, results) -> List[EntityCandidate]:
        """Parse API results into EntityCandidate objects"""
        candidates = []
        
        try:
            # Handle different response formats
            if isinstance(results, list):
                entities = results
            elif isinstance(results, dict):
                # Try various keys that might contain the results
                entities = (results.get("results") or 
                          results.get("entities") or 
                          results.get("data") or 
                          results.get("predictions") or
                          results.get("annotations") or
                          [])
            else:
                return []
            
            logger.debug(f"Parsing {len(entities)} entities from results")
            
            for entity in entities:
                if isinstance(entity, dict) and entity.get("id") and entity.get("name"):
                    # Extract entity information
                    entity_id = entity.get("id", "")
                    entity_name = entity.get("name", "Unknown")
                    score = float(entity.get("score", 0.5))
                    
                    # Create description
                    description = f"Wikidata entity: {entity_name}"
                    if entity.get("description"):
                        description = entity["description"]
                    
                    # Create entity type
                    entity_type = EntityType(
                        id=entity_id,
                        name=entity_name,
                        source=self.config.name
                    )
                    
                    candidates.append(EntityCandidate(
                        id=entity_id,
                        name=entity_name,
                        description=description,
                        types=[entity_type],
                        ed_score=score,
                        popularity=score,
                        source_kb=self.config.name,
                        kb_specific_data=entity
                    ))
            
            # Sort by score
            candidates.sort(key=lambda x: x.ed_score, reverse=True)
            
        except Exception as e:
            logger.error(f"Error parsing results: {e}")
        
        return candidates
    
    def get_config(self) -> KnowledgeBaseConfig:
        return self.config
    
    async def health_check(self) -> bool:
        """Quick health check"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(self.base_url, timeout=5)
            )
            return response.status_code in [200, 404, 405, 403]
        except:
            return False