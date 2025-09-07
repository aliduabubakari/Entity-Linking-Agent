import requests
import asyncio
from typing import List, Optional, Dict
from domain.entities import EntityCandidate, EntityType, KnowledgeBaseConfig
from domain.interfaces import KnowledgeBaseGateway
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from urllib.parse import quote

logger = logging.getLogger(__name__)

class SPARQLGateway(KnowledgeBaseGateway):
    """Implementation for SPARQL endpoints (Wikidata, DBpedia, etc.)"""
    
    def __init__(self, config: KnowledgeBaseConfig):
        self.config = config
        self.endpoint_url = config.url
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_candidates(self, mention: str, context: Optional[Dict] = None) -> List[EntityCandidate]:
        """Get entity candidates from SPARQL endpoint"""
        try:
            # Determine which SPARQL query to use based on endpoint
            if "wikidata" in self.endpoint_url.lower():
                sparql_query = self._build_wikidata_query(mention)
            else:
                sparql_query = self._build_dbpedia_query(mention)
            
            params = {
                "query": sparql_query,
                "format": "application/sparql-results+json"
            }
            
            headers = {
                "Accept": "application/sparql-results+json",
                "User-Agent": "Entity-Linking-Agent/1.0"
            }
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(self.endpoint_url, params=params, headers=headers, timeout=30)
            )
            response.raise_for_status()
            
            results = response.json()
            candidates = self._parse_sparql_results(results, mention)
            
            logger.debug(f"Retrieved {len(candidates)} candidates from {self.config.name} for '{mention}'")
            return candidates
            
        except requests.exceptions.RequestException as e:
            logger.error(f"SPARQL request to {self.config.name} failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error processing SPARQL response from {self.config.name}: {e}")
            return []
    
    def _build_wikidata_query(self, mention: str) -> str:
        """Build SPARQL query for Wikidata"""
        escaped_mention = mention.replace('"', '\\"')
        
        query = f"""
        SELECT DISTINCT ?item ?itemLabel ?itemDescription ?instanceLabel WHERE {{
            ?item rdfs:label ?label .
            FILTER(CONTAINS(LCASE(?label), LCASE("{escaped_mention}")))
            
            OPTIONAL {{ ?item wdt:P31 ?instance . }}
            
            SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". 
            }}
            
            FILTER(LANG(?label) = "en")
        }}
        ORDER BY ?itemLabel
        LIMIT 10
        """
        
        return query.strip()
    
    def _build_dbpedia_query(self, mention: str) -> str:
        """Build SPARQL query for DBpedia"""
        escaped_mention = mention.replace('"', '\\"')
        
        query = f"""
        SELECT DISTINCT ?entity ?label ?abstract ?type WHERE {{
            ?entity rdfs:label ?label .
            FILTER(CONTAINS(LCASE(?label), LCASE("{escaped_mention}")))
            
            OPTIONAL {{ ?entity dbo:abstract ?abstract . }}
            OPTIONAL {{ ?entity rdf:type ?type . }}
            
            FILTER(LANG(?label) = "" || LANG(?label) = "en")
            FILTER(LANG(?abstract) = "" || LANG(?abstract) = "en")
        }}
        ORDER BY ?label
        LIMIT 10
        """
        
        return query.strip()
    
    def _parse_sparql_results(self, results: Dict, mention: str) -> List[EntityCandidate]:
        """Parse SPARQL JSON results into EntityCandidate objects"""
        candidates = []
        
        try:
            bindings = results.get("results", {}).get("bindings", [])
            
            # Group results by entity URI
            entity_data = {}
            for binding in bindings:
                entity_uri = binding.get("item", binding.get("entity", {})).get("value", "")
                if not entity_uri:
                    continue
                
                if entity_uri not in entity_data:
                    entity_data[entity_uri] = {
                        "uri": entity_uri,
                        "labels": set(),
                        "descriptions": set(),
                        "types": set()
                    }
                
                # Collect labels
                label_key = "itemLabel" if "itemLabel" in binding else "label"
                if label_key in binding:
                    label = binding[label_key].get("value", "")
                    if label:
                        entity_data[entity_uri]["labels"].add(label)
                
                # Collect descriptions
                desc_key = "itemDescription" if "itemDescription" in binding else "abstract"
                if desc_key in binding:
                    desc = binding[desc_key].get("value", "")
                    if desc:
                        entity_data[entity_uri]["descriptions"].add(desc)
                
                # Collect types
                type_key = "instanceLabel" if "instanceLabel" in binding else "type"
                if type_key in binding:
                    type_uri = binding[type_key].get("value", "")
                    if type_uri:
                        entity_data[entity_uri]["types"].add(type_uri)
            
            # Convert to EntityCandidate objects
            for uri, data in entity_data.items():
                labels = list(data["labels"])
                if not labels:
                    continue
                
                best_label = min(labels, key=len)
                descriptions = list(data["descriptions"])
                best_description = descriptions[0][:500] if descriptions else ""  # Limit description length
                
                # Convert types to EntityType objects
                entity_types = []
                for type_uri in list(data["types"])[:5]:  # Limit to 5 types
                    type_name = self._extract_type_name(type_uri)
                    entity_types.append(EntityType(
                        id=type_uri,
                        name=type_name,
                        source=self.config.name
                    ))
                
                # Calculate similarity
                similarity_score = self._calculate_similarity(mention, best_label)
                
                candidates.append(EntityCandidate(
                    id=uri,
                    name=best_label,
                    description=best_description,
                    types=entity_types,
                    ed_score=similarity_score,
                    popularity=0.5,  # Default popularity for SPARQL results
                    source_kb=self.config.name,
                    kb_specific_data={
                        "uri": uri,
                        "all_labels": labels,
                        "all_types": list(data["types"])
                    }
                ))
            
            # Sort by similarity score
            candidates.sort(key=lambda x: x.ed_score or 0, reverse=True)
            
        except Exception as e:
            logger.error(f"Error parsing SPARQL results: {e}")
        
        return candidates
    
    def _extract_type_name(self, type_uri: str) -> str:
        """Extract a readable type name from URI"""
        if "/" in type_uri:
            return type_uri.split("/")[-1]
        elif "#" in type_uri:
            return type_uri.split("#")[-1]
        else:
            return type_uri
    
    def _calculate_similarity(self, mention: str, candidate_name: str) -> float:
        """Calculate simple string similarity"""
        mention_lower = mention.lower().strip()
        name_lower = candidate_name.lower().strip()
        
        if mention_lower == name_lower:
            return 1.0
        elif mention_lower in name_lower or name_lower in mention_lower:
            return 0.8
        else:
            # Simple Jaccard similarity on words
            mention_words = set(mention_lower.split())
            name_words = set(name_lower.split())
            
            if not mention_words or not name_words:
                return 0.1
            
            intersection = len(mention_words.intersection(name_words))
            union = len(mention_words.union(name_words))
            
            return intersection / union if union > 0 else 0.1
    
    def get_config(self) -> KnowledgeBaseConfig:
        return self.config
    
    async def health_check(self) -> bool:
        """Check if the SPARQL endpoint is accessible"""
        try:
            # Simple ASK query to test connectivity
            test_query = "ASK { ?s ?p ?o } LIMIT 1"
            
            params = {
                "query": test_query,
                "format": "application/sparql-results+json"
            }
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(self.endpoint_url, params=params, timeout=10)
            )
            return response.status_code == 200
        except:
            return False