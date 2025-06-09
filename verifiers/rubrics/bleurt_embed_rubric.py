from typing import List, Union, Dict, Any
import logging

from .embed_rubric import EmbedRubric


class BleurtEmbedRubric(EmbedRubric):
    """
    BLEURT rubric using auxiliary API server for process isolation.
    
    Provides semantic similarity scoring via external BLEURT server,
    avoiding in-process model loading that can interfere with training.
    """
    
    def __init__(self,
                 server_host: str = "localhost",
                 server_port: int = 8001,
                 parser = None,
                 **kwargs):
        server_url = f"http://{server_host}:{server_port}"
        super().__init__(server_url=server_url, parser=parser, **kwargs)
        
        # Add BLEURT scoring function in verifiers pattern
        self.add_reward_func(self._bleurt_embed_func, weight=1.0)
        
    async def batch_embed_score(self, 
                               completions: List[str], 
                               references: List[str]) -> List[float]:
        """Score completions against references using BLEURT API."""
        session = await self._get_session()
        
        payload = {
            "candidates": completions,
            "references": references
        }
        
        async with session.post(f"{self.server_url}/score", json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result["scores"]
            else:
                error_text = await response.text()
                self.logger.error(f"BLEURT API error {response.status}: {error_text}")
                return [0.0] * len(completions)
                
    def _bleurt_embed_func(self, 
                          completion: Union[str, List[Dict[str, Any]]], 
                          answer: Any, 
                          **kwargs) -> float:
        """
        Individual BLEURT function for verifiers compatibility.
        
        This maintains the standard rubric interface but is not used
        in batch mode due to _score_all override.
        """
        return 0.0