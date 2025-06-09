import asyncio
import logging
from typing import List, Dict, Any, Union, Optional

from .rubric import Rubric


class EmbedRubric(Rubric):
    """
    Base class for embedding-based rubrics using external API servers.
    
    Provides batch-optimized scoring via auxiliary hosting while maintaining
    the standard Rubric interface and verifiers design patterns.
    """
    
    def __init__(self, 
                 server_url: str,
                 timeout: float = 30.0,
                 parser = None,
                 **kwargs):
        super().__init__(parser=parser, **kwargs)
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.logger = logging.getLogger(f"verifiers.rubrics.{self.__class__.__name__}")
        self._session = None
        
    async def _get_session(self):
        """Get or create HTTP session."""
        import aiohttp
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
        
    async def batch_embed_score(self, 
                               completions: List[str], 
                               references: List[str]) -> List[float]:
        """
        Score completions against references via batch embedding API.
        
        Override in subclasses to implement specific API protocols.
        """
        raise NotImplementedError("Subclasses must implement batch_embed_score")
        
    def _extract_text(self, text_input: Union[str, List[Dict[str, Any]]]) -> str:
        """Extract text from completion formats."""
        if isinstance(text_input, str):
            if self.parser and hasattr(self.parser, 'parse_answer'):
                parsed = self.parser.parse_answer(text_input)
                return parsed if parsed else text_input
            return text_input
        
        elif isinstance(text_input, list):
            text_parts = []
            for msg in text_input:
                if isinstance(msg, dict) and msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    if self.parser and hasattr(self.parser, 'parse_answer'):
                        parsed = self.parser.parse_answer(content)
                        text_parts.append(parsed if parsed else content)
                    else:
                        text_parts.append(content)
            return ' '.join(text_parts)
        
        return str(text_input)
        
    async def _score_all(self,
                        prompts: List[Union[str, List[Dict[str, Any]]]],
                        completions: List[Union[str, List[Dict[str, Any]]]],
                        answers: List[Any],
                        states: List[Dict[str, Any]],
                        tasks: List[Optional[str]],
                        max_concurrent: int = 32,
                        **kwargs) -> Dict[str, List[float]]:
        """
        Override parent to use batch API calls instead of individual scoring.
        """
        completion_texts = [self._extract_text(c) for c in completions]
        answer_texts = [str(a) for a in answers]
        
        self.logger.info(f"Batch scoring {len(completion_texts)} completions via {self.server_url}")
        
        api_scores = await self.batch_embed_score(completion_texts, answer_texts)
        
        rewards = []
        for i, score in enumerate(api_scores):
            reward_dict = {}
            
            for func in self.get_reward_funcs():
                reward_dict[func.__name__] = score
                
            if self.get_reward_funcs():
                reward_dict['reward'] = sum(
                    score * weight 
                    for weight in self.get_reward_weights()
                )
            else:
                reward_dict['reward'] = score
                
            rewards.append(reward_dict)
            
        return {k: [item[k] for item in rewards] for k in rewards[0]} if rewards else {}
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session and not self._session.closed:
            await self._session.close()