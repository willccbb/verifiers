"""
BLEURT-based rubric for semantic similarity evaluation.

This module provides a Rubric subclass that uses Google's BLEURT metric
to evaluate semantic similarity between generated text and reference text.
"""

import logging
from typing import Union, List, Dict, Any, Optional

from verifiers.rubrics import Rubric
from verifiers.parsers import Parser

try:
    from bleurt import score
    BLEURT_AVAILABLE = True
except ImportError:
    BLEURT_AVAILABLE = False


class BleurtRubric(Rubric):
    """
    Rubric that uses BLEURT for semantic similarity evaluation.
    
    BLEURT (BERT-based Learned Evaluation for Robust Text Generation) provides
    semantic similarity scores between candidate and reference texts, making it
    suitable for tasks where multiple valid formulations exist.
    
    Best for:
    - Text summarization
    - Paraphrasing tasks
    - Open-ended question answering
    - Any NLG task where semantic adequacy matters more than exact matching
    
    Example:
        # Basic usage
        rubric = BleurtRubric()
        
        # With custom configuration
        rubric = BleurtRubric(
            checkpoint_path="BLEURT-20-D12",
            bleurt_weight=0.8,
            add_format_reward=True,
            parser=XMLParser(['think', 'answer'])
        )
    """
    
    def __init__(self,
                 checkpoint_path: Optional[str] = None,
                 use_fast_model: bool = True,
                 bleurt_weight: float = 1.0,
                 add_format_reward: bool = False,
                 format_weight: float = 0.1,
                 parser: Parser = None,
                 batch_size: int = 16,
                 **kwargs):
        """
        Initialize BLEURT rubric.
        
        Args:
            checkpoint_path: Path to BLEURT checkpoint (auto-download if None)
            use_fast_model: Use distilled model for speed (recommended)
            bleurt_weight: Weight for BLEURT reward in rubric
            add_format_reward: Whether to add format compliance reward
            format_weight: Weight for format reward (if enabled)
            parser: Parser for structured outputs
            batch_size: Batch size for BLEURT evaluation
            **kwargs: Additional arguments passed to parent Rubric
        """
        if not BLEURT_AVAILABLE:
            raise ImportError(
                "BLEURT not available. Install with: pip install bleurt tensorflow"
            )
        
        super().__init__(parser=parser, **kwargs)
        
        self.checkpoint_path = checkpoint_path
        self.use_fast_model = use_fast_model
        self.batch_size = batch_size
        self.logger = logging.getLogger(f"verifiers.rubrics.{self.__class__.__name__}")
        
        # Lazy initialization
        self._scorer = None
        self._initialized = False
        
        # Add BLEURT reward function
        self.add_reward_func(self._bleurt_reward_func, weight=bleurt_weight)
        
        # Add format reward if requested
        if add_format_reward and parser and hasattr(parser, 'get_format_reward_func'):
            self.add_reward_func(parser.get_format_reward_func(), weight=format_weight)
    
    def _get_checkpoint_path(self) -> str:
        """Get or download BLEURT checkpoint."""
        if self.checkpoint_path:
            return self.checkpoint_path
            
        # Choose default checkpoint based on speed preference
        import os
        if self.use_fast_model:
            checkpoint_name = "BLEURT-20-D12"
            checkpoint_url = "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip"
        else:
            checkpoint_name = "BLEURT-20"
            checkpoint_url = "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip"
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_name):
            self.logger.info(f"Downloading {checkpoint_name} checkpoint...")
            try:
                import subprocess
                import urllib.request
                
                # Download and extract
                zip_path = f"{checkpoint_name}.zip"
                urllib.request.urlretrieve(checkpoint_url, zip_path)
                subprocess.run(["unzip", "-q", zip_path], check=True)
                os.remove(zip_path)
                
                self.logger.info(f"{checkpoint_name} checkpoint downloaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to download checkpoint: {e}")
                raise
        
        return checkpoint_name
    
    def _initialize_scorer(self):
        """Initialize BLEURT scorer lazily."""
        if self._initialized:
            return
            
        try:
            checkpoint_path = self._get_checkpoint_path()
            self.logger.info(f"Initializing BLEURT scorer with {checkpoint_path}")
            
            # Use length batching for better performance
            self._scorer = score.LengthBatchingBleurtScorer(checkpoint_path)
            self._initialized = True
            
            self.logger.info("BLEURT scorer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize BLEURT scorer: {e}")
            raise
    
    def _extract_text(self, 
                     text_input: Union[str, List[Dict[str, Any]]]) -> str:
        """
        Extract text from various input formats.
        
        Args:
            text_input: String, message list, or structured output
            
        Returns:
            Extracted text string
        """
        if isinstance(text_input, str):
            # For string inputs, use parser if available
            if self.parser and hasattr(self.parser, 'parse_answer'):
                parsed = self.parser.parse_answer(text_input)
                return parsed if parsed else text_input
            return text_input
        
        elif isinstance(text_input, list):
            # For message lists, extract assistant messages
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
        
        # Fallback to string conversion
        return str(text_input)
    
    def _bleurt_reward_func(self, 
                           completion: Union[str, List[Dict[str, Any]]], 
                           answer: Any, 
                           **kwargs) -> float:
        """
        BLEURT reward function implementation.
        
        Args:
            completion: Model completion
            answer: Reference answer/ground truth
            **kwargs: Additional arguments (ignored)
            
        Returns:
            BLEURT similarity score
        """
        try:
            # Initialize scorer if needed
            self._initialize_scorer()
            
            # Extract text from inputs
            candidate_text = self._extract_text(completion)
            reference_text = str(answer)
            
            # Ensure we have non-empty strings
            if not candidate_text.strip() or not reference_text.strip():
                return 0.0
            
            # Compute BLEURT score
            scores = self._scorer.score(
                candidates=[candidate_text],
                references=[reference_text],
                batch_size=1
            )
            
            return float(scores[0])
            
        except Exception as e:
            self.logger.error(f"Error computing BLEURT score: {e}")
            return 0.0


# Convenience function for easy integration
def create_bleurt_rubric(**kwargs) -> BleurtRubric:
    """
    Create a BLEURT rubric with sensible defaults.
    
    Args:
        **kwargs: Arguments passed to BleurtRubric constructor
        
    Returns:
        Configured BleurtRubric instance
    """
    return BleurtRubric(**kwargs)