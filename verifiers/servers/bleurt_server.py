import asyncio
import logging
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Global scorer - lazy loaded
_scorer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BLEURT Auxiliary Server", version="1.0.0")


class ScoreRequest(BaseModel):
    candidates: List[str]
    references: List[str]


class ScoreResponse(BaseModel):
    scores: List[float]


async def get_scorer():
    """Lazy load BLEURT scorer."""
    global _scorer
    if _scorer is None:
        try:
            from bleurt import score
            logger.info("Loading BLEURT scorer...")
            _scorer = score.LengthBatchingBleurtScorer("BLEURT-20-D12")
            logger.info("BLEURT scorer loaded successfully")
        except ImportError:
            raise HTTPException(status_code=500, detail="BLEURT package not installed")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load BLEURT: {str(e)}")
    return _scorer


@app.post("/score", response_model=ScoreResponse)
async def score_texts(request: ScoreRequest):
    """Score candidates against references using BLEURT."""
    if len(request.candidates) != len(request.references):
        raise HTTPException(
            status_code=400, 
            detail="Candidates and references must have the same length"
        )
    
    if not request.candidates:
        return ScoreResponse(scores=[])
    
    scorer = await get_scorer()
    
    try:
        scores = await asyncio.to_thread(
            scorer.score,
            candidates=request.candidates,
            references=request.references
        )
        
        return ScoreResponse(scores=scores.tolist())
    
    except Exception as e:
        logger.error(f"BLEURT scoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": "BLEURT-20-D12"}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "BLEURT Auxiliary Server",
        "version": "1.0.0",
        "endpoints": {
            "score": "POST /score - Score text pairs",
            "health": "GET /health - Health check"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)