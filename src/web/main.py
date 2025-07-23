from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

from ..database.queries import DatabaseQueries
from ..llm.qwen import QwenLLM
from ..judge.engine import MTGJudgeEngine
from ..rag.vector_store import FAISSVectorStore

# Initialize FastAPI app
app = FastAPI(
    title="MTG Judge Engine",
    description="Magic: The Gathering judge chat engine with AI-powered rulings",
    version="0.1.0"
)

# Setup templates and static files
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Create static directory if it doesn't exist
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global instances
db_queries = None
judge_engine = None
vector_store = None


# Pydantic models
class QuestionRequest(BaseModel):
    question: str


class CorrectionRequest(BaseModel):
    question: str
    original_answer: str
    corrected_answer: str
    feedback: Optional[str] = None


class CardSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 20


@app.on_event("startup")
async def startup_event():
    """Initialize database, vector store, and LLM on startup."""
    global db_queries, judge_engine, vector_store
    
    # Initialize database
    db_queries = DatabaseQueries()
    
    # Initialize vector store
    vector_store = FAISSVectorStore()
    
    # Load data into vector store if empty
    if vector_store.get_stats()["total_documents"] == 0:
        await _populate_vector_store()
    
    # Initialize Qwen LLM
    try:
        llm = QwenLLM(
            model_name="Qwen/Qwen3-8B",
            temperature=0.7,
            max_tokens=2000
        )
        
        if not llm.is_available():
            raise RuntimeError("Qwen model is not available")
            
    except Exception as e:
        print(f"Warning: Could not initialize Qwen LLM: {e}")
        # Create a dummy LLM for development
        from ..llm.base import BaseLLM, LLMResponse
        
        class DummyLLM(BaseLLM):
            def generate(self, prompt: str, **kwargs) -> LLMResponse:
                return LLMResponse(
                    text=f"LLM initialization failed: {str(e)}. Please check your configuration.",
                    model_name="dummy"
                )
            
            def generate_with_tools(self, prompt: str, tools: List[Dict[str, Any]]) -> LLMResponse:
                return self.generate(prompt)
            
            def is_available(self) -> bool:
                return True
        
        llm = DummyLLM("dummy")
    
    # Initialize judge engine
    judge_engine = MTGJudgeEngine(llm, vector_store)


async def _populate_vector_store():
    """Populate vector store with data from database."""
    global vector_store, db_queries
    
    if not db_queries or not vector_store:
        return
    
    try:
        # Load rules
        rules = db_queries.get_all_rules()
        if rules:
            vector_store.add_rules(rules)
            logger.info(f"Added {len(rules)} rules to vector store")
        
        # Load cards (limit to avoid memory issues)
        cards = db_queries.get_all_cards(limit=10000)  
        if cards:
            vector_store.add_cards(cards)
            logger.info(f"Added {len(cards)} cards to vector store")
        
        # Load rulings
        rulings = db_queries.get_all_rulings(limit=5000)
        if rulings:
            vector_store.add_rulings(rulings)
            logger.info(f"Added {len(rulings)} rulings to vector store")
        
        # Save the populated index
        vector_store.save_index()
        logger.info("Vector store populated and saved")
        
    except Exception as e:
        logger.error(f"Error populating vector store: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    global db_queries, vector_store
    if db_queries:
        db_queries.close()
    if vector_store:
        vector_store.save_index()


# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/ask")
async def ask_question(request: QuestionRequest) -> Dict[str, Any]:
    """Ask a judge question."""
    if not judge_engine:
        raise HTTPException(status_code=500, detail="Judge engine not initialized")
    
    try:
        result = judge_engine.answer_question(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/api/cards/search")
async def search_cards(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search for cards."""
    if not db_queries:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        cards = db_queries.search_cards(query, limit)
        return cards
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cards/suggestions")
async def get_card_suggestions(query: str, limit: int = 10) -> List[str]:
    """Get card name suggestions for typeahead."""
    if not db_queries:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        suggestions = db_queries.get_card_suggestions(query, limit)
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cards/{card_name}")
async def get_card(card_name: str) -> Dict[str, Any]:
    """Get a specific card by name."""
    if not db_queries:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        card = db_queries.get_card_by_name(card_name)
        if not card:
            raise HTTPException(status_code=404, detail="Card not found")
        
        # Get rulings for the card
        rulings = db_queries.get_rulings_for_card(card["oracle_id"])
        card["rulings"] = rulings
        
        return card
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rules/search")
async def search_rules(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search comprehensive rules."""
    if not db_queries:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        rules = db_queries.search_rules(query, limit)
        return rules
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rulings/search")
async def search_rulings(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search rulings."""
    if not db_queries:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        rulings = db_queries.search_rulings(query, limit)
        return rulings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats() -> Dict[str, Any]:
    """Get database and vector store statistics."""
    if not db_queries:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        db_stats = db_queries.get_stats()
        
        # Add vector store stats if available
        if vector_store:
            vector_stats = vector_store.get_stats()
            return {
                "database": db_stats,
                "vector_store": vector_stats
            }
        
        return {"database": db_stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)