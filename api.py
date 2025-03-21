from typing import List, Optional

from fastapi import FastAPI, HTTPException
from local_rag import LocalRAG
from pydantic import BaseModel

app = FastAPI()
rag = LocalRAG()


class Query(BaseModel):
    text: str
    history: Optional[List[dict]] = None


@app.post("/query")
async def query_rag(query: Query):
    try:
        response = rag.get_rag_response(query.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
