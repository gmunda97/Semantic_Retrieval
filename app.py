from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pickle
import pandas as pd

from scripts.semantic_search import SemanticSearch
from scripts.sentence_embeddings import ColumnNames

app = FastAPI()

dataset_path = "datasets/papers.csv"
embeddings_path = "embeddings/embeddings_papers.pkl"
index_path = "index/index_papers.index"
index_type = "flatIP"


@app.on_event("startup")
def load_data():
    global search_engine, df, column_names
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    CROSS_ENCODER_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    search_engine = SemanticSearch(
        embeddings=embeddings,
        index_type=index_type,
        index_file=index_path,
        cross_encoder_name=CROSS_ENCODER_NAME
    )
    column_names = ColumnNames()
    df = pd.read_csv(dataset_path)


class SearchRequest(BaseModel):
    query: str
    number_of_documents: Optional[int] = 10
    similarity_threshold: Optional[float] = 0.3


class SearchResponse(BaseModel):
    index: int
    title: str
    link: str
    similarity: float


@app.get("/")
def read_root():
    return {"Hello": "Welcome to the Semantic Search API"}


@app.post("/search", response_model=List[SearchResponse])
def search(request: SearchRequest):
    results = search_engine.retrieve_documents(
        query=request.query,
        text_data=df[column_names.title],
        link_data=df[column_names.link],
        number_of_documents=request.number_of_documents,
        similarity_threshold=request.similarity_threshold
    )

    if not results:
        raise HTTPException(status_code=404, detail="No documents found")

    return [
        SearchResponse(index=index, title=title, link=link, similarity=similarity)
        for index, title, similarity, link in results
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
