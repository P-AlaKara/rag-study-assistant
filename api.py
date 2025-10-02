import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from app import (
	RAG_RETRIEVER,
	VECTORSTORE,
	llm,
	enhanced_past_paper,
	create_rag_chain,
	create_quiz_chain,
)

app = FastAPI(title="Study Assistant API", version="1.0.0")


if not RAG_RETRIEVER or not VECTORSTORE:
	raise RuntimeError("Vector store is not initialized. Run the indexing pipeline first.")

RAG_CHAIN = create_rag_chain(llm, RAG_RETRIEVER)
QUIZ_CHAIN = create_quiz_chain(llm, RAG_RETRIEVER)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


class RouteRequest(BaseModel):
	message: str
	sessionId: Optional[str] = "api"


class PastPaperStartRequest(BaseModel):
	sessionId: str
	unitCode: Optional[str] = None
	year: Optional[str] = None
	message: Optional[str] = None


class PastPaperContinueRequest(BaseModel):
	sessionId: str


class PastPaperClarifyRequest(BaseModel):
	sessionId: str
	questionNumber: int


class PastPaperAnswerRequest(BaseModel):
	sessionId: str
	questionNumber: int
	answer: str


class QARequest(BaseModel):
	sessionId: Optional[str] = "api"
	message: str


class QuizRequest(BaseModel):
	sessionId: Optional[str] = "api"
	topic: str


@app.post("/api/pastpaper/start")
def start_past_paper(req: PastPaperStartRequest):
	if not enhanced_past_paper:
		raise HTTPException(status_code=500, detail="Past paper flow unavailable. Index the data first.")

	# Build a user message from provided fields if not given
	message = req.message or ""
	if req.unitCode and req.year and not message:
		message = f"Go through {req.unitCode} {req.year} past paper"

	response = enhanced_past_paper.handle_past_paper_request(message, session_id=req.sessionId)
	return {"response": response}


@app.post("/api/pastpaper/continue")
def continue_past_paper(req: PastPaperContinueRequest):
	if not enhanced_past_paper:
		raise HTTPException(status_code=500, detail="Past paper flow unavailable. Index the data first.")
	response = enhanced_past_paper.handle_past_paper_request("next", session_id=req.sessionId)
	return {"response": response}


@app.post("/api/pastpaper/clarify")
def clarify_past_paper(req: PastPaperClarifyRequest):
	if not enhanced_past_paper:
		raise HTTPException(status_code=500, detail="Past paper flow unavailable. Index the data first.")
	message = f"clarify question {req.questionNumber}"
	response = enhanced_past_paper.handle_past_paper_request(message, session_id=req.sessionId)
	return {"response": response}


@app.post("/api/pastpaper/answer")
def answer_past_paper(req: PastPaperAnswerRequest):
	if not enhanced_past_paper:
		raise HTTPException(status_code=500, detail="Past paper flow unavailable. Index the data first.")
	message = f"my answer for question {req.questionNumber} is {req.answer}"
	response = enhanced_past_paper.handle_past_paper_request(message, session_id=req.sessionId)
	return {"response": response}


@app.post("/api/qa")
def qa(req: QARequest):
	if not RAG_CHAIN:
		raise HTTPException(status_code=500, detail="QA chain unavailable. Index the data first.")
	answer = RAG_CHAIN.invoke({"question": req.message})
	return {"answer": answer}


@app.post("/api/quiz")
def quiz(req: QuizRequest):
	if not QUIZ_CHAIN:
		raise HTTPException(status_code=500, detail="Quiz chain unavailable. Index the data first.")
	answer = QUIZ_CHAIN.invoke({"question": req.topic})
	return {"quiz": answer}



@app.get("/health")
def health():
	return {"status": "ok"}


