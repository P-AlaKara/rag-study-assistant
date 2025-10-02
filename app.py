import os
from dotenv import load_dotenv


load_dotenv()

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb import PersistentClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter 
import re
from pastpaper_handler import EnhancedPastPaperChain

CHROMA_PERSIST_DIR = './chroma_db'
COLLECTION_NAME = "student_notes_kb"
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
LLM_MODEL = "gemini-2.0-flash-exp"

def initialize_retriever():
    """Loads the persistent Chroma Vector Store and returns a Retriever instance."""
    print("--- Initializing Retriever from Persistent DB ---")
    
    try:
        chroma_client = PersistentClient(path=CHROMA_PERSIST_DIR)
    except Exception as e:
        print(f"Error initializing Chroma client: {e}")
        print("Did you forget to run the indexing pipeline?")
        return None
        
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=EMBEDDINGS
    )
    
    # Use similarity score threshold to avoid low-relevance distractors.
    rag_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 4,
            "score_threshold": 0.4,
        },
    )
    
    print("Knowledge Base loaded and Retriever is ready.")
    return rag_retriever, vectorstore

RAG_RETRIEVER, VECTORSTORE = initialize_retriever()

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL, 
    temperature=0.0
)

router_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """Analyze the user's request and output ONLY one of these words:
     - 'PASTPAPER' if the user wants to go through, practice, or work through a past paper/exam
     - 'QUIZ' if the user is asking for questions, a test, practice problems, or a quiz (but not specifically a past paper)
     - 'QA' for all other questions, factual queries, requests for explanations, or general greetings
     
     Look for keywords like: "past paper", "previous exam", "go through exam", "work through paper", "2023 paper", etc.
     Do not include any other punctuation or text in your output."""),
    ("human", "{question}")
])

router_chain = router_prompt | llm | StrOutputParser()


def get_past_paper_retriever(unit_code=None, year=None):
    """Creates a retriever specifically for past papers with filters.

    Chroma v0.5+ expects a top-level logical operator in the where/filter clause.
    We therefore always wrap equality conditions in a $and with $eq operators.
    """
    if not VECTORSTORE:
        return None

    conditions = [{"source_type": {"$eq": "PastPaper"}}]
    if unit_code:
        conditions.append({"unit_code": {"$eq": unit_code.upper()}})
    if year:
        conditions.append({"year": {"$eq": str(year)}})

    chroma_where = {"$and": conditions}

    return VECTORSTORE.as_retriever(
        search_kwargs={
            "k": 10,  # Get more chunks for past papers
            "filter": chroma_where
        }
    )


enhanced_past_paper = EnhancedPastPaperChain(llm, VECTORSTORE) if VECTORSTORE else None


def create_rag_chain(llm, retriever):
    """Creates the Retrieval-Augmented Generation chain."""

    # graceful fallback to general knowledge when context is insufficient
    rag_prompt_template = """
    You are a study assistant. Prefer to answer using ONLY the provided context (student notes) when it is sufficient.
    If the context is missing, insufficient, or not relevant to the question, you MUST still answer using general knowledge.

    INSTRUCTIONS:
    1. Answer concisely and accurately.
    2. If the answer can be derived from the context, include citations by appending the 'source_type' and 'unit_code' from metadata
       (e.g., [Notes: CSC231]). Only cite chunks you actually used.
    3. If the context does not clearly contain the necessary information, do NOT refuse. Answer using general knowledge and append: (source:internet)
    4. Do NOT say "I am sorry" or that the document/notes do not contain the information. Always provide the best possible answer.
    5. Do not mix sources in a single answer. Choose either notes citations OR [Source: Internet].

    CONTEXT:
    ---
    {context}
    ---
    QUESTION: {question}
    ANSWER:
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)

    # Fallback (no KB match): answer from general knowledge and tag internet source
    fallback_prompt_template = """
    Disclaimer: This answer is not found in your provided study material. Using general knowledge.

    Provide a clear, concise answer to the user's question. If there is uncertainty or multiple interpretations, note them briefly.
    Do NOT refuse. Do NOT say the document/notes do not contain the information. Always answer helpfully.
    At the end, append: (source:internet)

    QUESTION: {question}
    ANSWER:
    """
    fallback_prompt = ChatPromptTemplate.from_template(fallback_prompt_template)

    def format_docs(docs):
        return "\n\n".join([
            f"Document Source: {doc.metadata.get('source_type')}_{doc.metadata.get('unit_code')}\nContent: {doc.page_content}"
            for doc in docs
        ])

    # branching chain based on whether any documents were retrieved
    # Step 1: retrieve docs alongside the question
    pre = {
        "docs": itemgetter("question") | retriever,
        "question": itemgetter("question")
    }

    # Path A (fallback): no docs
    fallback_chain = fallback_prompt | llm | StrOutputParser()

    # Path B (RAG): docs exist
    rag_chain = (
        {
            "context": itemgetter("docs") | RunnableLambda(format_docs),
            "question": itemgetter("question")
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    branched = pre | RunnableBranch(
        (lambda x: len(x["docs"]) == 0, fallback_chain),
        rag_chain
    )

    return branched


def create_quiz_chain(llm, retriever):
    """Creates a chain to generate quizzes based on retrieved context."""
    
    quiz_prompt_template = """
    You are an expert academic quiz generator. Prefer to base questions on the provided study material. If the context is sparse or lacks detail, you MUST still produce a high-quality quiz by supplementing with general knowledge. Do not refuse.
    
    INSTRUCTIONS:
    1. Generate exactly 5 multiple-choice questions (MCQs) about the user's requested topic.
    2. Each question must have 4 options (A, B, C, D).
    3. Clearly indicate the correct answer for each question at the end of the response.
    4. Output format MUST be:
       - For each question: a new line starting with "Question {{n}}:"
       - Each option on its own line, like: "A. ...", "B. ..." etc
       - After all 5 questions, include a section starting with "Answers:" followed by "1) X" per line

    CONTEXT:
    ---
    {context}
    ---
    TOPIC REQUESTED BY USER: {question}
    """
    quiz_prompt = ChatPromptTemplate.from_template(quiz_prompt_template)

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    # Prepare retrieval first
    pre = {
        "docs": itemgetter("question") | retriever,
        "question": itemgetter("question")
    }

    # Fallback quiz when no documents are retrieved (general knowledge)
    fallback_quiz_prompt_template = """
    You are an expert academic quiz generator. The knowledge base returned no context. Create a high-quality quiz using general knowledge.

    INSTRUCTIONS:
    1. Generate exactly 5 multiple-choice questions (MCQs) about the user's requested topic.
    2. Each question must have 4 options (A, B, C, D).
    3. Clearly indicate the correct answer for each question at the end of the response.
    4. Output format MUST be:
       - For each question: a new line starting with "Question {{n}}:"
       - Each option on its own line, like: "A. ...", "B. ..." etc
        - After all 5 questions, include a section starting with "Answers:" followed by "1) X" per line

    TOPIC REQUESTED BY USER: {question}
    """
    fallback_quiz_prompt = ChatPromptTemplate.from_template(fallback_quiz_prompt_template)
    fallback_quiz_chain = fallback_quiz_prompt | llm | StrOutputParser()

    # Quiz based on retrieved docs (with augmentation allowed by prompt)
    quiz_rag_chain = (
        {
            "context": itemgetter("docs") | RunnableLambda(format_docs),
            "question": itemgetter("question")
        }
        | quiz_prompt
        | llm
        | StrOutputParser()
    )

    branched_quiz = pre | RunnableBranch(
        (lambda x: len(x["docs"]) == 0, fallback_quiz_chain),
        quiz_rag_chain
    )

    return branched_quiz

class PastPaperSession:
    """Manages the state of past paper walkthroughs."""
    def __init__(self):
        self.current_paper = None
        self.questions_shown = 0
        self.total_questions = 0
        self.user_answers = {}
    
    def reset(self):
        """Reset the session."""
        self.__init__()
    
    def update(self, paper_id, questions_shown):
        """Update session state."""
        self.current_paper = paper_id
        self.questions_shown = questions_shown

past_paper_session = PastPaperSession()

# Master Application Chain 
if RAG_RETRIEVER:
    RAG_CHAIN = create_rag_chain(llm, RAG_RETRIEVER)
    QUIZ_CHAIN = create_quiz_chain(llm, RAG_RETRIEVER)
    # Past paper will be handled explicitly via enhanced handler.
    final_chain = RunnableBranch(
        (
            lambda x: "QUIZ" in router_chain.invoke({"question": x["question"]}).upper(),
            QUIZ_CHAIN
        ),
        RAG_CHAIN
    )
else:
    final_chain = None

def run_assistant():
    """Interactive assistant with past paper support."""
    if not final_chain:
        print("\nCannot run assistant. Please fix errors and re-run the indexing pipeline.")
        return
    
    print("\n=== Study Assistant Ready ===")
    print("You can:")
    print("- Ask questions about your study material")
    print("- Request a quiz on any topic")
    print("- Go through past papers (e.g., 'Let me go through CSC231 2024 past paper')")
    print("Type 'quit' to exit\n")
    
    # Track whether we're in a past paper session
    pastpaper_active = False

    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! Good luck with your studies!")
            break
        
        if not user_input:
            continue
        
        try:
            # If in an active past paper session, route to enhanced handler directly
            if pastpaper_active and enhanced_past_paper:
                response = enhanced_past_paper.handle_past_paper_request(user_input, session_id="cli")
                print(f"\nAssistant: {response}")
                # end session conditions
                if any(kw in response for kw in [
                    "Ending past paper session",
                    "completed all questions",
                    "final questions",
                ]):
                    pastpaper_active = False
                continue

            # Otherwise, use router to decide
            route = router_chain.invoke({"question": user_input}).upper()
            if "PASTPAPER" in route and enhanced_past_paper:
                response = enhanced_past_paper.handle_past_paper_request(user_input, session_id="cli")
                print(f"\nAssistant: {response}")
                pastpaper_active = True
            else:
                response = final_chain.invoke({"question": user_input})
                print(f"\nAssistant: {response}")
        
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try rephrasing your question.")

def run_demo():
    """Run demo queries to test all functionality."""
    if not final_chain:
        print("\nCannot run assistant. Please fix errors and re-run the indexing pipeline.")
        return

    # Q&A query
    qa_query = "What are the three types of cryptography?"
    print(f"\n--- Q&A QUERY: {qa_query} ---")
    response_qa = final_chain.invoke({"question": qa_query})
    print(response_qa) 

    # Quiz query
    quiz_query = "Give me a short practice quiz on Cryptography"
    print(f"\n--- QUIZ QUERY: {quiz_query} ---")
    response_quiz = final_chain.invoke({"question": quiz_query})
    print(response_quiz)
    
    # Past Paper query 
    past_paper_query = "I want to go through the CSC231 2024 past paper"
    print(f"\n--- PAST PAPER QUERY: {past_paper_query} ---")
    if enhanced_past_paper:
        response_pp = enhanced_past_paper.handle_past_paper_request(past_paper_query, session_id="demo")
        print(response_pp)
    else:
        print("Past paper flow unavailable: vector store not initialized.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo()
    else:

        run_assistant()
