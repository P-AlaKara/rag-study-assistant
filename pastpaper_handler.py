import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

@dataclass
class PastPaperSession:
    """Manages the state of a past paper walkthrough session."""
    unit_code: Optional[str] = None
    year: Optional[str] = None
    current_batch: int = 0
    questions: List[str] = field(default_factory=list)
    user_answers: Dict[int, str] = field(default_factory=dict)
    model_answers: Dict[int, str] = field(default_factory=dict)
    is_active: bool = False
    total_questions: int = 0
    
    def reset(self):
        """Reset the session to initial state."""
        self.__init__()
    
    def start_paper(self, unit_code: str, year: str, questions: List[str]):
        """Initialize a new past paper session."""
        self.unit_code = unit_code
        self.year = year
        self.questions = questions
        self.total_questions = len(questions)
        self.current_batch = 0
        self.is_active = True
        self.user_answers = {}
        self.model_answers = {}
    
    def get_next_batch(self, batch_size: int = 5) -> Tuple[List[str], bool]:
        """Get the next batch of questions."""
        start_idx = self.current_batch * batch_size
        end_idx = min(start_idx + batch_size, self.total_questions)
        
        batch = self.questions[start_idx:end_idx]
        has_more = end_idx < self.total_questions
        
        self.current_batch += 1
        
        return batch, has_more
    
    def get_current_progress(self) -> str:
        """Get a string representing current progress."""
        questions_shown = min(self.current_batch * 5, self.total_questions)
        return f"Questions {questions_shown}/{self.total_questions}"
    
    def save_answer(self, question_num: int, answer: str):
        """Save a user's answer for a question."""
        self.user_answers[question_num] = answer


class PastPaperProcessor:
    """Processes and manages past paper content."""
    
    @staticmethod
    def extract_questions(documents: List[Any]) -> List[str]:
        """Extract individual questions from retrieved documents.

        Robustly segments content into question blocks using common numbering styles
        and tolerating leading indentation/whitespace:
        - "Question 1:", "Q1:", "1.", "1)", "(1)"
        Falls back to paragraph splitting if no markers are found.
        """
        all_content = "\n".join([doc.page_content for doc in documents])

        # Capture a question number and lazily consume until the next question marker
        # Improvements over previous version:
        # - Allow leading whitespace before markers (common in OCR/PDF text)
        # - Support numbers wrapped in parentheses: (1) ...
        # - Maintain the ability to match: Question 1:, Q1:, 1., 1)
        # The pattern captures the question number in either group 1 (parenthesized) or
        # group 2 (plain), and the text content in group 3.
        question_pattern = re.compile(
            (
                r"(?ms)"  # multiline + dotall
                r"^\s*(?:Q(?:uestion)?\s*)?(?:\((\d+)\)|(\d+)[\.:)])\s+"  # marker
                r"(.*?)"  # question text (lazy)
                r"(?=^\s*(?:Q(?:uestion)?\s*)?(?:\(\d+\)|\d+[\.:)])\s+|\Z)"  # next marker or end
            )
        )

        matches = list(question_pattern.finditer(all_content))
        questions: List[str] = []
        if matches:
            # Sort by numeric question number to enforce order
            extracted = []
            for m in matches:
                num_str = m.group(1) or m.group(2)
                q_text = m.group(3).strip()
                if q_text and len(q_text) > 3:
                    extracted.append((int(num_str), q_text))
            sorted_matches = sorted(extracted, key=lambda x: x[0])
            for q_num, q_text in sorted_matches:
                questions.append(f"Question {q_num}: {q_text}")

        # If no numbered questions found, split by double newlines as a fallback
        if not questions:
            sections = [s.strip() for s in re.split(r"\n\n+", all_content) if s.strip()]
            questions = [f"Question {i + 1}: {section}" for i, section in enumerate(sections) if len(section) > 20]

        return questions
    
    @staticmethod
    def format_batch(questions: List[str], start_num: int = 1, answers: Optional[Dict[int, str]] = None) -> str:
        """Format a batch of questions for display with clean separation and readability."""
        formatted_blocks: List[str] = []
        for offset, raw in enumerate(questions):
            q_index = start_num + offset
            cleaned = re.sub(rf"^\s*Question\s+{q_index}:\s*", "", raw).strip()
            block = f"**Question {q_index}:**\n{cleaned}\n"
            if answers and q_index in answers:
                block += f"\n**Answer:**\n{answers[q_index]}\n"
            formatted_blocks.append(block)

        return ("\n" + ("-" * 50) + "\n").join(formatted_blocks)
    
    @staticmethod
    def parse_user_intent(user_input: str) -> Dict[str, any]:
        """Parse user input to understand their intent regarding past papers."""
        intent = {
            "wants_next": False,
            "wants_clarification": False,
            "has_answer": False,
            "wants_stop": False,
            "question_num": None,
            "answer_text": None
        }
        
        # Check for continuation
        if re.search(r'\b(next|continue|more|yes|proceed)\b', user_input.lower()):
            intent["wants_next"] = True
        
        # Check for clarification request
        if re.search(r'\b(clarify|explain|help|confused|understand)\b', user_input.lower()):
            intent["wants_clarification"] = True
            # Try to extract question number
            num_match = re.search(r'question\s*(\d+)', user_input.lower())
            if num_match:
                intent["question_num"] = int(num_match.group(1))
        
        # Check for answer attempt
        if re.search(r'\b(answer|my answer|i think|solution)\b', user_input.lower()):
            intent["has_answer"] = True
            # Extract question number and answer
            num_match = re.search(r'question\s*(\d+)', user_input.lower())
            if num_match:
                intent["question_num"] = int(num_match.group(1))
            # The answer text would be everything after certain keywords
            answer_match = re.search(r'(?:answer is|my answer|i think it\'s|solution:)\s*(.+)', 
                                    user_input, re.IGNORECASE)
            if answer_match:
                intent["answer_text"] = answer_match.group(1).strip()
        
        # Check for stop request
        if re.search(r'\b(stop|quit|exit|done|finish)\b', user_input.lower()):
            intent["wants_stop"] = True
        
        return intent


class EnhancedPastPaperChain:
    """Enhanced chain for handling past paper interactions."""
    
    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
        self.sessions = {}  # Store sessions per user/conversation
        self.processor = PastPaperProcessor()
    
    def get_or_create_session(self, session_id: str = "default") -> PastPaperSession:
        """Get existing session or create new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = PastPaperSession()
        return self.sessions[session_id]
    
    def handle_past_paper_request(self, user_input: str, session_id: str = "default") -> str:
        """Main handler for past paper requests."""
        session = self.get_or_create_session(session_id)
        
        # If starting a new past paper request
        if not session.is_active or self._is_new_paper_request(user_input):
            return self._start_new_paper(user_input, session)
        
        # Otherwise, handle ongoing session
        intent = self.processor.parse_user_intent(user_input)
        
        if intent["wants_stop"]:
            session.reset()
            return "Ending past paper session. Good luck with your studies! Feel free to ask any questions or start another paper."
        
        if intent["wants_next"]:
            return self._show_next_batch(session)
        
        if intent["wants_clarification"]:
            return self._provide_clarification(intent, session)
        
        if intent["has_answer"]:
            return self._process_answer(intent, session)
        
        # Default: assume they want the next batch
        return self._show_next_batch(session)
    
    def _is_new_paper_request(self, user_input: str) -> bool:
        """Check if this is a request for a new past paper."""
        patterns = [
            r'(start|begin|go through|work through|practice).*(past paper|exam|test)',
            r'(CSC|MAT|PHY|[A-Z]{3})\d{3}.*\d{4}',  # Unit code patterns
            r'\d{4}.*(past paper|exam)',  # Year patterns
        ]
        
        for pattern in patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return True
        return False
    
    def _start_new_paper(self, user_input: str, session: PastPaperSession) -> str:
        """Start a new past paper session."""
        # Extract unit code and year
        unit_code, year = self._extract_paper_details(user_input)
        
        # Retrieve past paper documents using Chroma's expected where syntax
        base_conditions = [{"source_type": {"$eq": "PastPaper"}}]
        if unit_code:
            base_conditions.append({"unit_code": {"$eq": unit_code}})
        if year:
            base_conditions.append({"year": {"$eq": str(year)}})

        strict_filter = {"$and": base_conditions}

        # Try strict filter first; if empty, relax unit_code filter due to filename metadata variations
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 20, "filter": strict_filter})
        docs = retriever.get_relevant_documents(user_input)
        if not docs and unit_code:
            relaxed_conditions = [c for c in base_conditions if "unit_code" not in c]
            relaxed_filter = {"$and": relaxed_conditions} if relaxed_conditions else {}
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 20, "filter": relaxed_filter})
            docs = retriever.get_relevant_documents(user_input)
        
        if not docs:
            return f"I couldn't find a past paper matching your criteria (Unit: {unit_code or 'Any'}, Year: {year or 'Any'}). Please check the unit code and year, or try being more specific."
        
        # Extract questions from documents
        questions = self.processor.extract_questions(docs)
        
        if not questions:
            return "I found the past paper but couldn't extract the questions properly. The document might be in an unexpected format."
        
        # Start the session
        session.start_paper(unit_code or "Unknown", year or "Unknown", questions)
        
        # Show first batch (questions only; answers on demand)
        batch, has_more = session.get_next_batch()

        response = f"**Starting {session.unit_code} ({session.year}) Past Paper**\n"
        response += f"Total questions: {session.total_questions}\n"
        response += "-" * 50 + "\n"
        # Render questions only (no model answers shown by default)
        response += self.processor.format_batch(batch, 1)
        response += "\n" + "-" * 50 + "\n"
        
        if has_more:
            response += "\n📝 **What would you like to do?**\n"
            response += "• Type 'next' or 'continue' to see the next 5 questions\n"
            response += "• Answer a question (e.g., 'My answer for question 1 is...')\n"
            response += "• Ask for clarification (e.g., 'Can you explain question 3?')\n"
            response += "• Type 'stop' to end the session"
        else:
            response += "\n✅ **That's all the questions!**\n"
            response += "Feel free to attempt any question or ask for help to answer a question."
        
        return response
    
    def _show_next_batch(self, session: PastPaperSession) -> str:
        """Show the next batch of questions."""
        if not session.is_active:
            return "No active past paper session. Would you like to start one? Just tell me which paper you'd like to go through."
        
        batch, has_more = session.get_next_batch()
        
        if not batch:
            session.reset()
            return "You've completed all questions in this past paper! Great job! 🎉\nWould you like to review any answers or start another paper?"
        
        start_num = (session.current_batch - 1) * 5 + 1

        response = f"**Continuing {session.unit_code} ({session.year}) - {session.get_current_progress()}**\n"
        response += "-" * 50 + "\n"
        # Render questions only (no model answers shown by default)
        response += self.processor.format_batch(batch, start_num)
        response += "\n" + "-" * 50 + "\n"
        
        if has_more:
            response += "\n📝 Ready for more? Type 'next' to continue, or work on these questions first."
        else:
            response += "\n✅ **These are the final questions!**"
        
        return response
    
    def _provide_clarification(self, intent: Dict, session: PastPaperSession) -> str:
        """Provide a full clarification and worked solution for a specific question using model knowledge (no retrieval)."""
        if not session.is_active:
            return "No active past paper session. Please start one first."
        
        q_num = intent.get("question_num")
        if not q_num or q_num > len(session.questions):
            return "Please specify a valid question number for clarification."
        
        question = session.questions[q_num - 1]
		
        # Lazy import to avoid requiring langchain during smoke tests
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        prompt = ChatPromptTemplate.from_template(
			"""
			You are a patient tutor. Provide a clear, step-by-step explanation and a full worked solution for the following past paper question using your subject knowledge.
			- Explain key concept(s) succinctly
			- Break down the approach and show the solution
			- Highlight common pitfalls or misconceptions
			- Be precise and exam-ready

			QUESTION:
			{question}

			EXPLANATION AND FULL SOLUTION:
			"""
		)

        chain = prompt | self.llm | StrOutputParser()
        clarification = chain.invoke({"question": question})

        response = f"**Clarification and Solution for Question {q_num}:**\n\n{clarification}\n\n💡 Would you like to attempt this question now?"
        return response
    
    def _process_answer(self, intent: Dict, session: PastPaperSession) -> str:
        """Process and provide feedback on a user's answer using model knowledge (no retrieval)."""
        if not session.is_active:
            return "No active past paper session. Please start one first."
        
        q_num = intent.get("question_num")
        answer = intent.get("answer_text")
        
        if not q_num or not answer:
            return "Please provide both the question number and your answer."
        
        # Save the answer
        session.save_answer(q_num, answer)

        question_text = session.questions[q_num - 1]

        # Lazy import to avoid requiring langchain during smoke tests
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        eval_prompt = ChatPromptTemplate.from_template(
            """
			You are grading a student's short answer using your subject knowledge. Be fair, constructive, and concise.
            Provide:
            1) A brief verdict (Correct, Partially correct, Incorrect)
            2) Key points they got right or missed
            3) A short model answer (2-4 sentences)
            4) One suggestion for improvement

            QUESTION:
            {question}

            STUDENT ANSWER:
            {answer}

            FEEDBACK:
            """
        )

        chain = eval_prompt | self.llm | StrOutputParser()
        feedback = chain.invoke({"question": question_text, "answer": answer})

        response = (
            f"**Your answer for Question {q_num}:**\n{answer}\n\n"
            f"**Feedback:**\n{feedback}\n\n"
            "✅ Answer recorded. Would you like to continue with more questions?"
        )
        return response
    
    def _extract_paper_details(self, user_input: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract unit code and year from user input."""
        # Extract unit code (e.g., CSC231, MAT101)
        unit_match = re.search(r'\b([A-Z]{3}\d{3})\b', user_input.upper())
        unit_code = unit_match.group(1) if unit_match else None
        
        # Extract year (4-digit number between 2000-2030)
        year_match = re.search(r'\b(20[0-3]\d)\b', user_input)
        year = year_match.group(1) if year_match else None
        
        return unit_code, year

    def _generate_answers_for_batch(self, questions: List[str], start_index: int, session: PastPaperSession) -> None:
        """Generate concise model answers for a batch using model knowledge and store them in session.model_answers."""
        # Lazy import to avoid requiring langchain during smoke tests
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        prompt = ChatPromptTemplate.from_template(
			"""
			You are an exam tutor. Provide a concise, correct, exam-ready model answer using your subject knowledge.
			- Keep it focused: 3-6 bullet points or 4-8 sentences.
			- If multiple valid approaches exist, pick one and note alternatives briefly.
			- Be precise and avoid hedging.
			
			QUESTION:
			{question}
			
			MODEL ANSWER:
			"""
		)
        chain = prompt | self.llm | StrOutputParser()

        for offset, q in enumerate(questions):
            q_index = start_index + offset
			# Skip if already generated (e.g., user revisits)
            if q_index in session.model_answers:
                continue
            try:
                answer = chain.invoke({"question": q})
            except Exception:
                answer = "(Unable to generate an answer at this time.)"
            session.model_answers[q_index] = answer