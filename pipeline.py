"""
AI Career & Interview Coach - Full Pipeline
Draft used only to validate logic/syntax before splitting into notebook cells.
"""

# ---------------------------------------------------------------------------
# 1. Imports & Setup
# ---------------------------------------------------------------------------
import os
import json
import requests
from typing import List, Optional
from pydantic import BaseModel, Field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Self-hosted LLM server (Mistral/Llama running on Kaggle/Colab, exposed via
# ngrok). No external AI-provider API and no API key from any company here -
# LLM_SERVER_URL / LLM_API_KEY point at a server you control.
# ---------------------------------------------------------------------------
LLM_SERVER_URL = os.environ.get("LLM_SERVER_URL", "").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")


def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 3000) -> str:
    """Thin wrapper around the self-hosted Mistral/Llama server used by every
    chain in this notebook. Talks to the /generate endpoint from the
    model-server notebook (FastAPI + ngrok) instead of a paid AI API."""
    if not LLM_SERVER_URL:
        raise RuntimeError(
            "LLM_SERVER_URL is not set. Paste the ngrok URL printed by the "
            "model-server notebook into the sidebar (or set the env var)."
        )

    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    response = requests.post(
        f"{LLM_SERVER_URL}/generate",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json={"prompt": full_prompt, "max_length": max_tokens},
        timeout=180,  # CPU/free-GPU inference can be slow
    )
    response.raise_for_status()
    return response.json()["response"]


def parse_json_response(raw_text: str) -> dict:
    """Strip markdown code fences (if any) and parse the model's JSON output.

    Open models (Mistral/Llama) sometimes add extra text after the JSON
    object/array even when told not to - e.g. a trailing note or a repeated
    fragment. json.loads() rejects that ("Extra data" error), so instead we
    decode only the first valid JSON value and ignore anything after it."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        cleaned = cleaned.replace("json", "", 1).strip() if cleaned.lower().startswith("json") else cleaned
    cleaned = cleaned.strip()

    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(cleaned)
    return obj


# ---------------------------------------------------------------------------
# 2. Sample Knowledge Base (stand-in for a production RAG source)
# ---------------------------------------------------------------------------
JOB_DESCRIPTIONS = [
    {
        "job_id": "JD001",
        "title": "Machine Learning Engineer",
        "required_skills": ["Python", "TensorFlow/PyTorch", "SQL", "MLOps", "Statistics", "Docker"],
        "description": "Build, train and deploy ML models in production. Own the ML lifecycle from data to deployment.",
    },
    {
        "job_id": "JD002",
        "title": "Backend Developer (Python)",
        "required_skills": ["Python", "FastAPI/Django", "SQL", "REST APIs", "Docker", "System Design"],
        "description": "Design and build scalable backend services and APIs for web applications.",
    },
    {
        "job_id": "JD003",
        "title": "Data Analyst",
        "required_skills": ["SQL", "Excel", "Python", "Data Visualization", "Statistics", "Business Acumen"],
        "description": "Turn raw data into actionable business insights through analysis and dashboards.",
    },
    {
        "job_id": "JD004",
        "title": "AI/LLM Engineer",
        "required_skills": ["Python", "LangChain", "Prompt Engineering", "Vector Databases", "RAG", "APIs"],
        "description": "Build applications powered by large language models, including RAG pipelines and agents.",
    },
]

LEARNING_RESOURCES = [
    {"skill": "Docker", "resource": "Docker for Developers - hands-on containerization course", "level": "Beginner"},
    {"skill": "SQL", "resource": "Advanced SQL for Data Analysis", "level": "Intermediate"},
    {"skill": "System Design", "resource": "Grokking the System Design Interview", "level": "Intermediate"},
    {"skill": "MLOps", "resource": "MLOps Zoomcamp - end-to-end ML deployment", "level": "Advanced"},
    {"skill": "LangChain", "resource": "LangChain & LangGraph official documentation + tutorials", "level": "Intermediate"},
    {"skill": "Prompt Engineering", "resource": "Anthropic's Prompt Engineering Guide", "level": "Beginner"},
    {"skill": "Vector Databases", "resource": "Chroma / Pinecone quickstart guides", "level": "Intermediate"},
    {"skill": "Statistics", "resource": "Statistics for Data Science and Business Analysis", "level": "Beginner"},
]

INTERVIEW_QUESTIONS_BANK = [
    {"job_title": "AI/LLM Engineer", "type": "technical", "question": "How would you reduce hallucinations in a RAG pipeline?"},
    {"job_title": "AI/LLM Engineer", "type": "technical", "question": "Explain the tradeoffs between fine-tuning and RAG."},
    {"job_title": "AI/LLM Engineer", "type": "technical", "question": "How do you choose a chunking strategy for a vector database?"},
    {"job_title": "AI/LLM Engineer", "type": "behavioral", "question": "Tell me about a time you had to debug an unpredictable model output."},
    {"job_title": "Machine Learning Engineer", "type": "technical", "question": "How do you handle class imbalance in a classification problem?"},
    {"job_title": "Machine Learning Engineer", "type": "technical", "question": "Walk me through how you would deploy a model to production."},
    {"job_title": "Machine Learning Engineer", "type": "behavioral", "question": "Describe a project where your model underperformed in production. What did you do?"},
    {"job_title": "Backend Developer (Python)", "type": "technical", "question": "How would you design a rate limiter for a public API?"},
    {"job_title": "Backend Developer (Python)", "type": "behavioral", "question": "Tell me about a time you had to refactor legacy code under a deadline."},
    {"job_title": "Data Analyst", "type": "technical", "question": "How would you detect and handle outliers in a sales dataset?"},
]


# ---------------------------------------------------------------------------
# 3. Lightweight TF-IDF Retriever (stand-in for a vector DB)
# ---------------------------------------------------------------------------
class SimpleRetriever:
    """A minimal TF-IDF based retriever. Swap this for Chroma + real embeddings
    in production; the interface (retrieve) is what the rest of the pipeline
    depends on, so the swap is transparent to every chain below."""

    def __init__(self, documents: List[dict], text_key: str):
        self.documents = documents
        self.texts = [d[text_key] if isinstance(d[text_key], str) else " ".join(d[text_key]) for d in documents]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [self.documents[i] for i in top_indices]


jobs_retriever = SimpleRetriever(JOB_DESCRIPTIONS, text_key="required_skills")
resources_retriever = SimpleRetriever(LEARNING_RESOURCES, text_key="skill")
questions_retriever = SimpleRetriever(INTERVIEW_QUESTIONS_BANK, text_key="question")


# ---------------------------------------------------------------------------
# 4. Pydantic Output Schemas
# ---------------------------------------------------------------------------
class JobMatch(BaseModel):
    job_title: str
    match_score: int = Field(description="0-100 fit score")
    reason: str


class SkillGapAnalysis(BaseModel):
    current_skills: List[str]
    missing_skills: List[str]
    matched_jobs: List[JobMatch]


class RoadmapStep(BaseModel):
    skill: str
    resource: str
    priority: str  # High / Medium / Low


class QuizItem(BaseModel):
    question: str
    options: List[str]
    correct_answer: str


class PrepMaterial(BaseModel):
    topic: str
    summary: str
    sample_question: str
    sample_answer_tip: str
    quiz: List[QuizItem]


class InterviewTurn(BaseModel):
    question: str
    answer: str
    score: float
    missing_points: List[str]
    improvement_tip: str


class FinalReport(BaseModel):
    overall_readiness_score: float
    strengths: List[str]
    weaknesses: List[str]
    next_steps: List[str]


# ---------------------------------------------------------------------------
# 5. Chain 1 - CV Analysis & Gap Analysis
# ---------------------------------------------------------------------------
def chain_cv_analysis(cv_text: str, target_role: Optional[str] = None) -> SkillGapAnalysis:
    retrieved_jobs = jobs_retriever.retrieve(cv_text if not target_role else target_role, top_k=3)

    system_prompt = """You are an expert career analyst. You extract skills from a CV,
compare them against retrieved job descriptions, and return ONLY valid JSON matching
this schema, with no preamble and no markdown fences:

{
  "current_skills": [string],
  "missing_skills": [string],
  "matched_jobs": [{"job_title": string, "match_score": int, "reason": string}]
}"""

    user_prompt = f"""CV:
{cv_text}

Target role (optional): {target_role or "not specified"}

Candidate job descriptions retrieved from the database:
{json.dumps(retrieved_jobs, indent=2)}

Analyze the CV and return the JSON described in the system prompt."""

    raw = call_llm(system_prompt, user_prompt)
    return SkillGapAnalysis(**parse_json_response(raw))


# ---------------------------------------------------------------------------
# 6. Chain 2 - Roadmap Generation
# ---------------------------------------------------------------------------
def chain_roadmap(gap_analysis: SkillGapAnalysis) -> List[RoadmapStep]:
    retrieved_resources = []
    for skill in gap_analysis.missing_skills:
        retrieved_resources.extend(resources_retriever.retrieve(skill, top_k=1))

    system_prompt = """You are a learning-path designer. Given missing skills and
candidate learning resources, output ONLY a valid JSON list, no prose, no fences:

[{"skill": string, "resource": string, "priority": "High"|"Medium"|"Low"}]"""

    user_prompt = f"""Missing skills: {gap_analysis.missing_skills}

Retrieved resources:
{json.dumps(retrieved_resources, indent=2)}

Order the roadmap by priority (skills needed for the highest-scoring matched job first)."""

    raw = call_llm(system_prompt, user_prompt)
    return [RoadmapStep(**item) for item in parse_json_response(raw)]


# ---------------------------------------------------------------------------
# 7. Chain 3 - Interview Prep Module (Study Pack + Quiz)
# ---------------------------------------------------------------------------
def chain_interview_prep(job_title: str, gap_analysis: SkillGapAnalysis) -> List[PrepMaterial]:
    focus_topics = gap_analysis.missing_skills[:3] if gap_analysis.missing_skills else [job_title]
    prep_materials = []

    system_prompt = """You are an interview coach preparing a candidate before their
mock interview. For the given topic, output ONLY valid JSON matching this schema,
no prose, no fences:

{
  "topic": string,
  "summary": string,
  "sample_question": string,
  "sample_answer_tip": string,
  "quiz": [{"question": string, "options": [string, string, string, string], "correct_answer": string}]
}
Include exactly 2 quiz items."""

    for topic in focus_topics:
        user_prompt = f"Target role: {job_title}\nTopic to prepare: {topic}"
        raw = call_llm(system_prompt, user_prompt)
        prep_materials.append(PrepMaterial(**parse_json_response(raw)))

    return prep_materials


def compute_readiness_score(prep_materials: List[PrepMaterial], quiz_answers: dict) -> float:
    """quiz_answers: {question: chosen_answer}. Returns 0-100 readiness score."""
    total, correct = 0, 0
    for material in prep_materials:
        for item in material.quiz:
            total += 1
            if quiz_answers.get(item.question) == item.correct_answer:
                correct += 1
    return round((correct / total) * 100, 1) if total else 0.0


# ---------------------------------------------------------------------------
# 8. Chain 4 - Interview Simulator (Question Generation + Evaluation)
# ---------------------------------------------------------------------------
class InterviewSimulator:
    """Holds conversation memory across the mock interview so questions can
    build on previous answers, similar to a real interviewer."""

    def __init__(self, job_title: str, gap_analysis: SkillGapAnalysis, num_questions: int = 4):
        self.job_title = job_title
        self.gap_analysis = gap_analysis
        self.num_questions = num_questions
        self.history: List[InterviewTurn] = []

    def next_question(self) -> str:
        retrieved = questions_retriever.retrieve(self.job_title, top_k=3)
        asked = [t.question for t in self.history]

        # Adaptive difficulty: after 2 strong answers, push harder / gap-focused questions
        avg_score = sum(t.score for t in self.history) / len(self.history) if self.history else 5
        focus_hint = (
            f"Prioritize a question that probes one of these weak areas: {self.gap_analysis.missing_skills}"
            if avg_score >= 7
            else "Ask a solid foundational question for this role."
        )

        system_prompt = """You are an interviewer conducting a mock interview.
Return ONLY the next interview question as plain text, no JSON, no preamble."""

        user_prompt = f"""Role: {self.job_title}
Already asked: {asked}
Candidate questions bank (retrieved): {json.dumps(retrieved, indent=2)}
{focus_hint}
Avoid repeating any question already asked."""

        return call_llm(system_prompt, user_prompt, max_tokens=200).strip()

    def evaluate_answer(self, question: str, answer: str) -> InterviewTurn:
        system_prompt = """You are grading a mock interview answer. Output ONLY
valid JSON, no prose, no fences:

{"score": float (0-10), "missing_points": [string], "improvement_tip": string}"""

        user_prompt = f"Role: {self.job_title}\nQuestion: {question}\nCandidate answer: {answer}"
        raw = call_llm(system_prompt, user_prompt)
        parsed = parse_json_response(raw)
        turn = InterviewTurn(question=question, answer=answer, **parsed)
        self.history.append(turn)
        return turn

    def is_complete(self) -> bool:
        return len(self.history) >= self.num_questions


# ---------------------------------------------------------------------------
# 9. Chain 5 - Final Report
# ---------------------------------------------------------------------------
def chain_final_report(gap_analysis: SkillGapAnalysis, interview_history: List[InterviewTurn]) -> FinalReport:
    system_prompt = """You are summarizing a candidate's full career-coaching session
(CV analysis + mock interview). Output ONLY valid JSON, no prose, no fences:

{
  "overall_readiness_score": float (0-100),
  "strengths": [string],
  "weaknesses": [string],
  "next_steps": [string]
}"""

    user_prompt = f"""Gap analysis: {gap_analysis.model_dump_json()}

Interview history:
{json.dumps([t.model_dump() for t in interview_history], indent=2)}"""

    raw = call_llm(system_prompt, user_prompt)
    return FinalReport(**parse_json_response(raw))


# ---------------------------------------------------------------------------
# 10. Orchestrator - runs the whole pipeline end-to-end
# ---------------------------------------------------------------------------
def run_full_pipeline(cv_text: str, target_role: str, get_user_answer):
    """
    get_user_answer: callable(question: str) -> str
    Lets you plug in real input() in a notebook, or Streamlit's text input,
    without changing any pipeline logic.
    """
    print("Step 1/5: Analyzing CV...")
    gap_analysis = chain_cv_analysis(cv_text, target_role)

    print("Step 2/5: Building roadmap...")
    roadmap = chain_roadmap(gap_analysis)

    print("Step 3/5: Preparing study material...")
    prep_materials = chain_interview_prep(target_role, gap_analysis)

    print("Step 4/5: Running mock interview...")
    simulator = InterviewSimulator(target_role, gap_analysis)
    while not simulator.is_complete():
        question = simulator.next_question()
        answer = get_user_answer(question)
        simulator.evaluate_answer(question, answer)

    print("Step 5/5: Generating final report...")
    report = chain_final_report(gap_analysis, simulator.history)

    return {
        "gap_analysis": gap_analysis,
        "roadmap": roadmap,
        "prep_materials": prep_materials,
        "interview_history": simulator.history,
        "final_report": report,
    }
