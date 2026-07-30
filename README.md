# 🚀 [Tips Hindawi](https://www.tipshindawi.com/) Challenge (June–July) 2026

> 🏆 This repository is my official submission for the [ **Tips Hindawi** ](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.

## 👤 Participant

| Field            | Value                                |
| ---------------- | ------------------------------------ |
| Full Name        | *(Shahd Ahmed Hassan)*                    |
| Project Name     | AI Career & Interview Coach          |
| GitHub Username  | *(Shahdahmedhassan)*         |
| Challenge Batch  | June–July 2026                       |
| Training Program | Large Language Models (LLMs) Program |
| Organization     | [**Edrak for Ai**](https://edrak4ai.com/en) |

---

# 📖 Project Overview

**AI Career & Interview Coach** is an end-to-end Streamlit application that helps a job seeker go from "here's my CV" to "I'm ready for the interview" in one guided flow. The user pastes their CV and a target role, and the app:

1. Analyzes the CV against a set of job descriptions to find skill gaps.
2. Builds a prioritized learning roadmap for the missing skills.
3. Generates topic-based study material with quizzes to check readiness.
4. Runs an adaptive mock interview (questions get harder / gap-focused as the candidate does well).
5. Produces a final report with an overall readiness score, strengths, weaknesses, and next steps.

All the AI-powered logic is defined once in `pipeline.py` as a set of reusable "chains" (functions), which both a Jupyter notebook and the `app.py` Streamlit UI call into — so the notebook and the app always share the exact same logic.

Instead of a paid third-party AI API, the project talks to a **self-hosted open-source LLM** (Mistral-7B-Instruct, 4-bit quantized) running for free on a Kaggle/Colab GPU notebook, wrapped in a small FastAPI server and exposed publicly through an ngrok tunnel. The Streamlit app simply calls that URL.

---

# ✨ Features

* **CV → Skill Gap Analysis** — extracts current skills from a pasted CV and compares them against a retrieved set of job descriptions (TF-IDF based retriever), returning current skills, missing skills, and a ranked list of matched jobs with fit scores and reasons.
* **Personalized Learning Roadmap** — turns missing skills into a prioritized (High/Medium/Low) list of learning resources.
* **Interview Prep Study Pack** — for each weak topic, generates a summary, a sample question, an answer tip, and a short quiz; computes a quiz-based readiness score before the interview starts.
* **Adaptive Mock Interview** — an `InterviewSimulator` with memory of the conversation; question difficulty adapts based on the candidate's running average score, and each answer is scored (0–10) with missing points and an improvement tip.
* **Final Readiness Report** — summarizes the whole session (CV analysis + interview) into an overall readiness score (0–100), strengths, weaknesses, and concrete next steps.
* **Self-hosted LLM, no paid API key** — runs Mistral-7B-Instruct on a free Kaggle/Colab GPU via a FastAPI + ngrok bridge, keeping the whole pipeline free to run.
* **Shared pipeline logic** — one `pipeline.py` module used identically by the notebook (for experimentation) and the Streamlit app (for the UI), avoiding logic duplication.

---

# 🛠️ Technologies Used

* **Python 3.12**
* **Streamlit** — the web UI / multi-stage session flow
* **FastAPI + Uvicorn** — lightweight inference server for the LLM
* **pyngrok** — exposes the local model server with a public HTTPS URL
* **Hugging Face Transformers + Accelerate + BitsAndBytes** — loading and running **Mistral-7B-Instruct-v0.2** in 4-bit quantization on GPU
* **PyTorch** — model inference backend
* **scikit-learn (TF-IDF + cosine similarity)** — lightweight retriever standing in for a vector database, used to fetch relevant job descriptions, learning resources, and interview questions
* **Pydantic** — structured, validated schemas for every LLM output (skill gap analysis, roadmap steps, quiz items, interview turns, final report)
* **Jupyter Notebook** — hosts and runs the model server on Kaggle/Colab's free GPU

---

# ⚙️ Installation

### 1. Start the model server (Kaggle or Colab, GPU)

1. Open `final-project-tips-hindawi.ipynb` on Kaggle (or Colab).
2. Enable a free **GPU** accelerator (e.g. T4) and turn **Internet** on.
3. Create a free account at [ngrok.com](https://ngrok.com) and copy your Authtoken into the notebook.
4. Run all cells top to bottom. This loads Mistral-7B-Instruct in 4-bit, starts a FastAPI server on port 8000, and opens an ngrok tunnel.
5. Copy the printed **public ngrok URL** and the **API key** — you'll need both for the app.
6. Keep this notebook running the entire time you want the app to work.

### 2. Run the Streamlit app locally

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install streamlit requests pydantic scikit-learn
streamlit run app.py
```

On first launch, paste the ngrok URL and the server key into the sidebar (or set them as Streamlit secrets: `LLM_SERVER_URL`, `LLM_API_KEY`).

---

# 🚀 Usage

1. **Input** — paste your CV text and your target role, then click *Analyze my CV*.
2. **Analysis** — review your current vs. missing skills, matched jobs, and the generated learning roadmap.
3. **Prep** — go through the study material for your weakest topics and take the short quizzes to get a readiness score.
4. **Mock Interview** — answer a series of adaptive interview questions; each answer is scored with feedback.
5. **Final Report** — get your overall readiness score, strengths, weaknesses, and next steps. Start a new session anytime.

---

# 📸 Demo

*(Add screenshots, GIFs, or a demo video here.)*

---

# 📈 Results

*(Share your project's outcomes or achievements here — e.g. readiness score improvements across test sessions, feedback quality, etc.)*

---

# 🔮 Future Improvements

* Swap the TF-IDF retriever for a real vector database (e.g. Chroma / Pinecone) with proper embeddings.
* Support uploading a CV file (PDF/DOCX) instead of pasting raw text.
* Persist session history so users can track readiness improvement over multiple sessions.
* Add support for larger/better open models or a configurable model choice.
* Deploy the model server on a more persistent, always-on free/low-cost host instead of a notebook session.

---

# 📚 About the Challenge

This project was developed as part of the [**Tips Hindawi**](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.

[Tips Hindawi](https://www.tipshindawi.com/) is the internships department of [**Edrak for Ai**](https://edrak4ai.com/en), and the challenge encourages participants to build real-world projects, apply practical skills, and showcase their work through GitHub.

For more information about the challenge, training programs, and upcoming batches, visit the official [Tips Hindawi](https://www.tipshindawi.com/) website.

---

# 📄 License

This project is shared for educational and portfolio purposes.
