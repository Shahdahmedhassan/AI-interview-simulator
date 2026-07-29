"""
AI Career & Interview Coach - Streamlit app.
Reuses every chain defined in pipeline.py (same logic as the notebook) -
this file is only the UI / state-management layer.
"""

import os
import streamlit as st

st.set_page_config(page_title="AI Career & Interview Coach", page_icon="🎯", layout="centered")

# ---------------------------------------------------------------------------
# API key (from Streamlit secrets in production, or manual input for local dev)
# ---------------------------------------------------------------------------
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
elif not os.environ.get("GEMINI_API_KEY"):
    key = st.sidebar.text_input("Gemini API Key", type="password")
    if key:
        os.environ["GEMINI_API_KEY"] = key

import pipeline_gemini as pl  # noqa: E402  (import after key is set)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "stage": "input",           # input -> analysis -> prep -> interview -> report
    "gap_analysis": None,
    "roadmap": None,
    "prep_materials": None,
    "simulator": None,
    "target_role": "",
    "final_report": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("🎯 AI Career & Interview Coach")

if not os.environ.get("GEMINI_API_KEY"):
    st.warning("Enter your Gemini API key in the sidebar to start.")
    st.stop()

# ---------------------------------------------------------------------------
# Stage 1: Input
# ---------------------------------------------------------------------------
if st.session_state.stage == "input":
    st.subheader("1. Tell us about you")
    cv_text = st.text_area("Paste your CV text", height=220,
                            placeholder="Name, skills, experience, education...")
    target_role = st.text_input("Target role", placeholder="e.g. AI/LLM Engineer")

    if st.button("Analyze my CV", type="primary", disabled=not (cv_text and target_role)):
        with st.spinner("Analyzing your CV against the job market..."):
            st.session_state.gap_analysis = pl.chain_cv_analysis(cv_text, target_role)
            st.session_state.roadmap = pl.chain_roadmap(st.session_state.gap_analysis)
            st.session_state.target_role = target_role
        st.session_state.stage = "analysis"
        st.rerun()

# ---------------------------------------------------------------------------
# Stage 2: Show analysis + roadmap
# ---------------------------------------------------------------------------
elif st.session_state.stage == "analysis":
    ga = st.session_state.gap_analysis
    st.subheader("2. Your Skill Gap Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✅ Current Skills**")
        for s in ga.current_skills:
            st.markdown(f"- {s}")
    with col2:
        st.markdown("**⚠️ Missing Skills**")
        for s in ga.missing_skills:
            st.markdown(f"- {s}")

    st.markdown("**Matched Jobs**")
    for job in ga.matched_jobs:
        st.progress(job.match_score / 100, text=f"{job.job_title} — {job.match_score}%")
        st.caption(job.reason)

    st.markdown("**📚 Recommended Roadmap**")
    for step in st.session_state.roadmap:
        st.markdown(f"- **[{step.priority}]** {step.skill} → {step.resource}")

    if st.button("Continue to interview prep", type="primary"):
        with st.spinner("Preparing your study pack..."):
            st.session_state.prep_materials = pl.chain_interview_prep(
                st.session_state.target_role, ga
            )
        st.session_state.stage = "prep"
        st.rerun()

# ---------------------------------------------------------------------------
# Stage 3: Prep material + quiz
# ---------------------------------------------------------------------------
elif st.session_state.stage == "prep":
    st.subheader("3. Get ready before the mock interview")
    quiz_answers = {}

    for material in st.session_state.prep_materials:
        with st.expander(f"📖 {material.topic}", expanded=True):
            st.write(material.summary)
            st.markdown(f"**Sample question:** {material.sample_question}")
            st.markdown(f"**Tip:** {material.sample_answer_tip}")
            for i, item in enumerate(material.quiz):
                choice = st.radio(item.question, item.options, key=f"{material.topic}_{i}", index=None)
                quiz_answers[item.question] = choice

    if st.button("I'm ready — start the mock interview", type="primary"):
        readiness = pl.compute_readiness_score(st.session_state.prep_materials, quiz_answers)
        st.session_state.readiness = readiness
        st.session_state.simulator = pl.InterviewSimulator(
            st.session_state.target_role, st.session_state.gap_analysis
        )
        st.session_state.stage = "interview"
        st.rerun()

# ---------------------------------------------------------------------------
# Stage 4: Mock interview (adaptive, memory-based)
# ---------------------------------------------------------------------------
elif st.session_state.stage == "interview":
    sim = st.session_state.simulator
    st.subheader("4. Mock Interview")
    if "readiness" in st.session_state:
        st.caption(f"Quiz readiness score: {st.session_state.readiness}%")

    for turn in sim.history:
        with st.chat_message("assistant"):
            st.write(turn.question)
        with st.chat_message("user"):
            st.write(turn.answer)
        st.caption(f"Score: {turn.score}/10 — {turn.improvement_tip}")

    if not sim.is_complete():
        if "current_question" not in st.session_state:
            with st.spinner("Thinking of your next question..."):
                st.session_state.current_question = sim.next_question()

        with st.chat_message("assistant"):
            st.write(st.session_state.current_question)

        answer = st.text_area("Your answer", key="answer_box")
        if st.button("Submit answer", type="primary", disabled=not answer):
            with st.spinner("Evaluating..."):
                sim.evaluate_answer(st.session_state.current_question, answer)
            del st.session_state.current_question
            st.rerun()
    else:
        if st.button("Generate final report", type="primary"):
            with st.spinner("Wrapping up your session..."):
                st.session_state.final_report = pl.chain_final_report(
                    st.session_state.gap_analysis, sim.history
                )
            st.session_state.stage = "report"
            st.rerun()

# ---------------------------------------------------------------------------
# Stage 5: Final report
# ---------------------------------------------------------------------------
elif st.session_state.stage == "report":
    report = st.session_state.final_report
    st.subheader("5. Final Report")
    st.metric("Overall Readiness Score", f"{report.overall_readiness_score}/100")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**💪 Strengths**")
        for s in report.strengths:
            st.markdown(f"- {s}")
    with col2:
        st.markdown("**🎯 Areas to improve**")
        for w in report.weaknesses:
            st.markdown(f"- {w}")

    st.markdown("**Next steps**")
    for step in report.next_steps:
        st.markdown(f"- {step}")

    if st.button("Start a new session"):
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.rerun()
