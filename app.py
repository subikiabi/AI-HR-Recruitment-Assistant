"""AI HR Recruitment Assistant - Streamlit Web Dashboard (Agent + Tools + RAG)."""

import os
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

from src.parser import DocumentParser
from src.agent.recruiter_agent import RecruiterAgent
from src.utils.export_utils import ReportExporter
from src.config import STRONG_MATCH_THRESHOLD, MODERATE_MATCH_THRESHOLD

# Page Configuration
st.set_page_config(
    page_title="AI HR Recruitment Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .badge-blue { background-color: #E0F2FE; color: #0369A1; }
    .badge-green { background-color: #DCFCE7; color: #15803D; }
    .badge-orange { background-color: #FEF3C7; color: #B45309; }
    .badge-red { background-color: #FEE2E2; color: #B91C1C; }
    .card {
        border: 1px solid #E2E8F0;
        border-radius: 0.75rem;
        padding: 1.25rem;
        background-color: #FFFFFF;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .swot-box {
        border-radius: 0.5rem;
        padding: 1rem;
        min-height: 140px;
        margin-bottom: 0.8rem;
    }
    .swot-s { background-color: #F0FDF4; border-left: 4px solid #22C55E; }
    .swot-w { background-color: #FEF2F2; border-left: 4px solid #EF4444; }
    .swot-o { background-color: #EFF6FF; border-left: 4px solid #3B82F6; }
    .swot-t { background-color: #FFFBEB; border-left: 4px solid #F59E0B; }
</style>
""", unsafe_allow_html=True)


# Initialize Agent in Session State
if "agent" not in st.session_state:
    st.session_state.agent = RecruiterAgent()

if "active_jd_text" not in st.session_state:
    st.session_state.active_jd_text = ""

if "active_jd_title" not in st.session_state:
    st.session_state.active_jd_title = "Senior Backend Python Engineer"

agent: RecruiterAgent = st.session_state.agent


# Sidebar Configuration & Controls
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    # LLM Settings
    provider = st.selectbox(
        "AI Engine Mode",
        ["Offline / Demo Mode (No Key Needed)", "Google Gemini", "OpenAI"],
        help="The system runs completely offline with smart heuristic reasoning, or with live Gemini/OpenAI models if an API key is provided."
    )

    api_key = ""
    if provider == "Google Gemini":
        api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google AI Studio Gemini API Key")
        agent.gemini_key = api_key
    elif provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API Key")
        agent.openai_key = api_key

    st.markdown("---")
    st.markdown("### 📁 Quick Actions & Demo Data")
    
    # Load Demo Data Button
    if st.button("⚡ Load Demo Resumes & JDs", use_container_width=True, type="primary"):
        demo_resumes = [
            ("resume_senior_backend.txt", "sample_data/resumes/resume_senior_backend.txt", "cand_1"),
            ("resume_frontend_dev.txt", "sample_data/resumes/resume_frontend_dev.txt", "cand_2"),
            ("resume_data_engineer.txt", "sample_data/resumes/resume_data_engineer.txt", "cand_3"),
        ]
        demo_jd_path = "sample_data/job_descriptions/jd_senior_python_engineer.txt"
        
        # Reset and load
        agent.clear()
        
        for fname, path, cid in demo_resumes:
            if Path(path).exists():
                text = DocumentParser.extract_text_from_file(path)
                agent.ingest_resume(text, fname, candidate_id=cid)
        
        if Path(demo_jd_path).exists():
            jd_text = DocumentParser.extract_text_from_file(demo_jd_path)
            st.session_state.active_jd_text = jd_text
            parsed_jd = agent.set_active_job_description(jd_text, "active_jd")
            st.session_state.active_jd_title = parsed_jd.get("title", "Senior Backend Python Engineer")
            agent.evaluate_all_candidates("active_jd")

        st.success(f"Loaded {len(agent.candidates)} sample candidates and demo Job Description!")
        st.rerun()

    if st.button("🗑️ Clear All Data", use_container_width=True):
        agent.clear()
        st.session_state.active_jd_text = ""
        st.session_state.active_jd_title = ""
        st.info("System data cleared.")
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Indexing Status")
    st.metric("Ingested Candidates", len(agent.candidates))
    st.metric("RAG Vector Chunks", len(agent.vector_store.chunks))
    st.metric("Active Openings", len(agent.job_descriptions))


# Header
st.markdown('<div class="main-header">💼 AI HR Recruitment Assistant</div>', unsafe_allow_html=True)
st.markdown("""
<div class="sub-header">
    <span class="badge badge-blue">Architecture: Agent + Tools + RAG</span>
    <span class="badge badge-green">Resume Screening</span>
    <span class="badge badge-orange">Semantic JD Matching</span>
    <span class="badge badge-blue">Interview Kit Generator</span>
</div>
""", unsafe_allow_html=True)

# Main Application Tabs
tab_screen, tab_deepdive, tab_interview, tab_chat = st.tabs([
    "📊 Screening & Leaderboard",
    "🔍 Candidate Deep Dive & SWOT",
    "🎯 AI Interview Kit Generator",
    "💬 Recruiter AI Agent Chat"
])


# ==============================================================================
# TAB 1: SCREENING & LEADERBOARD
# ==============================================================================
with tab_screen:
    st.subheader("1. Job Description & Resume Ingestion")
    
    col_jd, col_upload = st.columns([1.2, 1])

    with col_jd:
        st.markdown("##### 📌 Target Job Description")
        sample_jd_options = [
            "Current Input / Custom",
            "Sample: Senior Backend Python Engineer",
            "Sample: Full Stack Software Engineer (React & Python)"
        ]
        selected_jd_opt = st.selectbox("Select Sample JD or paste your own:", sample_jd_options)
        
        if selected_jd_opt == "Sample: Senior Backend Python Engineer":
            jd_path = Path("sample_data/job_descriptions/jd_senior_python_engineer.txt")
            if jd_path.exists():
                st.session_state.active_jd_text = DocumentParser.extract_text_from_file(jd_path)
        elif selected_jd_opt == "Sample: Full Stack Software Engineer (React & Python)":
            jd_path = Path("sample_data/job_descriptions/jd_fullstack_react_python.txt")
            if jd_path.exists():
                st.session_state.active_jd_text = DocumentParser.extract_text_from_file(jd_path)

        jd_input_text = st.text_area(
            "Job Description Text",
            value=st.session_state.active_jd_text,
            height=200,
            placeholder="Paste your Job Description (role title, required skills, responsibilities)..."
        )
        st.session_state.active_jd_text = jd_input_text

    with col_upload:
        st.markdown("##### 📄 Ingest Candidate Resumes")
        uploaded_resumes = st.file_uploader(
            "Upload Resumes (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Upload one or multiple resumes to extract skills and evaluate against the JD."
        )

        if uploaded_resumes:
            with st.spinner("Parsing documents and chunking into RAG vector store..."):
                for uploaded_file in uploaded_resumes:
                    bytes_data = uploaded_file.read()
                    parsed_text = DocumentParser.extract_text_from_bytes(bytes_data, uploaded_file.name)
                    agent.ingest_resume(parsed_text, uploaded_file.name)
            st.success(f"Ingested {len(uploaded_resumes)} resume(s) successfully!")

    # Evaluation Trigger
    st.markdown("---")
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        run_screening = st.button("⚡ Run AI Screening & Matching", type="primary", use_container_width=True)
    with col_info:
        st.caption(f"Currently managing **{len(agent.candidates)} candidate(s)**. Click button to match against the active JD.")

    if run_screening or (agent.candidates and st.session_state.active_jd_text and not agent.match_results):
        if not st.session_state.active_jd_text.strip():
            st.error("Please enter or select a Job Description first.")
        elif not agent.candidates:
            st.warning("Please upload candidate resumes or click '⚡ Load Demo Resumes & JDs' in the sidebar.")
        else:
            with st.spinner("Agent invoking tools: JD Parser, RAG Matcher, and Candidate Comparer..."):
                parsed_jd = agent.set_active_job_description(st.session_state.active_jd_text, "active_jd")
                st.session_state.active_jd_title = parsed_jd.get("title", "Target Role")
                eval_results = agent.evaluate_all_candidates("active_jd")
            st.success(f"Evaluated {len(eval_results)} candidate(s) against '{st.session_state.active_jd_title}'!")

    # Display Leaderboard
    if agent.match_results:
        st.markdown("### 🏆 Candidate Match Leaderboard")
        comparison = agent.compare_candidates("active_jd")
        leaderboard = comparison.get("leaderboard", [])

        # Leaderboard Bar Chart
        if leaderboard:
            cand_names = [item["candidate_name"] for item in leaderboard][::-1]
            scores = [item["overall_fit_score"] for item in leaderboard][::-1]
            colors = [
                "#22C55E" if s >= STRONG_MATCH_THRESHOLD else ("#F59E0B" if s >= MODERATE_MATCH_THRESHOLD else "#EF4444")
                for s in scores
            ]

            fig = go.Figure(go.Bar(
                x=scores,
                y=cand_names,
                orientation='h',
                marker_color=colors,
                text=[f"{s}%" for s in scores],
                textposition='auto',
            ))
            fig.update_layout(
                title="Overall Candidate Fit Score Comparison",
                xaxis=dict(range=[0, 100], title="Fit Score (%)"),
                yaxis=dict(title="Candidate"),
                height=260,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Candidate Cards
        for item in leaderboard:
            badge_class = "badge-green" if item["overall_fit_score"] >= STRONG_MATCH_THRESHOLD else ("badge-orange" if item["overall_fit_score"] >= MODERATE_MATCH_THRESHOLD else "badge-red")
            
            with st.expander(f"Rank #{item['rank']} | {item['candidate_name']} — {item['overall_fit_score']}% Match ({item['fit_category']})", expanded=(item['rank'] == 1)):
                col1, col2, col3 = st.columns([1, 1.2, 1.2])
                with col1:
                    st.metric("Overall Fit Score", f"{item['overall_fit_score']}%")
                    st.write(f"**Experience:** ~{item['years_experience']} years")
                    st.write(f"**Seniority:** {item['seniority_level']}")
                    st.write(f"**Recommendation:** *{item['recommendation']}*")

                with col2:
                    st.write("**Matched Required Skills:**")
                    if item["matched_skills"]:
                        st.markdown(" ".join([f"`{s}`" for s in item["matched_skills"]]))
                    else:
                        st.write("None matched.")

                with col3:
                    st.write("**Missing / Gap Skills:**")
                    if item["missing_skills"]:
                        st.markdown(" ".join([f"<span class='badge badge-red'>{s}</span>" for s in item["missing_skills"]]), unsafe_allow_html=True)
                    else:
                        st.markdown("<span class='badge badge-green'>All core requirements met!</span>", unsafe_allow_html=True)
    else:
        st.info("💡 No evaluations to display. Click **'⚡ Load Demo Resumes & JDs'** in the sidebar to try it immediately!")


# ==============================================================================
# TAB 2: CANDIDATE DEEP DIVE & SWOT
# ==============================================================================
with tab_deepdive:
    st.subheader("2. Candidate Deep Dive & SWOT Analysis")
    
    if not agent.candidates:
        st.info("Please load or upload candidate resumes first.")
    else:
        cand_choice = st.selectbox(
            "Select Candidate to Inspect:",
            options=list(agent.candidates.keys()),
            format_func=lambda cid: f"{agent.candidates[cid].get('name')} ({agent.candidates[cid].get('seniority_level')})"
        )

        profile = agent.candidates[cand_choice]
        match_key = f"{cand_choice}_active_jd"
        match_data = agent.match_results.get(match_key)

        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            st.markdown(f"### {profile.get('name')}")
            st.write(f"**Profile Summary:** {profile.get('summary')}")
            
            c_info = profile.get("contact", {})
            st.write(f"📧 **Email:** {c_info.get('email', 'N/A')} | 📞 **Phone:** {c_info.get('phone', 'N/A')} | 📍 **Location:** {c_info.get('location', 'N/A')}")
            
            st.markdown("##### 🎓 Education & Certifications")
            for edu in profile.get("education", []):
                st.write(f"- {edu}")
            for cert in profile.get("certifications", []):
                st.write(f"- 🏅 {cert}")

            st.markdown("##### 🛠️ Extracted Technical Skills")
            for cat, skills in profile.get("skills", {}).get("categorized", {}).items():
                st.write(f"**{cat.replace('_', ' ').title()}:** {', '.join(skills)}")

        with col_right:
            if match_data:
                st.markdown("### 🎯 Job Fit Overview")
                st.metric("Fit Score", f"{match_data.get('overall_fit_score')}%", match_data.get("fit_category"))
                st.write(f"**Role:** {match_data.get('role_title')}")
                st.write(f"**Experience Check:** {match_data.get('experience_assessment')}")
                st.write(f"**Action Recommendation:** {match_data.get('recommendation')}")
                
                # Gauge Chart
                score = match_data.get("overall_fit_score", 0)
                gauge_color = "#22C55E" if score >= STRONG_MATCH_THRESHOLD else ("#F59E0B" if score >= MODERATE_MATCH_THRESHOLD else "#EF4444")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': gauge_color}},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ))
                fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                st.warning("Candidate has not been evaluated against the active Job Description yet.")

        # SWOT Matrix Display
        if match_data and match_data.get("swot_analysis"):
            st.markdown("---")
            st.markdown("### 🧩 SWOT Strategic Fit Matrix")
            swot = match_data["swot_analysis"]

            col_s, col_w = st.columns(2)
            with col_s:
                st.markdown("""<div class="swot-box swot-s"><b>🟢 STRENGTHS</b><br>""" +
                            "<br>".join([f"• {item}" for item in swot.get("strengths", [])]) +
                            "</div>", unsafe_allow_html=True)
            with col_w:
                st.markdown("""<div class="swot-box swot-w"><b>🔴 WEAKNESSES / GAPS</b><br>""" +
                            "<br>".join([f"• {item}" for item in swot.get("weaknesses", [])]) +
                            "</div>", unsafe_allow_html=True)

            col_o, col_t = st.columns(2)
            with col_o:
                st.markdown("""<div class="swot-box swot-o"><b>🟡 OPPORTUNITIES</b><br>""" +
                            "<br>".join([f"• {item}" for item in swot.get("opportunities", [])]) +
                            "</div>", unsafe_allow_html=True)
            with col_t:
                st.markdown("""<div class="swot-box swot-t"><b>🔵 THREATS / RISKS</b><br>""" +
                            "<br>".join([f"• {item}" for item in swot.get("threats", [])]) +
                            "</div>", unsafe_allow_html=True)

        # RAG Chunk Inspector
        with st.expander("🔎 View Stored RAG Vector Chunks for this Candidate"):
            cand_chunks = [c for c in agent.vector_store.chunks if c.metadata.get("candidate_id") == cand_choice]
            st.write(f"Total chunks indexed: **{len(cand_chunks)}**")
            for c in cand_chunks:
                st.caption(f"**Chunk ID:** `{c.chunk_id}` | **Section:** `{c.metadata.get('section')}`")
                st.code(c.text, language="text")


# ==============================================================================
# TAB 3: AI INTERVIEW KIT GENERATOR
# ==============================================================================
with tab_interview:
    st.subheader("3. Tailored Interview Kit Generator")
    
    if not agent.candidates or not st.session_state.active_jd_text:
        st.info("Please ingest resumes and establish a Job Description in Tab 1 first.")
    else:
        col_sel, col_action = st.columns([2, 1])
        with col_sel:
            interview_cand_id = st.selectbox(
                "Select Candidate for Interview Kit Generation:",
                options=list(agent.candidates.keys()),
                format_func=lambda cid: agent.candidates[cid].get("name", cid),
                key="interview_cand_sel"
            )
        with col_action:
            gen_btn = st.button("⚡ Generate Interview Kit", type="primary", use_container_width=True)

        if gen_btn or f"kit_{interview_cand_id}" in st.session_state:
            if gen_btn or f"kit_{interview_cand_id}" not in st.session_state:
                with st.spinner("AI Tool synthesizing customized interview kit with rubrics..."):
                    kit_data = agent.generate_interview_kit(interview_cand_id, "active_jd")
                    st.session_state[f"kit_{interview_cand_id}"] = kit_data
            else:
                kit_data = st.session_state[f"kit_{interview_cand_id}"]

            st.markdown(f"#### 📋 Interview Questionnaire: {kit_data.get('candidate_name')} ({kit_data.get('target_role')})")
            
            # Export / Download Button
            markdown_content = ReportExporter.format_interview_kit_markdown(kit_data)
            st.download_button(
                label="📥 Download Interview Kit (.md)",
                data=markdown_content,
                file_name=f"interview_kit_{kit_data.get('candidate_name', 'candidate').replace(' ', '_').lower()}.md",
                mime="text/markdown"
            )

            # Section 1: Technical Questions
            st.markdown("### 1. 💻 Technical Architecture & Depth Questions")
            for i, q in enumerate(kit_data.get("technical_questions", []), 1):
                with st.container():
                    st.markdown(f"**Q{i}: {q.get('category')}**")
                    st.info(f"❓ {q.get('question')}")
                    if q.get("context"):
                        st.caption(f"**Why ask this:** {q.get('context')}")
                    rubric = q.get("rubric", {})
                    if rubric:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.success(f"**✅ Expected Ideal Answer:**\n{rubric.get('ideal_answer')}")
                        with c2:
                            st.error(f"**⚠️ Watch Outs / Red Flags:**\n{rubric.get('watch_outs')}")
                    st.markdown("---")

            # Section 2: Gap & Probe Questions
            st.markdown("### 2. 🔍 Skill Gap & Probe Questions")
            gap_qs = kit_data.get("gap_questions", [])
            if gap_qs:
                for i, q in enumerate(gap_qs, 1):
                    with st.container():
                        st.markdown(f"**Probe {i}: {q.get('category')}**")
                        st.warning(f"❓ {q.get('question')}")
                        rubric = q.get("rubric", {})
                        if rubric:
                            c1, c2 = st.columns(2)
                            with c1:
                                st.success(f"**✅ Expected Answer:**\n{rubric.get('ideal_answer')}")
                            with c2:
                                st.error(f"**⚠️ Watch Outs:**\n{rubric.get('watch_outs')}")
                        st.markdown("---")
            else:
                st.write("No critical skill gaps identified.")

            # Section 3: Behavioral Questions
            st.markdown("### 3. 🤝 Behavioral & Leadership (STAR Method)")
            for i, q in enumerate(kit_data.get("behavioral_questions", []), 1):
                with st.container():
                    st.markdown(f"**Behavioral Q{i}: {q.get('category')}**")
                    st.write(f"❓ {q.get('question')}")
                    rubric = q.get("rubric", {})
                    if rubric:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.success(f"**✅ Positive Indicators:**\n{rubric.get('ideal_answer')}")
                        with c2:
                            st.error(f"**⚠️ Concerning Indicators:**\n{rubric.get('watch_outs')}")
                    st.markdown("---")

            # Section 4: Scenario Challenge
            scenario = kit_data.get("scenario_challenge", {})
            if scenario:
                st.markdown("### 4. ⚡ Real-World Architecture Scenario")
                st.markdown(f"**{scenario.get('title')}**")
                st.markdown(f"> {scenario.get('scenario')}")
                st.write("**Key Evaluation Dimensions:**")
                for dim in scenario.get("key_evaluation_dimensions", []):
                    st.write(f"- {dim}")


# ==============================================================================
# TAB 4: RECRUITER AI AGENT CHAT
# ==============================================================================
with tab_chat:
    st.subheader("4. Conversational Recruiter Assistant")
    st.caption("Ask questions about candidates, compare profiles, explore technical weaknesses, or verify resume claims using Agent + Tools + RAG.")

    # Suggested Prompts
    st.markdown("**Quick Prompts:**")
    prompt_cols = st.columns(3)
    with prompt_cols[0]:
        if st.button("🏆 Compare candidates for this role", use_container_width=True):
            st.session_state.pending_query = "Compare all candidates for this role and tell me who to interview first."
    with prompt_cols[1]:
        if st.button("⚠️ Analyze candidate gaps & red flags", use_container_width=True):
            st.session_state.pending_query = "What are the key technical weaknesses or red flags across the candidates?"
    with prompt_cols[2]:
        if st.button("🎯 Interview advice for top candidate", use_container_width=True):
            st.session_state.pending_query = "What specific questions should I ask our top candidate during the architecture round?"

    # Chat Messages History
    for msg in agent.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input
    user_query = st.chat_input("Ask the AI Recruiter Agent (e.g. 'Compare Alex and Priya for the backend role')...")
    
    if "pending_query" in st.session_state and st.session_state.pending_query:
        user_query = st.session_state.pending_query
        st.session_state.pending_query = None

    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Agent reasoning over RAG evidence and invoking tools..."):
                response = agent.ask_recruiter_agent(user_query, "active_jd")
                st.markdown(response["answer"])

                # Expandable Tool Execution Traces
                with st.expander("🛠️ View Agent Tool Traces & RAG Sources", expanded=False):
                    for trace in response.get("tool_traces", []):
                        st.markdown(f"**Tool:** `{trace.get('tool')}`")
                        st.caption(trace.get("action"))
                        if trace.get("chunks"):
                            for ch in trace["chunks"]:
                                st.text(f"• {ch}")
