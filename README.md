# AI HR Recruitment Assistant

> **Architecture: Agent + Tools + RAG**  
> Screens resumes, matches candidates with job descriptions, and generates interview questions with objective evaluation rubrics.

---

## Overview

The **AI HR Recruitment Assistant** is an intelligent talent acquisition platform designed for modern recruiting teams. It automates candidate screening, performs high-precision semantic matching against Job Descriptions (JDs), highlights critical skill gaps and resume red-flags, and produces tailored interview kits with scoring rubrics.

---

## System Architecture

```
                                  +-------------------------------------+
                                  |        Streamlit Web Dashboard       |
                                  |  (Screening | Deep-Dive | Chatbot)  |
                                  +-------------------------------------+
                                                     |
                                                     v
                                  +-------------------------------------+
                                  |            HR Recruiter Agent       |
                                  |    (Reasoning, Planning, Synthesis) |
                                  +-------------------------------------+
                                                     |
                    +-----------------+--------------+-----------------+-----------------+
                    |                 |                                |                 |
                    v                 v                                v                 v
            +---------------+ +---------------+                +---------------+ +---------------+
            |  Resume RAG   | |   JD Matcher  |                | Interview Gen | | Comparer Tool |
            | Tool (Search  | | Tool (Scores, |                | Tool (Technical| | (Multi-person |
            |  & Retrieve)  | |  Gaps, SWOT)  |                |  & Behavioral)| |  Leaderboard) |
            +---------------+ +---------------+                +---------------+ +---------------+
                    |
                    v
       +-------------------------+
       | Vector Store & Ingestion|
       | (PDF/DOCX/TXT Parser,   |
       | Chunker, Embeddings)    |
       +-------------------------+
```

### 1. RAG (Retrieval-Augmented Generation) Pipeline
- **Multi-Format Parsing**: Extracts text cleanly from PDF, DOCX, and TXT documents.
- **Section-Aware Chunking**: Detects headers (`Summary`, `Experience`, `Skills`, `Education`, `Projects`, `Certifications`) and assigns structured metadata.
- **Hybrid Search Vector Store**: Combines cosine similarity over semantic vector representations and lexical term weighting for accurate resume retrieval.

### 2. Specialized HR Agent Tools
- **`ResumeScreenerTool`**: Extracts candidate contact information, estimated years of experience, seniority level, 40+ categorized technical skills, and detects potential red flags.
- **`JDMatcherTool`**: Calculates an objective weighted match score (0-100%), identifies missing required vs preferred skills, and generates a 4-quadrant SWOT analysis (Strengths, Weaknesses, Opportunities, Threats).
- **`InterviewQuestionGeneratorTool`**: Produces technical in-depth questions based on claimed projects, skill gap probe questions, STAR behavioral questions, and scoring rubrics (ideal answers vs watch-outs).
- **`CandidateComparerTool`**: Ranks candidates into a leaderboard with actionable hiring recommendations (`Strong Match`, `Moderate Match`, `Low Match`).

### 3. Autonomous Recruiter Agent
- Multi-step reasoning loop coordinating tool execution and RAG grounding.
- Answers complex conversational questions with full tool execution transparency.
- Runs **100% offline out-of-the-box** using smart heuristic synthesizers, and seamlessly connects to **Google Gemini** or **OpenAI** when an API key is provided.

---

## Getting Started

### 1. Installation
Ensure Python 3.10+ is installed, then run:

```bash
python -m pip install -r requirements.txt
```

### 2. Configuration (Optional)
If you wish to use live Google Gemini or OpenAI models, copy `.env.example` to `.env` and set your key:

```bash
cp .env.example .env
```
*(Or enter your key directly in the web dashboard sidebar).*

### 3. Launch Application
Start the Streamlit web dashboard:

```bash
python -m streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## ⚡ Quick Demo Walkthrough

1. Open the sidebar and click **`⚡ Load Demo Resumes & JDs`**.
2. Go to **Tab 1: 📊 Screening & Leaderboard** to see the candidate match leaderboard and bar chart.
3. Switch to **Tab 2: 🔍 Candidate Deep Dive & SWOT** to inspect Alex Morgan or Priya Sharma, view their fit score gauge, SWOT matrix, and RAG vector chunks.
4. Navigate to **Tab 3: 🎯 AI Interview Kit Generator** to generate and export an interview guide with scoring rubrics to Markdown.
5. In **Tab 4: 💬 Recruiter AI Agent Chat**, try asking:
   - *"Compare the candidates for this role and tell me who to interview first."*
   - *"What are the main skill gaps for Priya Sharma regarding the backend role?"*
   - *"What questions should I ask Alex Morgan about microservices and Kafka?"*

---

## Running Tests

Run the automated backend test suite:

```bash
python -m tests.test_recruitment_assistant
```
