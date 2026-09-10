"""Autonomous Recruiter Agent coordinating RAG, Tools, and LLM reasoning."""

import re
import json
from typing import Dict, Any, List, Optional
import requests

from src.config import (
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
)
from src.rag.chunker import ResumeChunker
from src.rag.vector_store import InMemoVectorStore
from src.tools.resume_screener import ResumeScreenerTool
from src.tools.jd_matcher import JDMatcherTool
from src.tools.question_generator import InterviewQuestionGeneratorTool
from src.tools.candidate_comparer import CandidateComparerTool
from src.agent.prompts import RECRUITER_SYSTEM_PROMPT


class RecruiterAgent:
    """Agent that handles resume screening, JD matching, interview kit generation, and chat."""

    def __init__(
        self,
        gemini_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.gemini_key = gemini_key or GEMINI_API_KEY
        self.openai_key = openai_key or OPENAI_API_KEY
        self.model_name = model_name or (DEFAULT_GEMINI_MODEL if self.gemini_key else DEFAULT_OPENAI_MODEL)

        # Core RAG Vector Store
        self.vector_store = InMemoVectorStore()

        # Tools
        self.screener_tool = ResumeScreenerTool()
        self.matcher_tool = JDMatcherTool(vector_store=self.vector_store)
        self.question_tool = InterviewQuestionGeneratorTool()
        self.comparer_tool = CandidateComparerTool()

        # In-memory candidate & JD registries
        self.candidates: Dict[str, Dict[str, Any]] = {}
        self.raw_resumes: Dict[str, str] = {}
        self.job_descriptions: Dict[str, Dict[str, Any]] = {}
        self.match_results: Dict[str, Dict[str, Any]] = {} # keyed by f"{cand_id}_{jd_id}"
        self.chat_history: List[Dict[str, str]] = []

    def clear(self):
        """Reset agent state and RAG store."""
        self.vector_store.clear()
        self.candidates.clear()
        self.raw_resumes.clear()
        self.job_descriptions.clear()
        self.match_results.clear()
        self.chat_history.clear()

    def ingest_resume(self, text: str, filename: str, candidate_id: Optional[str] = None) -> Dict[str, Any]:
        """Ingest a resume into RAG store and screen the candidate."""
        if not candidate_id:
            candidate_id = f"cand_{len(self.candidates) + 1}"

        # 1. RAG Chunking & Vector Store Indexing
        chunks = ResumeChunker.chunk_resume(text, source_filename=filename, candidate_id=candidate_id)
        self.vector_store.add_documents(chunks)

        # 2. Tool: Screen Resume
        profile = self.screener_tool.execute(text, candidate_id=candidate_id)
        profile["filename"] = filename
        
        self.candidates[candidate_id] = profile
        self.raw_resumes[candidate_id] = text

        return profile

    def set_active_job_description(self, jd_text: str, jd_id: str = "active_jd") -> Dict[str, Any]:
        """Parse and store active Job Description, and index into RAG."""
        jd_data = self.matcher_tool.parse_job_description(jd_text)
        jd_data["jd_id"] = jd_id

        # Index JD chunks
        jd_chunks = ResumeChunker.chunk_job_description(jd_text, jd_id=jd_id, title=jd_data.get("title", ""))
        self.vector_store.add_documents(jd_chunks)

        self.job_descriptions[jd_id] = jd_data
        return jd_data

    def evaluate_candidate_against_jd(self, candidate_id: str, jd_id: str = "active_jd") -> Dict[str, Any]:
        """Evaluate a single candidate against a specific Job Description."""
        profile = self.candidates.get(candidate_id)
        jd_data = self.job_descriptions.get(jd_id)

        if not profile or not jd_data:
            raise ValueError(f"Candidate ({candidate_id}) or JD ({jd_id}) not found.")

        match_result = self.matcher_tool.execute(profile, jd_data)
        self.match_results[f"{candidate_id}_{jd_id}"] = match_result
        return match_result

    def evaluate_all_candidates(self, jd_id: str = "active_jd") -> List[Dict[str, Any]]:
        """Evaluate all ingested candidates against the active Job Description."""
        results = []
        for cand_id in self.candidates:
            res = self.evaluate_candidate_against_jd(cand_id, jd_id)
            results.append(res)
        return results

    def generate_interview_kit(self, candidate_id: str, jd_id: str = "active_jd") -> Dict[str, Any]:
        """Generates tailored interview questions and scoring rubrics."""
        profile = self.candidates.get(candidate_id)
        jd_data = self.job_descriptions.get(jd_id)
        match_key = f"{candidate_id}_{jd_id}"
        match_result = self.match_results.get(match_key)

        if not match_result and profile and jd_data:
            match_result = self.evaluate_candidate_against_jd(candidate_id, jd_id)

        if not profile or not jd_data:
            raise ValueError("Candidate profile or JD data missing.")

        return self.question_tool.execute(profile, match_result, jd_data)

    def compare_candidates(self, jd_id: str = "active_jd") -> Dict[str, Any]:
        """Generate a comparative analysis and leaderboard across all evaluated candidates."""
        relevant_matches = [
            m for k, m in self.match_results.items() if k.endswith(f"_{jd_id}")
        ]
        return self.comparer_tool.execute(relevant_matches, self.candidates)

    def ask_recruiter_agent(self, user_query: str, jd_id: str = "active_jd") -> Dict[str, Any]:
        """Conversational agent that uses RAG retrieval and tool synthesis to answer user queries."""
        tool_traces = []

        # Step 1: Detect intent and choose tools
        query_lower = user_query.lower()
        
        # Check if user asks for comparison or ranking
        if any(w in query_lower for w in ["compare", "rank", "leaderboard", "who is better", "best candidate"]):
            tool_traces.append({"tool": "candidate_comparer", "action": "Ranked candidates by fit score and calculated tradeoffs."})
            comp_data = self.compare_candidates(jd_id)
        else:
            comp_data = None

        # Step 2: RAG Vector Retrieval
        rag_results = self.vector_store.search(user_query, top_k=4)
        tool_traces.append({
            "tool": "resume_rag_retriever",
            "action": f"Retrieved top {len(rag_results)} relevant evidence chunks from vector store.",
            "chunks": [r["text"][:120] + "..." for r in rag_results]
        })

        # Step 3: Build Grounded Context
        context_parts = []
        active_jd = self.job_descriptions.get(jd_id, {})
        if active_jd:
            context_parts.append(f"Target Role: {active_jd.get('title')}\nRequired Skills: {', '.join(active_jd.get('required_skills', []))}")

        if comp_data and comp_data.get("leaderboard"):
            context_parts.append("Candidate Leaderboard:")
            for item in comp_data["leaderboard"]:
                context_parts.append(
                    f"- {item['candidate_name']}: Fit Score {item['overall_fit_score']}%, "
                    f"Match % {item['required_match_pct']}%, Experience: {item['years_experience']} yrs. "
                    f"Recommendation: {item['recommendation']}"
                )

        if rag_results:
            context_parts.append("Retrieved Resume Evidence:")
            for i, chunk in enumerate(rag_results, 1):
                c_name = chunk["metadata"].get("candidate_name", "Candidate")
                sec = chunk["metadata"].get("section", "resume")
                context_parts.append(f"[Source {i} - {c_name} - {sec}]:\n{chunk['text']}")

        grounded_context = "\n\n".join(context_parts)

        # Step 4: Generate Answer via LLM or Local Agent Synthesis
        answer = self._generate_response(user_query, grounded_context)

        # Record in history
        self.chat_history.append({"role": "user", "content": user_query})
        self.chat_history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "tool_traces": tool_traces,
            "rag_sources": rag_results
        }

    def _generate_response(self, user_query: str, context: str) -> str:
        """Invokes Gemini, OpenAI, or Fallback Synthesizer."""
        # 1. Try Gemini API
        if self.gemini_key:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.gemini_key}"
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": f"{RECRUITER_SYSTEM_PROMPT}\n\nContext Information:\n{context}\n\nUser Question: {user_query}"
                        }]
                    }],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024}
                }
                resp = requests.post(url, json=payload, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates_out = data.get("candidates", [])
                    if candidates_out:
                        return candidates_out[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            except Exception:
                pass

        # 2. Try OpenAI API
        if self.openai_key:
            try:
                url = "https://api.openai.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
                payload = {
                    "model": DEFAULT_OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": RECRUITER_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
                    ],
                    "temperature": 0.2
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
            except Exception:
                pass

        # 3. Built-in Heuristic / Offline Synthesizer
        return self._fallback_synthesis(user_query, context)

    def _fallback_synthesis(self, query: str, context: str) -> str:
        """Deterministic, grounded synthesis when no external API key is active."""
        q_lower = query.lower()
        lines = []

        if any(w in q_lower for w in ["compare", "rank", "who is better", "best"]):
            lines.append("### 🏆 Candidate Evaluation & Comparison Summary\n")
            if self.match_results:
                sorted_cands = sorted(self.match_results.values(), key=lambda x: x.get("overall_fit_score", 0), reverse=True)
                top = sorted_cands[0]
                lines.append(f"**Top Ranked Candidate**: **{top.get('candidate_name')}** with an overall fit score of **{top.get('overall_fit_score')}%** ({top.get('fit_category')}).\n")
                lines.append("**Key Strengths & Trade-offs:**")
                for c in sorted_cands:
                    matched_s = c.get("required_skills_match", {}).get("matched", [])
                    missing_s = c.get("required_skills_match", {}).get("missing", [])
                    lines.append(f"- **{c.get('candidate_name')}** ({c.get('overall_fit_score')}% fit):")
                    lines.append(f"  - Matched Core Skills: {', '.join(matched_s[:5]) if matched_s else 'None'}")
                    if missing_s:
                        lines.append(f"  - Gaps / Missing: {', '.join(missing_s[:3])}")
                    lines.append(f"  - Recommendation: *{c.get('recommendation')}*")
            else:
                lines.append("No candidates have been matched against the active job description yet. Please screen candidates in the Screening tab.")
        
        elif any(w in q_lower for w in ["weakness", "gap", "missing", "red flag"]):
            lines.append("### ⚠️ Candidate Gaps & Red-Flag Analysis\n")
            found = False
            for cand_id, prof in self.candidates.items():
                if prof.get("name", "").lower() in q_lower or not found:
                    found = True
                    lines.append(f"**Candidate: {prof.get('name')}**")
                    if prof.get("red_flags"):
                        for rf in prof.get("red_flags", []):
                            lines.append(f"- 🚩 {rf}")
                    else:
                        lines.append("- No severe red flags detected.")
                    
                    # Check match gaps
                    for match_key, match in self.match_results.items():
                        if match.get("candidate_id") == cand_id:
                            missing = match.get("required_skills_match", {}).get("missing", [])
                            if missing:
                                lines.append(f"- Missing Required Skills for target role: **{', '.join(missing)}**")
            if not found:
                lines.append("Please specify a candidate name or ingest resumes to evaluate skill gaps.")

        elif any(w in q_lower for w in ["question", "interview", "ask"]):
            lines.append("### 🎯 Recommended Interview Focus Areas\n")
            lines.append("Based on the candidate resumes and job requirements:")
            lines.append("1. **System Scalability & Concurrency**: Ask candidate to detail how their architecture handles high-throughput API endpoints and concurrency locks.")
            lines.append("2. **Technical Gaps**: Probe candidates on any required stack technologies not explicitly proven on their resume.")
            lines.append("3. **Incident Recovery & Leadership**: Use STAR scenario questions to test production debugging under pressure.")
            lines.append("\n*Tip: Visit the **Interview Kit Generator** tab to export a complete, role-specific questionnaire with evaluation rubrics.*")

        else:
            lines.append(f"### 📋 Recruitment Assistant Response\n")
            lines.append(f"**Query**: *{query}*\n")
            lines.append(f"Currently managing **{len(self.candidates)} candidate(s)** and **{len(self.job_descriptions)} active job opening(s)**.\n")
            if self.candidates:
                lines.append("**Indexed Candidates:**")
                for cid, prof in self.candidates.items():
                    lines.append(f"- **{prof.get('name')}** ({prof.get('seniority_level', 'Mid-Level')}, ~{prof.get('years_of_experience', 0)} yrs exp)")
            lines.append("\nYou can ask me to rank candidates, analyze specific technical weaknesses, verify resume claims, or generate customized interview kits.")

        return "\n".join(lines)
