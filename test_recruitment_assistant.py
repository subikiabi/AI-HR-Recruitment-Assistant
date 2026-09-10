"""Verification tests for the AI HR Recruitment Assistant backend."""

import os
from pathlib import Path
from src.parser import DocumentParser
from src.rag.chunker import ResumeChunker
from src.rag.vector_store import InMemoVectorStore
from src.tools.resume_screener import ResumeScreenerTool
from src.tools.jd_matcher import JDMatcherTool
from src.tools.question_generator import InterviewQuestionGeneratorTool
from src.tools.candidate_comparer import CandidateComparerTool
from src.agent.recruiter_agent import RecruiterAgent
from src.utils.export_utils import ReportExporter


def test_system_pipeline():
    print("--- 1. Testing Document Parser ---")
    resume_path = Path("sample_data/resumes/resume_senior_backend.txt")
    jd_path = Path("sample_data/job_descriptions/jd_senior_python_engineer.txt")
    
    resume_text = DocumentParser.extract_text_from_file(resume_path)
    jd_text = DocumentParser.extract_text_from_file(jd_path)
    assert len(resume_text) > 100, "Resume text should not be empty"
    assert len(jd_text) > 100, "JD text should not be empty"
    print(f"Parsed resume length: {len(resume_text)} chars, JD length: {len(jd_text)} chars")

    print("\n--- 2. Testing RAG Chunker & Vector Store ---")
    chunks = ResumeChunker.chunk_resume(resume_text, "resume_senior_backend.txt", "cand_1")
    assert len(chunks) >= 3, f"Expected at least 3 chunks, got {len(chunks)}"
    print(f"Generated {len(chunks)} chunks with metadata sections: {[c.metadata['section'] for c in chunks]}")

    store = InMemoVectorStore()
    store.add_documents(chunks)
    search_res = store.search("FastAPI Kafka microservices latency", top_k=2)
    assert len(search_res) > 0, "RAG search should return results"
    print(f"Top RAG match score: {search_res[0]['score']} for query 'FastAPI Kafka microservices latency'")

    print("\n--- 3. Testing Resume Screener Tool ---")
    screener = ResumeScreenerTool()
    profile = screener.execute(resume_text, "cand_1")
    print(f"Candidate Name: {profile['name']}")
    print(f"Experience: {profile['years_of_experience']} yrs, Seniority: {profile['seniority_level']}")
    print(f"Extracted Skills Count: {profile['skills']['count']}")
    assert "Alex Morgan" in profile["name"], f"Expected Alex Morgan, got {profile['name']}"
    assert profile["years_of_experience"] >= 5, "Expected 5+ years of experience"

    print("\n--- 4. Testing JD Matcher Tool & SWOT ---")
    matcher = JDMatcherTool(vector_store=store)
    jd_data = matcher.parse_job_description(jd_text)
    match_result = matcher.execute(profile, jd_data)
    print(f"Overall Fit Score: {match_result['overall_fit_score']}% ({match_result['fit_category']})")
    print(f"Required Skills Matched: {match_result['required_skills_match']['matched']}")
    print(f"Missing Skills: {match_result['required_skills_match']['missing']}")
    assert match_result["overall_fit_score"] >= 70, "Expected strong fit for Alex Morgan on Senior Python JD"

    print("\n--- 5. Testing Interview Question Generator Tool ---")
    question_tool = InterviewQuestionGeneratorTool()
    interview_kit = question_tool.execute(profile, match_result, jd_data)
    print(f"Generated {len(interview_kit['technical_questions'])} Technical Questions")
    print(f"Generated {len(interview_kit['behavioral_questions'])} Behavioral Questions")
    assert len(interview_kit["technical_questions"]) >= 2

    # Check export formatting
    exported_md = ReportExporter.format_interview_kit_markdown(interview_kit)
    assert "# Interview Kit: Alex Morgan" in exported_md

    print("\n--- 6. Testing Autonomous Recruiter Agent ---")
    agent = RecruiterAgent()
    agent.ingest_resume(resume_text, "resume_senior_backend.txt", "cand_1")
    
    # Ingest 2nd candidate
    frontend_text = DocumentParser.extract_text_from_file("sample_data/resumes/resume_frontend_dev.txt")
    agent.ingest_resume(frontend_text, "resume_frontend_dev.txt", "cand_2")
    
    agent.set_active_job_description(jd_text, "active_jd")
    eval_results = agent.evaluate_all_candidates("active_jd")
    assert len(eval_results) == 2, "Expected 2 evaluated candidates"

    # Test comparison
    comp_result = agent.compare_candidates("active_jd")
    print(f"Leaderboard Rank 1: {comp_result['leaderboard'][0]['candidate_name']} ({comp_result['leaderboard'][0]['overall_fit_score']}%)")
    assert comp_result["leaderboard"][0]["candidate_name"] == "Alex Morgan"

    # Test conversational query
    chat_resp = agent.ask_recruiter_agent("Compare the candidates and tell me who to interview first.")
    clean_preview = chat_resp['answer'][:250].encode('ascii', 'backslashreplace').decode('ascii')
    print(f"Agent Chat Answer preview:\n{clean_preview}...")
    assert len(chat_resp["tool_traces"]) > 0, "Expected tool execution traces in response"

    print("\nALL BACKEND TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_system_pipeline()
