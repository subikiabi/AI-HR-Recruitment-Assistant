"""Tool: Resume Screener for parsing, profile extraction, and red-flag analysis."""

import re
from typing import Dict, Any, List, Optional
from src.config import TECH_SKILLS_TAXONOMY, FLAT_SKILLS
from src.rag.chunker import ResumeChunker


class ResumeScreenerTool:
    """Tool for screening resumes, extracting candidate competencies, experience, and red flags."""

    name = "resume_screener"
    description = "Extracts candidate details, technical skills, years of experience, strengths, and potential red flags from resume text."

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client

    def execute(self, resume_text: str, candidate_id: str = "candidate_1") -> Dict[str, Any]:
        """Screen and parse a candidate resume into a structured profile."""
        candidate_name = ResumeChunker.extract_candidate_name(resume_text)
        sections = ResumeChunker.segment_resume_sections(resume_text)
        
        # 1. Contact & Info Extraction
        contact_info = self._extract_contact_info(resume_text)
        
        # 2. Skills Extraction
        skills_found = self._extract_skills(resume_text)

        # 3. Experience Estimation
        years_of_experience = self._estimate_years_experience(resume_text, sections.get("experience", ""))
        seniority = self._infer_seniority(years_of_experience, resume_text)

        # 4. Education & Certifications
        education = self._extract_education(sections.get("education", "") or resume_text)
        certifications = self._extract_certifications(sections.get("certifications", "") or resume_text)

        # 5. Strengths & Red Flags
        strengths = self._identify_strengths(resume_text, skills_found, years_of_experience)
        red_flags = self._detect_red_flags(resume_text, sections.get("experience", ""))

        # 6. Overall Candidate Summary
        summary = (
            sections.get("summary", "").strip() 
            or f"{candidate_name} is a {seniority} level professional with ~{years_of_experience} years of experience specializing in {', '.join(skills_found['all'][:5])}."
        )

        return {
            "candidate_id": candidate_id,
            "name": candidate_name,
            "contact": contact_info,
            "years_of_experience": years_of_experience,
            "seniority_level": seniority,
            "skills": skills_found,
            "education": education,
            "certifications": certifications,
            "strengths": strengths,
            "red_flags": red_flags,
            "summary": summary,
            "raw_sections": sections
        }

    @staticmethod
    def _extract_contact_info(text: str) -> Dict[str, str]:
        """Extracts email, phone, location, and web links."""
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
        phone_match = re.search(r"(?:\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", text)
        linkedin_match = re.search(r"linkedin\.com/in/[\w\-]+", text, re.I)
        github_match = re.search(r"github\.com/[\w\-]+", text, re.I)
        
        location = ""
        loc_match = re.search(r"(?:Location|Address)?[:\-]?\s*([A-Za-z\s]+,\s*[A-Z]{2})", text)
        if loc_match:
            location = loc_match.group(1).strip()

        return {
            "email": email_match.group(0) if email_match else "Not provided",
            "phone": phone_match.group(0) if phone_match else "Not provided",
            "location": location if location else "Not provided",
            "linkedin": linkedin_match.group(0) if linkedin_match else "",
            "github": github_match.group(0) if github_match else ""
        }

    @staticmethod
    def _extract_skills(text: str) -> Dict[str, Any]:
        """Extracts and categorizes technical skills found in the resume."""
        text_lower = text.lower()
        categorized: Dict[str, List[str]] = {}
        all_skills = []

        for category, skills in TECH_SKILLS_TAXONOMY.items():
            matched = []
            for skill in skills:
                # Word boundary match
                pattern = r"(?:\b|_)" + re.escape(skill) + r"(?:\b|_)"
                if re.search(pattern, text_lower):
                    formatted_skill = skill.title()
                    if formatted_skill not in matched:
                        matched.append(formatted_skill)
                        if formatted_skill not in all_skills:
                            all_skills.append(formatted_skill)
            if matched:
                categorized[category] = matched

        return {
            "categorized": categorized,
            "all": all_skills,
            "count": len(all_skills)
        }

    @staticmethod
    def _estimate_years_experience(full_text: str, exp_text: str) -> float:
        """Estimates total years of experience from year intervals (e.g. 2018 - 2022)."""
        search_target = exp_text if len(exp_text) > 100 else full_text
        
        # Look for explicit statements like "7+ years of experience"
        explicit_match = re.search(r"(\d+)\+?\s+years?(?:\s+of)?\s+experience", search_target, re.I)
        if explicit_match:
            return float(explicit_match.group(1))

        # Parse date ranges like 2018 - Present, 2016 - 2020
        date_pattern = r"\b(19\d\d|20\d\d)\b\s*(?:–|-|to)\s*\b(19\d\d|20\d\d|present|current)\b"
        matches = re.findall(date_pattern, search_target, re.I)
        
        current_year = 2026
        intervals = []
        for start, end in matches:
            try:
                start_yr = int(start)
                end_yr = current_year if end.lower() in ["present", "current"] else int(end)
                if end_yr >= start_yr:
                    intervals.append((start_yr, end_yr))
            except ValueError:
                continue

        if not intervals:
            return 3.0  # Safe default if undetermined

        # Merge overlapping intervals
        intervals.sort(key=lambda x: x[0])
        merged = []
        for interval in intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))

        total_years = sum(end - start for start, end in merged)
        return max(1.0, float(total_years))

    @staticmethod
    def _infer_seniority(years: float, text: str) -> str:
        """Infers candidate seniority level based on years and leadership cues."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["principal", "staff engineer", "engineering director", "head of engineering"]):
            return "Staff / Principal"
        elif years >= 7 or "lead" in text_lower or "senior backend" in text_lower or "senior engineer" in text_lower:
            return "Senior"
        elif years >= 3:
            return "Mid-Level"
        else:
            return "Junior / Entry-Level"

    @staticmethod
    def _extract_education(text: str) -> List[str]:
        """Extracts degree and institution mentions."""
        degrees = []
        degree_patterns = [
            r"(?:Bachelor|Master|Doctor|PhD|B\.S\.|M\.S\.|B\.A\.|B\.Tech|M\.Tech)[^\n,]+(?:University|College|Institute)?[^\n]*",
            r"(?:B\.S\.|M\.S\.|B\.E\.|B\.Tech|M\.Tech)\s+in\s+[^\n]+"
        ]
        for pattern in degree_patterns:
            matches = re.findall(pattern, text, re.I)
            for m in matches:
                clean_m = re.sub(r"\s+", " ", m).strip()
                if clean_m and clean_m not in degrees:
                    degrees.append(clean_m)

        if not degrees and ("bachelor" in text.lower() or "master" in text.lower()):
            for line in text.split("\n"):
                if any(k in line.lower() for k in ["bachelor", "master", "phd", "degree", "university"]):
                    degrees.append(line.strip())
        
        return degrees[:3] if degrees else ["Degree / Education not explicitly listed"]

    @staticmethod
    def _extract_certifications(text: str) -> List[str]:
        """Extracts professional certifications."""
        certs = []
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines:
            if any(term in line.lower() for term in ["certified", "aws", "ckad", "cka", "azure", "gcp", "pmp", "scrum master"]):
                clean_c = re.sub(r"^[\*\-\•\d\.]+\s*", "", line)
                if clean_c and clean_c not in certs:
                    certs.append(clean_c)
        return certs[:4]

    @staticmethod
    def _identify_strengths(text: str, skills: Dict[str, Any], years: float) -> List[str]:
        """Identifies key candidate strengths."""
        strengths = []
        if years >= 5:
            strengths.append(f"Substantial depth of industry experience (~{int(years)} years).")
        
        if "cloud_and_devops" in skills.get("categorized", {}):
            cloud_skills = ", ".join(skills["categorized"]["cloud_and_devops"][:3])
            strengths.append(f"Strong modern Cloud & DevOps tooling expertise ({cloud_skills}).")

        if any(w in text.lower() for w in ["mentored", "led", "architected", "spearheaded"]):
            strengths.append("Demonstrated technical leadership, architecture design, and mentoring capabilities.")

        if any(w in text.lower() for w in ["reduced", "improved", "optimized", "increased", "%", "ms"]):
            strengths.append("Results-oriented work history highlighting quantifiable impact and metrics.")

        if len(skills.get("all", [])) >= 10:
            strengths.append(f"Broad and versatile technical toolkit spanning {len(skills['all'])} identified technologies.")

        return strengths if strengths else ["Solid foundational skill set matching core responsibilities."]

    @staticmethod
    def _detect_red_flags(text: str, exp_text: str) -> List[str]:
        """Identifies potential concerns, ambiguities, or resume red flags."""
        red_flags = []
        target = exp_text if len(exp_text) > 100 else text

        # Check for lack of quantifiable metrics
        has_metrics = bool(re.search(r"(\d+%\s*|\d+x\s*|\$\d+|\d+\s*ms|\d+\s*(?:million|m|k))", target, re.I))
        if not has_metrics:
            red_flags.append("Vague impact: Lack of quantified achievements, percentages, or concrete business metrics.")

        # Check for short tenure or frequent job hops
        short_tenure_pattern = r"\b(20\d\d)\b\s*-\s*\b(20\d\d)\b"
        hops = re.findall(short_tenure_pattern, target)
        short_stints = [int(end) - int(start) for start, end in hops if (int(end) - int(start)) <= 1]
        if len(short_stints) >= 3:
            red_flags.append("Multiple short-tenure positions (under 1 year) indicating potential job-hopping risk.")

        # Check for missing cloud/testing evidence
        if not any(k in target.lower() for k in ["pytest", "jest", "unit test", "test", "tdd"]):
            red_flags.append("No explicit mention of automated testing practices (unit tests, integration, TDD).")

        return red_flags
