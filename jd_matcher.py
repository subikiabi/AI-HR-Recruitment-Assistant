"""Tool: Job Description Matcher for fit scoring, skill gap analysis, and SWOT generation."""

import re
from typing import Dict, Any, List, Optional
from src.config import TECH_SKILLS_TAXONOMY, STRONG_MATCH_THRESHOLD, MODERATE_MATCH_THRESHOLD


class JDMatcherTool:
    """Matches a screened candidate profile against a Job Description."""

    name = "jd_matcher"
    description = "Compares candidate resume against a Job Description, generating fit score, skill gap breakdown, and SWOT analysis."

    def __init__(self, vector_store: Optional[Any] = None, llm_client: Optional[Any] = None):
        self.vector_store = vector_store
        self.llm_client = llm_client

    def parse_job_description(self, jd_text: str) -> Dict[str, Any]:
        """Extracts structured requirements from raw JD text."""
        lines = [line.strip() for line in jd_text.split("\n") if line.strip()]
        
        # Extract title
        title = "Software Engineer"
        title_match = re.search(r"(?:Job Title|Role|Position)?[:\-]?\s*([A-Za-z0-9\s\/\(\)]+(?:Engineer|Developer|Architect|Manager|Lead|Analyst|Scientist)[^\n]*)", jd_text, re.I)
        if title_match:
            title = title_match.group(1).strip()
        elif lines:
            title = lines[0]

        # Extract required years
        exp_match = re.search(r"(\d+)\+?\s*(?:-\s*\d+)?\s*years?(?:\s+of)?\s*(?:relevant|professional|backend|software)?\s*experience", jd_text, re.I)
        min_years = float(exp_match.group(1)) if exp_match else 3.0

        # Extract skills present in JD
        jd_lower = jd_text.lower()
        found_skills = set()
        for category, skills in TECH_SKILLS_TAXONOMY.items():
            for s in skills:
                if re.search(r"(?:\b|_)" + re.escape(s) + r"(?:\b|_)", jd_lower):
                    found_skills.add(s.title())

        # Distinguish required vs preferred
        required_skills = []
        preferred_skills = []

        preferred_pattern = r"(?:preferred|bonus|nice to have|plus|additional qualifications)[\s\S]*?(?:$|(?=[A-Z\s]{4,}:))"
        preferred_match = re.search(preferred_pattern, jd_text, re.I)
        preferred_block = preferred_match.group(0).lower() if preferred_match else ""

        for skill in found_skills:
            if preferred_block and re.search(r"\b" + re.escape(skill.lower()) + r"\b", preferred_block):
                preferred_skills.append(skill)
            else:
                required_skills.append(skill)

        # Ensure at least some required skills exist
        if not required_skills and found_skills:
            required_skills = list(found_skills)

        return {
            "title": title,
            "min_years_experience": min_years,
            "required_skills": sorted(required_skills),
            "preferred_skills": sorted(preferred_skills),
            "raw_text": jd_text
        }

    def execute(self, candidate_profile: Dict[str, Any], jd_data: Dict[str, Any]) -> Dict[str, Any]:
        """Matches a candidate profile against parsed JD requirements."""
        cand_skills_set = {s.lower() for s in candidate_profile.get("skills", {}).get("all", [])}
        cand_years = candidate_profile.get("years_of_experience", 0.0)

        # 1. Required Skills Match
        required_skills = jd_data.get("required_skills", [])
        matched_required = []
        missing_required = []

        for req in required_skills:
            if req.lower() in cand_skills_set:
                matched_required.append(req)
            else:
                missing_required.append(req)

        req_coverage = len(matched_required) / max(1, len(required_skills))

        # 2. Preferred Skills Match
        preferred_skills = jd_data.get("preferred_skills", [])
        matched_preferred = []
        missing_preferred = []

        for pref in preferred_skills:
            if pref.lower() in cand_skills_set:
                matched_preferred.append(pref)
            else:
                missing_preferred.append(pref)

        pref_coverage = len(matched_preferred) / max(1, len(preferred_skills)) if preferred_skills else 1.0

        # 3. Experience Match
        min_years = jd_data.get("min_years_experience", 3.0)
        if cand_years >= min_years:
            exp_score = 1.0
            exp_assessment = f"Exceeds minimum requirements ({cand_years:.1f} yrs vs {min_years:.1f} yrs required)"
        elif cand_years >= (min_years - 1.0):
            exp_score = 0.75
            exp_assessment = f"Near required threshold ({cand_years:.1f} yrs vs {min_years:.1f} yrs required)"
        else:
            exp_score = max(0.2, cand_years / min_years)
            exp_assessment = f"Below minimum threshold ({cand_years:.1f} yrs vs {min_years:.1f} yrs required)"

        # 4. Semantic / Domain Overlap via RAG (if available)
        rag_score = 0.8
        if self.vector_store and hasattr(self.vector_store, "search"):
            results = self.vector_store.search(
                query=f"{jd_data.get('title', '')} {' '.join(required_skills[:5])}",
                top_k=3,
                filter_dict={"candidate_id": candidate_profile.get("candidate_id")}
            )
            if results:
                rag_score = sum(r["score"] for r in results) / len(results)

        # 5. Calculate Weighted Fit Score (0 - 100)
        weighted_score = (
            (req_coverage * 55.0) +
            (pref_coverage * 15.0) +
            (exp_score * 20.0) +
            (min(1.0, rag_score) * 10.0)
        )
        overall_fit_score = round(min(100.0, max(0.0, weighted_score)), 1)

        # 6. Fit Category & Recommendation
        if overall_fit_score >= STRONG_MATCH_THRESHOLD:
            fit_category = "Strong Match"
            recommendation = "High Priority Interview (Fast-track candidate)"
            badge_color = "green"
        elif overall_fit_score >= MODERATE_MATCH_THRESHOLD:
            fit_category = "Moderate Match"
            recommendation = "Proceed to Technical Screening (Focus on gap areas)"
            badge_color = "orange"
        else:
            fit_category = "Low Match"
            recommendation = "Review closely or consider for alternate roles"
            badge_color = "red"

        # 7. SWOT Analysis Generation
        swot = self._generate_swot(
            candidate_profile, jd_data, matched_required, missing_required, matched_preferred, exp_assessment
        )

        return {
            "candidate_id": candidate_profile.get("candidate_id"),
            "candidate_name": candidate_profile.get("name"),
            "role_title": jd_data.get("title"),
            "overall_fit_score": overall_fit_score,
            "fit_category": fit_category,
            "badge_color": badge_color,
            "recommendation": recommendation,
            "experience_assessment": exp_assessment,
            "required_skills_match": {
                "percentage": round(req_coverage * 100, 1),
                "matched": matched_required,
                "missing": missing_required
            },
            "preferred_skills_match": {
                "percentage": round(pref_coverage * 100, 1) if preferred_skills else 100.0,
                "matched": matched_preferred,
                "missing": missing_preferred
            },
            "swot_analysis": swot
        }

    @staticmethod
    def _generate_swot(
        profile: Dict[str, Any],
        jd: Dict[str, Any],
        matched_req: List[str],
        missing_req: List[str],
        matched_pref: List[str],
        exp_notes: str
    ) -> Dict[str, List[str]]:
        """Constructs a SWOT analysis for the candidate-job pairing."""
        strengths = []
        if matched_req:
            strengths.append(f"Strong overlap on core required technologies: {', '.join(matched_req[:4])}.")
        if matched_pref:
            strengths.append(f"Possesses bonus qualifications: {', '.join(matched_pref[:3])}.")
        strengths.extend(profile.get("strengths", [])[:2])

        weaknesses = []
        if missing_req:
            weaknesses.append(f"Missing stated required skills: {', '.join(missing_req)}.")
        if "below" in exp_notes.lower():
            weaknesses.append(f"Total experience {exp_notes}.")
        weaknesses.extend(profile.get("red_flags", [])[:2])

        opportunities = [
            f"Could ramp up quickly on {missing_req[0]} given their foundational knowledge." if missing_req else "Immediate contributor to architecture and feature delivery.",
            f"Provides potential cross-functional leadership in {profile.get('seniority_level', 'Mid-Level')} engineering capacity.",
            "Can assist in establishing testing or CI/CD pipelines based on past achievements."
        ]

        threats = [
            f"Gaps in {', '.join(missing_req[:2])} may cause onboarding delay or require pair programming." if missing_req else "Risk of candidate having competing offers given strong background.",
            "Potential domain adjustment period depending on product complexity."
        ]

        return {
            "strengths": strengths,
            "weaknesses": weaknesses if weaknesses else ["No critical technical deficiencies identified."],
            "opportunities": opportunities,
            "threats": threats
        }
