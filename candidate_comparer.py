"""Tool: Candidate Comparer for multi-candidate evaluation, ranking, and hiring tradeoffs."""

from typing import Dict, Any, List, Optional


class CandidateComparerTool:
    """Compares multiple candidates against a target Job Description."""

    name = "candidate_comparer"
    description = "Ranks and compares multiple candidates against a Job Description, highlighting key tradeoffs and top recommendations."

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client

    def execute(self, match_results: List[Dict[str, Any]], profiles_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Produces a ranked comparison summary and trade-off matrix."""
        if not match_results:
            return {"ranked_candidates": [], "top_recommendation": None, "comparison_summary": "No candidates to compare."}

        # Sort descending by fit score
        ranked = sorted(match_results, key=lambda x: x.get("overall_fit_score", 0), reverse=True)

        leaderboard = []
        for rank, match in enumerate(ranked, start=1):
            cand_id = match.get("candidate_id")
            prof = profiles_by_id.get(cand_id, {})
            leaderboard.append({
                "rank": rank,
                "candidate_id": cand_id,
                "candidate_name": match.get("candidate_name"),
                "overall_fit_score": match.get("overall_fit_score"),
                "fit_category": match.get("fit_category"),
                "years_experience": prof.get("years_of_experience", 0),
                "seniority_level": prof.get("seniority_level", "Unknown"),
                "required_match_pct": match.get("required_skills_match", {}).get("percentage", 0),
                "matched_skills": match.get("required_skills_match", {}).get("matched", []),
                "missing_skills": match.get("required_skills_match", {}).get("missing", []),
                "recommendation": match.get("recommendation")
            })

        top_candidate = leaderboard[0] if leaderboard else None
        
        # Synthesize tradeoffs
        tradeoffs = []
        if len(leaderboard) >= 2:
            first = leaderboard[0]
            second = leaderboard[1]
            tradeoffs.append(
                f"**{first['candidate_name']} (Score: {first['overall_fit_score']}%)** ranks #1 primarily due to "
                f"higher alignment on core requirements ({first['required_match_pct']}% match) and {first['years_experience']} years of experience."
            )
            tradeoffs.append(
                f"**{second['candidate_name']} (Score: {second['overall_fit_score']}%)** is a strong runner-up, but lacks "
                f"{', '.join(second['missing_skills'][:3]) if second['missing_skills'] else 'minor qualifications'}."
            )
        else:
            tradeoffs.append(f"Only one candidate currently evaluated: {top_candidate['candidate_name']}.")

        return {
            "leaderboard": leaderboard,
            "top_candidate": top_candidate,
            "tradeoffs": tradeoffs,
            "total_evaluated": len(leaderboard)
        }
