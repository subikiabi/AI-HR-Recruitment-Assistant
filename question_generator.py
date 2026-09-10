"""Tool: Interview Question Generator with evaluation rubrics and gap-probing capabilities."""

from typing import Dict, Any, List, Optional


class InterviewQuestionGeneratorTool:
    """Generates tailored interview questions, scenario probes, and scoring rubrics."""

    name = "interview_question_generator"
    description = "Generates technical, gap-probing, and behavioral interview questions tailored to the candidate's resume and JD."

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client

    def execute(
        self,
        candidate_profile: Dict[str, Any],
        match_result: Dict[str, Any],
        jd_data: Dict[str, Any],
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generates a complete structured interview kit."""
        candidate_name = candidate_profile.get("name", "Candidate")
        role_title = jd_data.get("title", "Target Role")
        
        # Pull matched and missing skills
        req_match = match_result.get("required_skills_match", {})
        matched_skills = req_match.get("matched", [])
        missing_skills = req_match.get("missing", [])

        # 1. Technical Deep-Dive Questions (based on their actual matched skills and resume claims)
        technical_questions = self._generate_technical_questions(candidate_profile, matched_skills)

        # 2. Gap & Probe Questions (focused specifically on missing skills or potential risks)
        gap_questions = self._generate_gap_questions(candidate_profile, missing_skills)

        # 3. Behavioral & STAR Leadership Questions (tailored to seniority level)
        behavioral_questions = self._generate_behavioral_questions(candidate_profile)

        # 4. Role-Specific System Design / Coding Challenge Scenario
        scenario_challenge = self._generate_scenario(candidate_profile, role_title, matched_skills)

        return {
            "candidate_name": candidate_name,
            "target_role": role_title,
            "overall_fit_score": match_result.get("overall_fit_score", 0),
            "technical_questions": technical_questions,
            "gap_questions": gap_questions,
            "behavioral_questions": behavioral_questions,
            "scenario_challenge": scenario_challenge,
        }

    def _generate_technical_questions(self, profile: Dict[str, Any], matched_skills: List[str]) -> List[Dict[str, Any]]:
        """Generates technical questions probing actual projects and claimed skills."""
        questions = []
        
        if "Python" in matched_skills or "Fastapi" in matched_skills or "Django" in matched_skills:
            questions.append({
                "category": "Technical Architecture",
                "question": "In your recent projects, how did you handle asynchronous I/O and concurrency in Python (e.g., async/await, background workers like Celery or asyncio)? Walk us through an optimization you made that significantly reduced response latency.",
                "context": f"Probing practical depth in Python frameworks mentioned in {profile.get('name')}'s experience.",
                "rubric": {
                    "ideal_answer": "Mentions event loops, non-blocking calls, connection pooling, profiling tools (cProfile/py-spy), and concrete metric impact.",
                    "watch_outs": "Confuses CPU-bound with I/O-bound tasks; cannot explain thread safety or GIL implications."
                }
            })

        if any(s in matched_skills for s in ["Postgresql", "Postgres", "Mysql", "Redis"]):
            questions.append({
                "category": "Database & Caching Design",
                "question": "When scaling relational databases under heavy read/write traffic, how do you approach schema indexing, query optimization, and cache invalidation strategies with Redis?",
                "context": "Assessing database scaling and caching practices for mission-critical endpoints.",
                "rubric": {
                    "ideal_answer": "Explains EXPLAIN ANALYZE, composite indexes, read replicas, cache-aside vs write-through patterns, and TTL expiration.",
                    "watch_outs": "Defaults to 'just add more memory' or lacks understanding of index locking and cache stampede."
                }
            })

        if any(s in matched_skills for s in ["Kafka", "Rabbitmq", "Microservices"]):
            questions.append({
                "category": "Distributed Systems & Messaging",
                "question": "How do you guarantee idempotency, message ordering, and failure recovery when building event-driven microservices with message brokers like Kafka or RabbitMQ?",
                "context": "Verifying event-driven claims and failure mode resilience.",
                "rubric": {
                    "ideal_answer": "Discusses deduplication keys, dead-letter queues (DLQ), retry exponential backoff, and consumer group offset management.",
                    "watch_outs": "Assumes exactly-once delivery without deduplication logic or ignores network partition handling."
                }
            })

        if any(s in matched_skills for s in ["React", "Next.Js", "Typescript"]):
            questions.append({
                "category": "Frontend Performance & State",
                "question": "How do you architect large-scale React applications to prevent unnecessary re-renders, optimize bundle size, and ensure smooth Core Web Vitals?",
                "context": "Probing frontend engineering standards and modern React patterns.",
                "rubric": {
                    "ideal_answer": "Cites React.memo, useMemo/useCallback nuances, code splitting via dynamic imports, TanStack Query / Redux caching, and accessibility.",
                    "watch_outs": "Overuses global state for local inputs or cannot explain virtual DOM reconciliation."
                }
            })

        if any(s in matched_skills for s in ["Spark", "Snowflake", "Airflow", "Dbt"]):
            questions.append({
                "category": "Data Engineering Pipelines",
                "question": "Can you describe how you design and monitor resilient ETL/ELT pipelines in Airflow and dbt, ensuring data quality and idempotency on backfills?",
                "context": "Evaluating data orchestration and lakehouse architecture experience.",
                "rubric": {
                    "ideal_answer": "Discusses idempotent writes, partition overwrites, Great Expectations / dbt test validations, and cost optimization on compute clusters.",
                    "watch_outs": "Relies on manual script reruns without automated lineage or schema change alerting."
                }
            })

        # Default fallback technical question if list is short
        if len(questions) < 2:
            questions.append({
                "category": "System Design & Code Quality",
                "question": "Can you walk through an end-to-end service or feature you designed from scratch? What architectural trade-offs did you make between delivery speed and maintainability?",
                "context": "General architectural reasoning and design philosophy evaluation.",
                "rubric": {
                    "ideal_answer": "Articulates clear requirement trade-offs, modular design, automated testing, and observability telemetry.",
                    "watch_outs": "Describes only visual or trivial tasks without depth on failure handling or scaling constraints."
                }
            })

        return questions

    def _generate_gap_questions(self, profile: Dict[str, Any], missing_skills: List[str]) -> List[Dict[str, Any]]:
        """Generates targeted probe questions for missing requirements or red flags."""
        questions = []

        for skill in missing_skills[:2]:
            questions.append({
                "category": f"Skill Gap: {skill}",
                "question": f"The job description highlights {skill} as an important component of our stack, which wasn't prominently featured on your resume. Have you worked with {skill} or an equivalent technology, and how would you approach getting up to speed?",
                "context": f"Validating candidate's adaptability and willingness to ramp up on {skill}.",
                "rubric": {
                    "ideal_answer": "Draws analogies to similar tools they have mastered, shows proactive self-learning approach, or demonstrates hands-on lab experiments.",
                    "watch_outs": "Dismisses the importance of the technology or claims deep knowledge that contradicts their resume."
                }
            })

        for flag in profile.get("red_flags", [])[:1]:
            questions.append({
                "category": "Resume Clarification & Best Practices",
                "question": "Could you share an example of how you measure and track the success and business ROI of the engineering projects you work on?",
                "context": "Probing metric-driven impact to clarify vague resume bullet points.",
                "rubric": {
                    "ideal_answer": "Provides concrete metrics: latency reduction, server cost savings, uptime, or user conversion boosts.",
                    "watch_outs": "Speaks only in high-level generalizations without any measurable outcomes."
                }
            })

        return questions

    def _generate_behavioral_questions(self, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generates STAR behavioral questions tailored to seniority level."""
        seniority = profile.get("seniority_level", "Mid-Level")
        
        if "Senior" in seniority or "Staff" in seniority:
            return [
                {
                    "category": "Technical Disagreement & Influence",
                    "question": "Tell us about a time when you strongly disagreed with an engineering decision or product direction proposed by a team member or leader. How did you handle it and what was the outcome?",
                    "rubric": {
                        "ideal_answer": "Demonstrates empathetic communication, uses data and prototypes to persuade, and adheres to 'disagree and commit' if overruled.",
                        "watch_outs": "Shows passive aggression, stubbornness, or blames others for technical debt."
                    }
                },
                {
                    "category": "Mentorship & Team Elevation",
                    "question": "Can you share a specific example of how you helped a junior or struggling engineer level up their technical skills and engineering confidence?",
                    "rubric": {
                        "ideal_answer": "Highlights pair programming, constructive code review culture, personalized growth plans, and psychological safety.",
                        "watch_outs": "Believes mentorship is merely pointing out mistakes or doing the work for them."
                    }
                }
            ]
        else:
            return [
                {
                    "category": "Handling Ambiguity & Production Incidents",
                    "question": "Describe a production bug or unexpected deployment issue that occurred under pressure. How did you triage the problem, communicate with the team, and resolve it?",
                    "rubric": {
                        "ideal_answer": "Follows systematic root-cause analysis, prioritizes customer mitigation first, and writes a blameless post-mortem.",
                        "watch_outs": "Panics, attempts unvetted hotfixes directly in prod, or hides errors from stakeholders."
                    }
                },
                {
                    "category": "Collaboration & Cross-functional Work",
                    "question": "Tell us about a situation where project requirements were shifting rapidly. How did you organize your priorities and collaborate with designers or product managers?",
                    "rubric": {
                        "ideal_answer": "Demonstrates agile adaptation, seeks clear definitions of done, and balances perfectionism with pragmatic delivery.",
                        "watch_outs": "Resists change or becomes paralyzed when specifications are incomplete."
                    }
                }
            ]

    def _generate_scenario(self, profile: Dict[str, Any], role_title: str, matched_skills: List[str]) -> Dict[str, Any]:
        """Generates an interactive real-world system scenario challenge."""
        return {
            "title": f"Real-world Architecture Challenge for {role_title}",
            "scenario": (
                "Suppose our core service experiences a sudden 10x traffic spike during a major company launch event. "
                "Database CPU hits 98%, API p99 latency degrades to 4 seconds, and queue consumer workers start lagging. "
                "How would you diagnose the primary bottleneck, stabilize the system within the first 15 minutes, and design a long-term architectural safeguard?"
            ),
            "key_evaluation_dimensions": [
                "Immediate incident triage & mitigation (circuit breakers, rate limiting, shed load)",
                "Monitoring & diagnostics (tracing p99 queries, queue depth metrics, database connection saturation)",
                "Long-term architectural hardening (horizontal autoscaling, read replicas, cache warming, decoupled async workers)"
            ]
        }
