"""Resume and Job Description Chunker with section-aware intelligence."""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


class ResumeChunker:
    """Intelligently chunks resumes and job descriptions using section boundaries and windowing."""

    SECTION_PATTERNS = {
        "summary": r"(?:professional\s+summary|executive\s+summary|summary|profile|about\s+me|objective)",
        "skills": r"(?:core\s+technical\s+skills|technical\s+skills|core\s+competencies|skills\s*&?\s*technologies|skills|competencies|areas\s+of\s+expertise)",
        "experience": r"(?:work\s+experience|professional\s+experience|employment\s+history|experience|career\s+history)",
        "education": r"(?:education\s*&?\s*credentials|academic\s+background|education|qualifications|academic\s+qualifications)",
        "projects": r"(?:key\s+projects|notable\s+projects|technical\s+projects|projects)",
        "certifications": r"(?:certifications\s*&?\s*workshops|certifications\s*&?\s*licenses|certifications|licenses|awards)",
    }

    @classmethod
    def extract_candidate_name(cls, text: str) -> str:
        """Heuristic to extract candidate name from the top lines of a resume."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return "Unknown Candidate"
        
        # Usually candidate name is on the 1st or 2nd non-empty line
        for line in lines[:3]:
            # Clean unwanted characters
            clean_line = re.sub(r"[^a-zA-Z\s\.\-]", "", line).strip()
            words = clean_line.split()
            if 2 <= len(words) <= 4 and not any(kw in clean_line.lower() for kw in ["resume", "curriculum", "vitae", "email", "phone", "summary", "page"]):
                return clean_line.title()
        
        return lines[0][:40].strip()

    @classmethod
    def segment_resume_sections(cls, text: str) -> Dict[str, str]:
        """Splits resume text into canonical sections based on common headings."""
        sections: Dict[str, str] = {"header": "", "summary": "", "skills": "", "experience": "", "education": "", "projects": "", "certifications": "", "other": ""}
        
        combined_pattern = "|".join(
            fr"(?P<{sec_name}>^[ \t]*(?:#+\s*)?{pattern}(?:\s*:|\s*$))"
            for sec_name, pattern in cls.SECTION_PATTERNS.items()
        )
        regex = re.compile(combined_pattern, re.IGNORECASE | re.MULTILINE)

        matches = list(regex.finditer(text))
        if not matches:
            sections["summary"] = text[:500]
            sections["other"] = text[500:]
            return sections

        # Extract header (everything before the first matched section)
        first_match = matches[0]
        if first_match.start() > 0:
            sections["header"] = text[: first_match.start()].strip()

        # Extract content between matched headings
        for i, match in enumerate(matches):
            matched_sec = match.lastgroup
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sec_content = text[start_pos:end_pos].strip()

            if matched_sec in sections:
                if sections[matched_sec]:
                    sections[matched_sec] += "\n\n" + sec_content
                else:
                    sections[matched_sec] = sec_content
            else:
                sections["other"] += "\n\n" + sec_content

        return sections

    @classmethod
    def chunk_resume(
        cls,
        text: str,
        source_filename: str,
        candidate_id: str,
        max_chunk_chars: int = 800,
        chunk_overlap: int = 150
    ) -> List[DocumentChunk]:
        """Creates semantic, section-tagged chunks for a candidate resume."""
        candidate_name = cls.extract_candidate_name(text)
        sections = cls.segment_resume_sections(text)
        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        # Always add a compact profile summary chunk
        profile_summary = f"Candidate: {candidate_name}\n"
        if sections.get("header"):
            profile_summary += f"Contact & Info:\n{sections['header']}\n"
        if sections.get("summary"):
            profile_summary += f"Summary:\n{sections['summary']}\n"
        if sections.get("skills"):
            profile_summary += f"Skills:\n{sections['skills'][:400]}\n"

        chunks.append(
            DocumentChunk(
                chunk_id=f"{candidate_id}_chunk_{chunk_idx}",
                text=profile_summary.strip(),
                metadata={
                    "candidate_id": candidate_id,
                    "candidate_name": candidate_name,
                    "source_filename": source_filename,
                    "section": "profile_overview",
                    "chunk_index": chunk_idx
                }
            )
        )
        chunk_idx += 1

        # Process each section into sliding windows if needed
        for sec_name, content in sections.items():
            if not content.strip() or sec_name in ["header"]:
                continue
            
            # If section content is reasonably short, create a single chunk
            if len(content) <= max_chunk_chars:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{candidate_id}_chunk_{chunk_idx}",
                        text=f"[{sec_name.upper()}]\n{content}",
                        metadata={
                            "candidate_id": candidate_id,
                            "candidate_name": candidate_name,
                            "source_filename": source_filename,
                            "section": sec_name,
                            "chunk_index": chunk_idx
                        }
                    )
                )
                chunk_idx += 1
            else:
                # Break down long sections (like Experience) by paragraphs or sliding window
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                current_block = ""
                for p in paragraphs:
                    if len(current_block) + len(p) <= max_chunk_chars:
                        current_block = f"{current_block}\n\n{p}".strip()
                    else:
                        if current_block:
                            chunks.append(
                                DocumentChunk(
                                    chunk_id=f"{candidate_id}_chunk_{chunk_idx}",
                                    text=f"[{sec_name.upper()}]\n{current_block}",
                                    metadata={
                                        "candidate_id": candidate_id,
                                        "candidate_name": candidate_name,
                                        "source_filename": source_filename,
                                        "section": sec_name,
                                        "chunk_index": chunk_idx
                                    }
                                )
                            )
                            chunk_idx += 1
                        current_block = p
                
                if current_block:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{candidate_id}_chunk_{chunk_idx}",
                            text=f"[{sec_name.upper()}]\n{current_block}",
                            metadata={
                                "candidate_id": candidate_id,
                                "candidate_name": candidate_name,
                                "source_filename": source_filename,
                                "section": sec_name,
                                "chunk_index": chunk_idx
                            }
                        )
                    )
                    chunk_idx += 1

        return chunks

    @classmethod
    def chunk_job_description(cls, text: str, jd_id: str, title: str = "") -> List[DocumentChunk]:
        """Chunks a Job Description into manageable queryable segments."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not title and lines:
            title = lines[0]

        chunks = []
        # Main overview chunk
        chunks.append(
            DocumentChunk(
                chunk_id=f"{jd_id}_overview",
                text=text[:1000],
                metadata={"doc_type": "job_description", "title": title, "section": "overview"}
            )
        )
        return chunks
