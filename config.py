"""Configuration settings for AI HR Recruitment Assistant."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# LLM Configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "models/text-embedding-004"

# Scoring thresholds
STRONG_MATCH_THRESHOLD = 80
MODERATE_MATCH_THRESHOLD = 60

# Common tech skills taxonomy for parsing & matching
TECH_SKILLS_TAXONOMY = {
    "languages": [
        "python", "javascript", "typescript", "java", "c++", "c#", "go", "golang",
        "rust", "ruby", "php", "swift", "kotlin", "scala", "sql", "bash", "shell", "r"
    ],
    "frontend": [
        "react", "react.js", "next.js", "vue", "vue.js", "angular", "svelte", "html", "html5",
        "css", "css3", "tailwind", "tailwind css", "bootstrap", "redux", "redux toolkit",
        "react query", "tanstack query", "material-ui", "framer motion", "webpack", "vite"
    ],
    "backend_frameworks": [
        "fastapi", "django", "flask", "express", "express.js", "node.js", "nestjs",
        "spring", "spring boot", "ruby on rails", "asp.net", "laravel", "celery",
        "pydantic", "sqlalchemy"
    ],
    "databases_and_cache": [
        "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
        "cassandra", "dynamodb", "sqlite", "neo4j", "mariadb", "snowflake", "bigquery"
    ],
    "cloud_and_devops": [
        "aws", "amazon web services", "azure", "gcp", "google cloud", "docker",
        "kubernetes", "k8s", "terraform", "helm", "ci/cd", "github actions", "gitlab ci",
        "jenkins", "ansible", "prometheus", "grafana", "datadog", "ecs", "eks", "lambda", "s3"
    ],
    "data_and_ai": [
        "apache spark", "spark", "pyspark", "apache kafka", "kafka", "apache airflow",
        "airflow", "dbt", "databricks", "hadoop", "flink", "pandas", "numpy", "scikit-learn",
        "tensorflow", "pytorch", "hugging face", "langchain", "llamaindex", "rag", "tableau", "power bi"
    ],
    "architecture_and_practices": [
        "microservices", "rest", "restful", "grpc", "graphql", "event-driven", "system design",
        "distributed systems", "tdd", "agile", "scrum", "oauth2", "jwt", "clean architecture"
    ]
}

FLAT_SKILLS = set()
for category, skills in TECH_SKILLS_TAXONOMY.items():
    for s in skills:
        FLAT_SKILLS.add(s.lower())
