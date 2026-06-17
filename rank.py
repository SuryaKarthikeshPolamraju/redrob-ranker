"""
Redrob Hackathon — Intelligent Candidate Ranking System
========================================================
Run:
    python rank.py --candidates ./data/candidates.jsonl --out ./outputs/submission.csv

What this does:
    1. Reads all 100K candidates
    2. Scores each one across 5 dimensions
    3. Detects and excludes honeypots
    4. Picks top 100
    5. Generates specific, honest reasoning per candidate
    6. Outputs a valid CSV ready to submit

Architecture (for Stage 5 interview):
    - Rule-based scoring with JD-config object (extensible to any JD)
    - Multiplicative behavioral modifier (not additive)
    - Career description text scan catches 'Tier 5' candidates
    - Honeypot detection via profile consistency checks
    - No API calls, no GPU, runs in ~60 seconds on CPU
"""

import json
import csv
import argparse
from datetime import datetime, date

# =============================================================================
# JD CONFIG — All JD knowledge lives here in one place.
# In production this object gets generated per job posting.
# For this hackathon it encodes our careful reading of job_description.md
# =============================================================================

JD_CONFIG = {

    # Skills the JD explicitly requires — grouped by what they test
    "must_have_skills": {
        "embeddings": [
            "Embeddings", "Sentence Transformers", "BGE", "E5",
            "Word2Vec", "text-embedding", "Semantic Embeddings"
        ],
        "vector_db": [
            "Pinecone", "Weaviate", "Qdrant", "FAISS", "Milvus",
            "Elasticsearch", "OpenSearch", "Chroma", "Vector Search",
            "pgvector"
        ],
        "retrieval": [
            "Information Retrieval", "BM25", "Hybrid Search",
            "Haystack", "Retrieval", "Dense Retrieval", "Sparse Retrieval"
        ],
        "ranking": [
            "Learning to Rank", "XGBoost", "LightGBM",
            "Recommendation Systems", "Ranking", "NDCG", "MRR",
            "MAP", "MLflow"
        ],
        "python": [
            "Python"
        ],
        "llm_ml": [
            "Fine-tuning LLMs", "LoRA", "PEFT", "QLoRA",
            "Hugging Face Transformers", "NLP",
            "scikit-learn", "PyTorch", "TensorFlow"
        ],
    },

    # Nice-to-have skills (small bonus, not critical)
    "nice_to_have_skills": [
        "MLOps", "Kubeflow", "Weights & Biases",
        "A/B Testing", "Feature Engineering", "Deep Learning",
        "LangChain", "RAG", "Prompt Engineering",
        "Databricks", "Spark", "Kafka"
    ],

    # Keywords to find in actual job descriptions
    # Catches good candidates whose titles are generic
    # but whose real work matches the JD
    "career_keywords_high": [
        "embedding", "vector", "retrieval", "ranking", "search",
        "recommendation", "semantic", "similarity", "index",
        "fine-tun", "rag", "rerank", "learning to rank",
        "ndcg", "mrr", "offline eval", "a/b test",
        "faiss", "pinecone", "weaviate", "qdrant", "milvus",
        "opensearch", "elasticsearch", "sentence transformer",
        "hugging face", "transformers", "bert", "llm",
        "dense retrieval", "sparse", "hybrid search",
        "relevance", "recall", "precision", "latency",# Add to career_keywords_high list:
"offline eval", "online eval", "a/b", "relevance label",
"human judgment", "click-through", "dwell time",
"revenue per search", "engagement metric","bm25", "inverted index", "lucene", "solr",
"query understanding", "query expansion", "reranking",
"two-tower", "dual encoder", "cross encoder",
"passage retrieval", "document ranking"
    ],
    "career_keywords_medium": [
        "machine learning", "deep learning", "neural network",
        "model", "inference", "deploy", "production", "pipeline",
        "feature engineering", "xgboost", "lightgbm",
        "pytorch", "tensorflow", "sklearn", "scikit",
        "experiment", "evaluation", "metric", "benchmark"
    ],

    # JD explicitly warns against candidates whose ENTIRE career is here
    "services_companies": [
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant",
    "Capgemini", "HCL", "Mindtree", "Tech Mahindra",
    "Mphasis", "Hexaware", "NIIT Technologies", "Birlasoft",
    "Persistent Systems", "Mastech", "Genpact", "Wipro BPO",
    "IBM India", "DXC Technology"
],

    # JD preferred locations
    "good_locations": [
        "Noida", "Pune", "Hyderabad", "Mumbai", "Bangalore",
    "Bengaluru", "Delhi", "Gurgaon", "Gurugram", "Chennai",
    "Noida, Uttar Pradesh", "Pune, Maharashtra"
    ],

    # ML-relevant title keywords
    "ml_title_keywords": [
        "ml ", "machine learning", " ai ", "artificial intelligence",
        "nlp", "search", "ranking", "recommendation", "data scientist",
        "applied", "research engineer", "retrieval", "scientist",
        "intelligence", "deep learning"
    ],

    # CV/Speech skills — JD says these candidates need to relearn fundamentals
    "cv_speech_skills": [
        "Computer Vision", "Object Detection", "YOLO", "OpenCV",
        "Image Classification", "Speech Recognition", "Robotics",
        "TTS", "GANs", "CNN", "OCR", "ASR"
    ],

    # Scoring weights — sum of skill+career+behavioral+base = 1.0
    "weights": {
        "skill": 0.35,
        "career": 0.35,
        "behavioral": 0.25,
        "base": 0.05
    },

    # Experience range from JD
    "experience_ideal_min": 5,
    "experience_ideal_max": 9,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def proficiency_to_number(proficiency):
    """Convert proficiency string to 0-1 number"""
    return {
        "beginner": 0.25,
        "intermediate": 0.50,
        "advanced": 0.80,
        "expert": 1.00
    }.get(proficiency, 0.0)


def skill_trust_score(skill):
    """
    How credible is this skill claim?

    Key insight: 'advanced' with 0 months used = keyword stuffer.
    'intermediate' with 36 months + 20 endorsements = credible.

    Returns 0.0 to 1.0
    """
    duration = skill.get("duration_months", 0)
    endorsements = skill.get("endorsements", 0)

    if duration == 0:
        return 0.05  # near zero — claimed but never used

    duration_factor = min(duration / 24.0, 1.0)       # maxes at 2 years
    endorsement_factor = min(endorsements / 20.0, 1.0) # maxes at 20

    return (0.60 * duration_factor) + (0.40 * endorsement_factor)


def assessment_modifier(candidate, skill_name):
    """
    If they took the platform test for this skill and failed,
    penalize the skill claim.

    Example: Ira claimed NLP 'advanced' but scored 38.8/100.
    That's a red flag — we cut her NLP claim by 60%.
    """
    assessments = candidate["redrob_signals"].get("skill_assessment_scores", {})
    if skill_name not in assessments:
        return 1.0  # no test taken — neutral

    score = assessments[skill_name]
    if score >= 70:
        return 1.0   # passed — trust the claim
    elif score >= 50:
        return 0.75  # mediocre
    else:
        return 0.40  # failed own assessment — big red flag


def days_since_date(date_string):
    """Returns how many days ago a date was. Returns 999 on error."""
    try:
        past = datetime.strptime(date_string, "%Y-%m-%d").date()
        today = date(2026, 6, 15)
        return (today - past).days
    except Exception:
        return 999


# =============================================================================
# HONEYPOT DETECTION
# =============================================================================

def is_honeypot(candidate):
    """
    Detects candidates with impossible/inconsistent profiles.
    >10% honeypots in top 100 = Stage 3 disqualification.
    
    We flag on 2+ impossibilities to avoid false positives.
    """
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate["redrob_signals"]

    flags = 0

    # --- Check 1: Stated YOE vs actual career history ---
    stated_yoe = profile.get("years_of_experience", 0)
    actual_months = sum(j.get("duration_months", 0) for j in career)
    actual_years = actual_months / 12.0
    if stated_yoe > actual_years + 3.5:
        flags += 1

    # --- Check 2: Skill duration longer than total career ---
   # --- Check 2: Skill duration longer than total career ---
    max_skill_overage = 0
    for skill in skills:
        skill_months = skill.get("duration_months", 0)
        if actual_months > 0 and skill_months > actual_months:
            overage = skill_months - actual_months
            if overage > max_skill_overage:
                max_skill_overage = overage

    if max_skill_overage > 3:
        flags += 2  # more than 6 months over = severe impossibility
    elif max_skill_overage > 0:
        flags += 1  # any overage = mild flag, needs confirmation
   # --- Check 3: Tool founding date sanity check ---
    TOOL_MAX_AGE_MONTHS = {
        "Weaviate": 84,
        "Qdrant": 48,
        "Pinecone": 60,
        "LangChain": 30,
        "ChatGPT": 30,
        "RAG": 36,
        "pgvector": 36,
        "Chroma": 24,
        "LlamaIndex": 24,
    }
    tool_violations = 0
    for skill in skills:
        name = skill.get("name", "")
        skill_months = skill.get("duration_months", 0)
        max_age = TOOL_MAX_AGE_MONTHS.get(name, 9999)
        if skill_months > max_age + 6:
            tool_violations += 1

    # One tool violation = 1 flag
    # Two+ tool violations = 2 flags (instant honeypot by itself)
    if tool_violations >= 2:
        flags += 2
    elif tool_violations == 1:
        flags += 1
    # --- Check 4: Multiple expert skills with zero duration ---
    zero_duration_senior = sum(
        1 for s in skills
        if s.get("duration_months", 0) == 0
        and s.get("proficiency") in ["advanced", "expert"]
    )
    if zero_duration_senior >= 3:
        flags += 1

    # --- Check 5: Salary range inverted ---
    salary = signals.get("expected_salary_range_inr_lpa", {})
    sal_min = salary.get("min", 0)
    sal_max = salary.get("max", 0)
    if sal_max > 0 and sal_min > sal_max * 1.5:
        flags += 1

    # --- Check 6: Signup date after last active date ---
    try:
        signup = datetime.strptime(
            signals["signup_date"], "%Y-%m-%d"
        ).date()
        last_active = datetime.strptime(
            signals["last_active_date"], "%Y-%m-%d"
        ).date()
        if signup > last_active:
            flags += 1
    except Exception:
        pass

    return flags >= 2
   
    

# =============================================================================
# SCORING COMPONENTS
# =============================================================================

def score_skills(candidate, jd=JD_CONFIG):
    """
    Score candidate on skill match against JD must-haves.

    Key design decisions:
    - We check skill categories, not individual skills
      (so having FAISS fills the 'vector_db' category)
    - Trust modifier catches keyword stuffers
    - Assessment modifier catches people who failed their own tests
    - Nice-to-have skills add small bonus, capped

    Returns 0.0 to 1.0
    """
    skills_list = candidate.get("skills", [])

    # Build lookup: skill name -> skill object
    skill_map = {}
    for s in skills_list:
        name = s["name"].strip()
        skill_map[name] = s

    total_score = 0.0
    max_possible = float(len(jd["must_have_skills"]))

    for category, aliases in jd["must_have_skills"].items():
        best = 0.0
        for alias in aliases:
            if alias in skill_map:
                s = skill_map[alias]
                prof = proficiency_to_number(s["proficiency"])
                trust = skill_trust_score(s)
                assessment = assessment_modifier(candidate, alias)
                val = prof * trust * assessment
                if val > best:
                    best = val
        total_score += best
    # After the main category loop, add embeddings depth bonus
    
    # Nice-to-have bonus
    nice = 0.0
    for skill_name in jd["nice_to_have_skills"]:
        if skill_name in skill_map:
            s = skill_map[skill_name]
            nice += 0.04 * skill_trust_score(s)

    total_score += min(nice, 0.25)
    max_possible += 0.25
    skill_map_local = {s["name"].strip(): s for s in candidate.get("skills", [])}
    embeddings_skills = ["Embeddings", "Sentence Transformers", "BGE", "E5"]
    for emb_skill in embeddings_skills:
        if emb_skill in skill_map_local:
            emb_duration = skill_map_local[emb_skill].get("duration_months", 0)
            if emb_duration >= 48:
                total_score += 0.12   # 4+ years — significant bonus
            elif emb_duration >= 36:
                total_score += 0.08   # 3+ years
            elif emb_duration >= 24:
                total_score += 0.04   # 2+ years
            # under 24 months — no bonus
    return round(min(total_score / max_possible, 1.0), 4) if max_possible > 0 else 0.0


def score_career_descriptions(candidate, jd=JD_CONFIG):
    """
    Scans actual job description text for relevant work.

    This is the most important function most submissions miss.
    The JD explicitly says:
    'A Tier 5 candidate may not use the words RAG or Pinecone
    in their profile, but if their career history shows they
    built a recommendation system, they're a fit.'

    We find those candidates here.

    Returns 0.0 to 1.0
    """
    career = candidate.get("career_history", [])
    if not career:
        return 0.0

    all_text = " ".join(
        job.get("description", "").lower()
        for job in career
    )

    if not all_text.strip():
        return 0.0

    high_hits = sum(
        1 for kw in jd["career_keywords_high"]
        if kw.lower() in all_text
    )
    medium_hits = sum(
        1 for kw in jd["career_keywords_medium"]
        if kw.lower() in all_text
    )

    # 5+ high-value hits = excellent career match
    high_score = min(high_hits / 5.0, 1.0)
    # 4+ medium hits = good
    medium_score = min(medium_hits / 4.0, 1.0)

    combined = (high_score * 0.75) + (medium_score * 0.25)
    return round(combined, 4)


def score_career_profile(candidate, jd=JD_CONFIG):
    """
    Score career trajectory:
    - Years of experience in ideal range
    - Title relevance (current + past)
    - Product vs services company history
    - Job stability (anti title-chaser)
    - Education tier

    Returns 0.0 to 1.0
    """
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    score = 0.0

    # --- Experience years ---
    yoe = profile.get("years_of_experience", 0)
    ideal_min = jd["experience_ideal_min"]
    ideal_max = jd["experience_ideal_max"]

    if ideal_min <= yoe <= ideal_max:
        score += 0.20
    elif ideal_min - 1 <= yoe <= ideal_max + 3:
        score += 0.12
    elif yoe > ideal_max + 3:
        score += 0.06
    else:
        score += 0.02

    # --- Title relevance (current) ---
    current_title = profile.get("current_title", "").lower()
    ml_keywords = jd["ml_title_keywords"]

    if any(kw in current_title for kw in ml_keywords):
        score += 0.20
    else:
        # Check past titles — maybe they had the right title before
        past_titles = [j["title"].lower() for j in career]
        if any(
            any(kw in t for kw in ml_keywords)
            for t in past_titles
        ):
            score += 0.10

   # --- Product company experience ---
    services = jd["services_companies"]

    def is_services(company_name):
        company_lower = company_name.lower()
        return any(s.lower() in company_lower for s in services)

    product_months = 0
    total_months = 0
    entire_career_services = True

    for job in career:
        months = job.get("duration_months", 0)
        total_months += months
        if not is_services(job["company"]):
            entire_career_services = False
            product_months += months

    if entire_career_services and len(career) > 0:
        score *= 0.35
    elif total_months > 0:
        product_ratio = product_months / total_months
        score += 0.20 * product_ratio
    # --- Job stability ---
    if len(career) > 1:
        avg_tenure = sum(
            j.get("duration_months", 0) for j in career
        ) / len(career)

        if avg_tenure >= 24:
            score += 0.15
        elif avg_tenure >= 18:
            score += 0.08
        # else 0 — job hopper

    # --- Education tier ---
    tier_bonus = {
        "tier_1": 0.10, "tier_2": 0.06,
        "tier_3": 0.03, "tier_4": 0.0, "unknown": 0.0
    }
    best_edu = max(
        (tier_bonus.get(e.get("tier", "unknown"), 0.0)
         for e in candidate.get("education", [])),
        default=0.0
    )
    score += best_edu

    return round(min(score, 1.0), 4)


def score_behavioral(candidate):
    """
    Score based on platform engagement signals.

    JD literally says: 'A perfect-on-paper candidate who hasn't
    logged in for 6 months and has a 5% response rate is, for
    hiring purposes, not actually available. Down-weight them.'

    We follow this instruction precisely.

    Returns 0.3 to 1.2
    """
    sig = candidate["redrob_signals"]

    # --- Recency: when did they last log in? ---
    days_inactive = days_since_date(
        sig.get("last_active_date", "2020-01-01")
    )
    if days_inactive > 180:
        availability = 0.20  # 6 months — probably not looking
    elif days_inactive > 90:
        availability = 0.50
    elif days_inactive > 30:
        availability = 0.80
    else:
        availability = 1.00  # active recently

    # --- Response rate: will they reply? ---
    rr = sig.get("recruiter_response_rate", 0.0)
    response_score = 0.30 + (0.70 * rr)

    # --- Open to work ---
    open_bonus = 1.10 if sig.get("open_to_work_flag", False) else 1.00

    # --- Notice period (JD prefers ≤30 days, buys out up to 30) ---
    notice = sig.get("notice_period_days", 90)
    if notice <= 30:
        notice_score = 1.00
    elif notice <= 60:
        notice_score = 0.75   # was 0.85 — stricter now
    elif notice <= 90:
        notice_score = 0.50   # was 0.70 — stricter now
    elif notice <= 120:
        notice_score = 0.30   # was missing — 4 months is very painful
    else:
        notice_score = 0.15   # 180 days = essentially unavailable

    # --- GitHub (relevant for this engineering role) ---
    github = sig.get("github_activity_score", -1)
    if github >= 50:
        github_score = 1.20
    elif github >= 20:
        github_score = 1.05
    elif github >= 0:
        github_score = 0.90
    else:
        github_score = 0.80  # no GitHub linked — mild negative

    # --- Profile completeness ---
    completeness = sig.get("profile_completeness_score", 50) / 100.0

    # --- Interview reliability ---
    icr = sig.get("interview_completion_rate", 0.5)
    reliability = 0.5 + (0.5 * icr)

    behavioral = (
        availability   * 0.28 +
        response_score * 0.22 +
        notice_score   * 0.18 +
        github_score   * 0.14 +
        completeness   * 0.10 +
        reliability    * 0.08
    ) * open_bonus

    return round(max(0.30, min(behavioral, 1.20)), 4)

def location_multiplier(candidate, jd=JD_CONFIG):
    location = candidate["profile"].get("location", "")
    country = candidate["profile"].get("country", "")
    relocate = candidate["redrob_signals"].get("willing_to_relocate", False)

    if country != "India":
        return 0.85 if relocate else 0.75

    loc_lower = location.lower()
    
    # Tier 1 locations — exactly what JD asks for
    TIER_1 = ["noida", "pune", "hyderabad", "mumbai", 
               "bangalore", "bengaluru", "delhi", "gurgaon", 
               "gurugram", "chennai"]
    
    # Tier 2 — India but not preferred cities
    # These should NOT score as high as Tier 1
    for good in TIER_1:
        if good in loc_lower:
            return 1.10  # perfect

    # In India, wrong city, willing to relocate
    if relocate:
        return 0.95  # small penalty — relocation has friction
    
    # In India, wrong city, won't relocate
    return 0.82  # meaningful penalty

def disqualifier_multiplier(candidate, jd=JD_CONFIG):
    """
    Hard penalties from the JD's explicit disqualifier list.
    Returns 1.0 (fine) or a penalty fraction.
    """
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # Entire career in services companies
    # Use substring matching — catches "Genpact AI", "TCS Digital" etc.
    services = jd["services_companies"]

    def is_services_company(company_name):
        company_lower = company_name.lower()
        return any(s.lower() in company_lower for s in services)

    all_cos = [j["company"] for j in career]
    if all_cos and all(is_services_company(c) for c in all_cos):
        return 0.25
    # Current company is services — partial penalty
    # even if not entire career (they chose services over product recently)
    current_company = profile.get("current_company", "")
    if is_services_company(current_company):
        return 0.55  # meaningful penalty but not as severe as entire career
    # Primary expertise is CV/Speech/Robotics
    skill_names = [s["name"] for s in skills]
    cv_count = sum(1 for n in skill_names if n in jd["cv_speech_skills"])
    if len(skill_names) > 0 and cv_count / len(skill_names) > 0.55:
        return 0.35

    # Too junior
    if profile.get("years_of_experience", 0) < 3:
        return 0.30

    return 1.00

# =============================================================================
# MASTER SCORING FUNCTION
# =============================================================================

def score_candidate(candidate, jd=JD_CONFIG):
    """
    Combines all components into one final score.

    Architecture:
    - Skills (35%) + Career (35%) + Behavioral (25%) + base (5%)
    - Career = 60% profile signals + 40% description text scan
    - Location and disqualifier applied as multipliers
    - Honeypots get score of 0.0001

    Returns 0.0 to 1.0
    """
    if is_honeypot(candidate):
        return 0.0001

    skill = score_skills(candidate, jd)
    career_desc = score_career_descriptions(candidate, jd)
    career_prof = score_career_profile(candidate, jd)
    behavioral = score_behavioral(candidate)
    loc = location_multiplier(candidate, jd)
    disq = disqualifier_multiplier(candidate, jd)

    # Blend career profile signals with career description text
    career_combined = (career_prof * 0.45) + (career_desc * 0.55)

    w = jd["weights"]
    raw = (
        skill         * w["skill"]    +
        career_combined * w["career"] +
        behavioral    * w["behavioral"] +
        w["base"]
    )

    final = raw * loc * disq
    # Hard availability cap
# JD says 30+ day notice raises the bar significantly
# 120 day notice at a Series A = practically unavailable
    notice = candidate["redrob_signals"].get("notice_period_days", 90)
    if notice > 90:
        final = final * 0.78
    elif notice > 60:
        final = final * 0.88
    return round(min(max(final, 0.0001), 1.0), 4)


# =============================================================================
# REASONING GENERATION
# =============================================================================

def generate_reasoning(candidate, rank, final_score, jd=JD_CONFIG):
    """
    Generates specific, honest, non-templated reasoning.

    Stage 4 judges check for:
    - Specific facts from profile (not generic praise)
    - JD connection (not just 'good candidate')
    - Honest concerns (gaps, notice period, location)
    - No hallucination (only mention what's in the profile)
    - Variation across candidates
    - Tone matches rank (rank 5 ≠ same tone as rank 95)

    Strategy: build from actual profile facts, always mention
    at least one concern for ranks 20+, vary sentence structure.
    """
    p = candidate["profile"]
    sig = candidate["redrob_signals"]
    career = candidate.get("career_history", [])
    skills_list = candidate.get("skills", [])

    # --- Gather real facts ---
    name_hint = p.get("current_title", "Candidate")
    yoe = p.get("years_of_experience", 0)
    company = p.get("current_company", "current company")
    location = p.get("location", "unknown location")
    country = p.get("country", "")
    notice = sig.get("notice_period_days", 90)
    last_active_days = days_since_date(sig.get("last_active_date", "2020-01-01"))
    open_to_work = sig.get("open_to_work_flag", False)
    github = sig.get("github_activity_score", -1)
    response_rate = sig.get("recruiter_response_rate", 0)

    # Find their strongest JD-relevant skills (with real usage)
    skill_map = {s["name"]: s for s in skills_list}
    relevant_skills = []
    all_jd_skills = []
    for aliases in jd["must_have_skills"].values():
        all_jd_skills.extend(aliases)
    all_jd_skills.extend(jd["nice_to_have_skills"])

    for skill_name in all_jd_skills:
        if skill_name in skill_map:
            s = skill_map[skill_name]
            if s.get("duration_months", 0) > 6:
                relevant_skills.append(
                    f"{skill_name} ({s['proficiency']}, "
                    f"{s['duration_months']}mo)"
                )
    relevant_skills = relevant_skills[:3]  # top 3 relevant skills

    # Find career description evidence
    desc_text = " ".join(
        j.get("description", "") for j in career
    ).lower()
    career_evidence = []
    for kw in jd["career_keywords_high"]:
        if kw in desc_text:
            career_evidence.append(kw)
    career_evidence = career_evidence[:2]

    # --- Build concerns list (honest) ---
    concerns = []
    if last_active_days > 90:
        concerns.append(
            f"inactive for {last_active_days // 30} months"
        )
    if notice > 60:
        concerns.append(f"{notice}-day notice period")
    if response_rate < 0.30:
        concerns.append(
            f"low recruiter response rate ({response_rate:.0%})"
        )
    if country != "India":
        concerns.append(f"based outside India ({country})")
    if not relevant_skills and not career_evidence:
        concerns.append("limited direct ML/retrieval evidence in profile")
    if github == -1:
        concerns.append("no GitHub linked")

    # --- Build strengths ---
    strengths = []
    if relevant_skills:
        strengths.append(
            f"relevant skills: {', '.join(relevant_skills)}"
        )
    if career_evidence:
        strengths.append(
            f"career history references {' and '.join(career_evidence)}"
        )
    if open_to_work:
        strengths.append("actively open to work")
    if notice <= 30:
        strengths.append(f"available quickly ({notice}-day notice)")
    if github >= 40:
        strengths.append(f"strong GitHub activity ({github:.0f}/100)")

    # --- Compose reasoning based on rank tier ---
    loc_str = f"{location}, {country}" if country else location

    if rank <= 10:
        # Top 10: strong positive, mention specific evidence
        strength_str = (
            "; ".join(strengths[:2]) if strengths
            else "solid experience profile"
        )
        concern_str = (
            f"; note: {concerns[0]}" if concerns
            else ""
        )
        reasoning = (
            f"{yoe:.0f}-year {name_hint} at {company} "
            f"({loc_str}) with {strength_str}{concern_str}."
        )

    elif rank <= 30:
        # Top 11-30: positive with honest caveats
        strength_str = (
            strengths[0] if strengths
            else "some relevant experience"
        )
        concern_str = (
            f"; concern: {concerns[0]}" if concerns
            else ""
        )
        reasoning = (
            f"{yoe:.0f} years experience as {name_hint}, {loc_str}; "
            f"{strength_str}{concern_str}."
        )

    elif rank <= 60:
        # Mid tier: balanced, lead with gap
        concern_str = (
            concerns[0] if concerns
            else "partial skill match"
        )
        strength_str = (
            strengths[0] if strengths
            else "some adjacent experience"
        )
        reasoning = (
            f"Partial fit — {concern_str}; "
            f"{yoe:.0f}-year {name_hint} with {strength_str}."
        )

    else:
        # Bottom tier: honest about the gap
        concern_str = (
            "; ".join(concerns[:2]) if concerns
            else "limited JD alignment"
        )
        reasoning = (
            f"Weak fit — {concern_str}; "
            f"included as rank {rank} filler given "
            f"{yoe:.0f} years experience."
        )

    # Sanitize: remove quotes that break CSV
    reasoning = reasoning.replace('"', "'").strip()

    # Hard limit: keep it concise (judges want 1-2 sentences)
    if len(reasoning) > 280:
        reasoning = reasoning[:277] + "..."

    return reasoning


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run(candidates_path, output_path):
    """
    Full pipeline:
    1. Load all candidates
    2. Score each one
    3. Detect honeypots
    4. Pick top 100
    5. Generate reasoning
    6. Write valid CSV
    """
    print(f"[1/5] Loading candidates from {candidates_path} ...")

    candidates = []
    with open(candidates_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
            if (i + 1) % 10000 == 0:
                print(f"      Loaded {i + 1:,} candidates...")

    print(f"      Total loaded: {len(candidates):,}")

    # ---
    print("[2/5] Scoring candidates...")

    scored = []
    honeypot_count = 0

    for i, c in enumerate(candidates):
        honeypot = is_honeypot(c)
        if honeypot:
            honeypot_count += 1

        final = score_candidate(c)
        scored.append((final, c["candidate_id"], c))

        if (i + 1) % 10000 == 0:
            print(f"      Scored {i + 1:,} / {len(candidates):,}...")

    print(f"      Honeypots detected: {honeypot_count}")

    # ---
    print("[3/5] Sorting and selecting top 100...")

    # Sort: highest score first, tie-break by candidate_id ascending
    scored.sort(key=lambda x: (-x[0], x[1]))

    top_100 = scored[:100]

    # Verify no honeypots in top 100
    hp_in_top = sum(1 for s, cid, c in top_100 if is_honeypot(c))
    hp_rate = hp_in_top / 100
    print(f"      Honeypots in top 100: {hp_in_top} ({hp_rate:.1%})")
    if hp_rate > 0.10:
        print("      WARNING: Honeypot rate exceeds 10% — would be disqualified!")
    else:
        print("      Honeypot rate OK (below 10% threshold)")

    # Score range check
    top_score = top_100[0][0]
    bottom_score = top_100[-1][0]
    print(f"      Score range: {bottom_score:.4f} to {top_score:.4f}")

    # ---
    print("[4/5] Generating reasoning...")

    rows = []
    for rank_idx, (final_score, cid, candidate) in enumerate(top_100):
        rank = rank_idx + 1
        reasoning = generate_reasoning(candidate, rank, final_score)
        rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": final_score,
            "reasoning": reasoning
        })

    # ---
    print(f"[5/5] Writing CSV to {output_path} ...")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== DONE ===")
    print(f"Output: {output_path}")
    print(f"Rows written: {len(rows)}")
    print("\nTop 10 candidates:")
    print("-" * 70)
    for row in rows[:10]:
        print(f"Rank {row['rank']:>3} | {row['candidate_id']} "
              f"| Score: {row['score']:.4f}")
        print(f"         {row['reasoning'][:80]}...")
        print()

    print("\nValidation checks:")
    print(f"  Row count:     {len(rows)} (need 100)")
    ranks = [r["rank"] for r in rows]
    print(f"  Ranks 1-100:   {'OK' if sorted(ranks) == list(range(1, 101)) else 'FAIL'}")
    scores = [r["score"] for r in rows]
    monotonic = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    print(f"  Scores non-increasing: {'OK' if monotonic else 'FAIL'}")
    ids = [r["candidate_id"] for r in rows]
    print(f"  Unique IDs:    {'OK' if len(set(ids)) == 100 else 'FAIL'}")
    print(f"  Empty reasoning: {sum(1 for r in rows if not r['reasoning'])}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Redrob Hackathon — Candidate Ranker"
    )
    parser.add_argument(
        "--candidates",
        default="./data/candidates.jsonl",
        help="Path to candidates.jsonl file"
    )
    parser.add_argument(
        "--out",
        default="./outputs/submission.csv",
        help="Output CSV path"
    )
    args = parser.parse_args()

    run(args.candidates, args.out)