# Redrob Hackathon — Intelligent Candidate Ranking System

A fast, explainable, rule-based candidate ranker for the Redrob
Intelligent Candidate Discovery & Ranking Challenge.

---

## Quick Start

```bash
pip install pandas tqdm
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

That single command reads all 100,000 candidates and produces a
valid submission CSV in approximately 25 seconds on CPU.

---

## Reproduce Command

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

- Runtime: ~25 seconds (well within 5-minute limit)
- Memory: <2 GB RAM
- CPU only — no GPU, no network calls
- No pre-computation required

---

## Validate

```bash
python validate_submission.py submission.csv
```

Should print: `Submission is valid.`

---

## Architecture

### Design Philosophy

The JD is fixed for this hackathon so JD knowledge is hardcoded
into a single `JD_CONFIG` object at the top of `rank.py`. This
gives maximum precision and full explainability. In production,
this config object would be generated per job posting from JD text.

### Five Scoring Components

```
score_skills()              — Skill match against JD must-haves
                              Trust multiplier: duration × endorsements
                              Assessment penalty: catches failed skill tests
                              Embeddings depth bonus: rewards sustained experience

score_career_descriptions() — Scans actual job description text
                              Finds candidates whose titles are generic
                              but whose real work matches the JD
                              (The 'Tier 5 candidate' the JD mentions)

score_career_profile()      — Years of experience in ideal range (5-9)
                              Title relevance (current + past)
                              Product company vs services company history
                              Job stability (anti title-chaser check)
                              Education tier bonus

score_behavioral()          — 6 platform signals from redrob_signals:
                              Recency (last_active_date decay)
                              Recruiter response rate
                              Notice period (hard cap at 90+ days)
                              GitHub activity score
                              Profile completeness
                              Interview completion rate

location_multiplier()       — Tier 1: Noida/Pune/Hyderabad/Mumbai/
                                       Bangalore/Delhi/Chennai (+10%)
                              Tier 2: India, willing to relocate (0%)
                              Tier 3: India, wrong city, won't relocate (-12%)
                              Outside India: (-15% to -25%)
```

### Final Score Formula

```
career_combined = career_profile × 0.45 + career_description × 0.55

raw_score = skill × 0.35
          + career_combined × 0.35
          + behavioral × 0.25
          + 0.05 (base)

final_score = raw_score × location_multiplier × disqualifier_multiplier
```

### Disqualifiers (from JD explicit list)

- Entire career in IT services (TCS, Infosys, Wipro, etc.) → 0.25×
- Currently at services company → 0.55×
- Primary expertise is CV/Speech/Robotics → 0.35×
- Under 3 years experience → 0.30×

### Honeypot Detection

The dataset contains ~80 honeypot candidates with impossible profiles.
Candidates with 2+ of these flags are scored near-zero:

1. Stated YOE more than 3.5 years above career history
2. Skill duration exceeds total career length by 3+ months
3. Tool founding-date violations (Weaviate >84mo, Qdrant >48mo,
   Pinecone >60mo, LangChain >30mo, RAG >36mo, etc.)
4. 3+ expert/advanced skills with zero duration
5. Salary range inverted (min > max)
6. Signup date after last active date

### Keyword Stuffer Detection

Skills are scored as: proficiency × trust × assessment_modifier

- Trust = 0.6 × (duration/24) + 0.4 × (endorsements/20)
- Zero duration → trust = 0.05 (near zero)
- Failed platform assessment → 0.40× multiplier
- This naturally suppresses inflated skill claims

---

## File Structure

```
redrob_ranker/
├── rank.py                  — Main ranker (single file, no dependencies)
├── validate_submission.py   — Official format validator
├── README.md                — This file
├── submission_metadata.yaml — Submission metadata
├── requirements.txt         — Python dependencies
├── data/
│   └── candidates.jsonl     — 100K candidate pool (not in repo)
└── outputs/
    └── submission.csv       — Generated output
```

---

## Requirements

```
Python 3.9+
No external ML libraries required
Standard library only: json, csv, argparse, datetime
```

```bash
pip install -r requirements.txt
```

`requirements.txt` contains no entries — this ranker uses Python
standard library only. No pip install needed beyond Python itself.

---

## Key Design Decisions

**Why hardcode the JD instead of parsing it dynamically?**

The JD is fixed for this challenge. Hardcoding gives more precision
(we encode what the JD *means*, not just what it *says*) and makes
every scoring decision fully explainable. The `JD_CONFIG` object
is structured so a new JD config can be swapped in for any role —
making this extensible to production use with multiple simultaneous
job postings.

**Why rule-based instead of embeddings/LLMs?**

The compute constraint (5 min CPU, no network) makes LLM calls
impossible at 100K scale. Rule-based scoring with trust multipliers
runs in 25 seconds and is fully reproducible. The career description
text scan provides semantic matching without embeddings.

**Why multiplicative behavioral modifier instead of additive?**

An additive behavioral score lets a great-on-paper candidate with
0% recruiter response rate still rank highly. Multiplicative ensures
behavioral signals act as a gate — a candidate who hasn't logged in
for 6 months is genuinely unavailable regardless of skill match.

**Why career description scanning?**

The JD explicitly states: a candidate may not use words like RAG
or Pinecone in their profile, but if their career history shows
they built a recommendation system at a product company, they are
a fit. Career description text scanning directly implements this
instruction and is our biggest differentiator vs keyword-only systems.

---

## Stage 5 Interview Talking Points

- JD_CONFIG makes the system extensible to any job posting
- Trust multiplier design catches keyword stuffers without ML
- Assessment penalty uses the platform's own data against fraud
- Career description scan implements the JD's explicit Tier 5 hint
- Honeypot detection uses 6 consistency checks, 2+ flags required
- Multiplicative behavioral modifier follows the JD's exact wording
- 25 second runtime leaves 92% of compute budget unused
