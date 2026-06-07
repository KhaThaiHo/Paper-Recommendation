# ============================================================
# prompt_builder.py — Prompt engineering for journal extraction
# ============================================================
#
# Design principles:
#   1. Categories (structured) are used as strong prior → reduces hallucination
#   2. Aims & Scope is parsed for evidence phrases (verbatim short spans)
#   3. Output is strict JSON → easy to parse, no markdown wrapper
#   4. Few-shot examples calibrate label granularity
# ============================================================

SYSTEM_PROMPT = """You are a scientific metadata extraction specialist.
Your task is to extract two structured fields from a journal's metadata:
  • scientific_domains  — broad scientific/medical disciplines the journal belongs to
  • research_focuses    — specific research topics, study objects, or application areas the journal covers

Extraction rules:
1. Use the Categories field as the PRIMARY source for scientific_domains (they are curated taxonomy labels).
2. Use the Aims & Scope text as the PRIMARY source for research_focuses AND as evidence for both fields.
3. For each domain/focus you extract, capture 1–3 SHORT evidence phrases (≤10 words each) directly from the Aims & Scope text that justify the label.
4. Labels must be noun phrases in English, title-cased (e.g. "Clinical Neurology", not "clinical neurology" or "studies on neurology").
5. scientific_domains should be broad disciplines (5–15 words max per label).
6. research_focuses should be specific topics or methodologies (can be more granular).
7. Aim for 2–6 scientific_domains and 3–10 research_focuses depending on journal breadth.
8. Do NOT hallucinate domains not supported by either Categories or Aims text.
9. Output ONLY a valid JSON object — no markdown, no explanation, no preamble.

Output schema (strict):
{
  "scientific_domains": ["Domain1", "Domain2"],
  "scientific_domains_evidence": {
    "Domain1": ["evidence phrase 1", "evidence phrase 2"],
    "Domain2": ["evidence phrase 1"]
  },
  "research_focuses": ["Focus1", "Focus2", "Focus3"],
  "research_focuses_evidence": {
    "Focus1": ["evidence phrase 1", "evidence phrase 2"],
    "Focus2": ["evidence phrase 1"]
  }
}"""


FEW_SHOT_EXAMPLES = [
    {
        "journal": "Therapeutic Advances in Neurological Disorders",
        "categories": "Neurology (clinical), Neurology, Pharmacology",
        "aims": (
            "Therapeutic Advances in Neurological Disorders delivers the highest quality "
            "peer reviewed original research articles, reviews, and scholarly comment on "
            "pioneering efforts and innovative studies in the medical treatment of "
            "neurological conditions. The journal has a strong clinical and pharmacological "
            "focus and is aimed at an international audience of clinicians and researchers "
            "in neurology and related disciplines, providing an online forum for rapid "
            "dissemination of recent research and perspectives in this area."
        ),
        "output": """{
  "scientific_domains": ["Neurology", "Clinical Neurology", "Pharmacology"],
  "scientific_domains_evidence": {
    "Neurology": ["neurological conditions", "clinicians and researchers in neurology"],
    "Clinical Neurology": ["medical treatment of neurological conditions", "strong clinical focus"],
    "Pharmacology": ["pharmacological focus", "innovative studies"]
  },
  "research_focuses": [
    "Pharmacotherapy for Neurological Disorders",
    "Clinical Neurology Treatment",
    "Neurological Disease Management",
    "Translational Neuroscience"
  ],
  "research_focuses_evidence": {
    "Pharmacotherapy for Neurological Disorders": ["pharmacological focus", "medical treatment of neurological conditions"],
    "Clinical Neurology Treatment": ["strong clinical and pharmacological focus", "clinicians and researchers in neurology"],
    "Neurological Disease Management": ["innovative studies in the medical treatment", "neurological conditions"],
    "Translational Neuroscience": ["pioneering efforts", "recent research and perspectives"]
  }
}"""
    },
    {
        "journal": "Journal of Public Health Informatics",
        "categories": "Public Health, Health Informatics, Epidemiology",
        "aims": (
            "The Journal of Public Health Informatics focuses on the application of "
            "information technology and data science to improve public health outcomes. "
            "It covers disease surveillance systems, digital epidemiology, health data "
            "analytics, and the use of electronic health records in population health "
            "monitoring and intervention design."
        ),
        "output": """{
  "scientific_domains": ["Public Health", "Health Informatics", "Epidemiology", "Data Science"],
  "scientific_domains_evidence": {
    "Public Health": ["improve public health outcomes", "population health monitoring"],
    "Health Informatics": ["information technology and data science", "electronic health records"],
    "Epidemiology": ["disease surveillance systems", "digital epidemiology"],
    "Data Science": ["data science to improve public health", "health data analytics"]
  },
  "research_focuses": [
    "Disease Surveillance Systems",
    "Digital Epidemiology",
    "Health Data Analytics",
    "Electronic Health Records",
    "Population Health Monitoring",
    "Public Health Intervention Design"
  ],
  "research_focuses_evidence": {
    "Disease Surveillance Systems": ["disease surveillance systems"],
    "Digital Epidemiology": ["digital epidemiology"],
    "Health Data Analytics": ["health data analytics"],
    "Electronic Health Records": ["use of electronic health records"],
    "Population Health Monitoring": ["population health monitoring"],
    "Public Health Intervention Design": ["intervention design", "improve public health outcomes"]
  }
}"""
    }
]


def build_prompt(journal_name: str, categories: str, aims: str) -> list[dict]:
    """
    Build the messages list for Ollama chat API.
    Uses few-shot examples embedded in the user turn for models
    that don't support a 'system' role well (e.g. some Ollama backends).
    Returns: list of message dicts {"role": ..., "content": ...}
    """
    # Format few-shot block
    few_shot_block = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        few_shot_block += f"""
--- EXAMPLE {i} ---
Journal: {ex['journal']}
Categories: {ex['categories']}
Aims & Scope: {ex['aims']}
Output:
{ex['output']}
"""

    user_content = f"""Below are {len(FEW_SHOT_EXAMPLES)} worked examples showing the expected extraction format:
{few_shot_block}
--- YOUR TASK ---
Now extract for this journal. Follow the same format exactly.

Journal: {journal_name}
Categories: {categories if categories else "Not provided"}
Aims & Scope: {aims if aims else "Not provided"}

Output:"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def build_prompt_no_system(journal_name: str, categories: str, aims: str) -> list[dict]:
    """
    Fallback for models that don't support system role.
    Embeds the system instructions into the first user message.
    """
    few_shot_block = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        few_shot_block += f"""
--- EXAMPLE {i} ---
Journal: {ex['journal']}
Categories: {ex['categories']}
Aims & Scope: {ex['aims']}
Output:
{ex['output']}
"""

    user_content = f"""{SYSTEM_PROMPT}

Below are {len(FEW_SHOT_EXAMPLES)} worked examples:
{few_shot_block}
--- YOUR TASK ---
Journal: {journal_name}
Categories: {categories if categories else "Not provided"}
Aims & Scope: {aims if aims else "Not provided"}

Output:"""

    return [{"role": "user", "content": user_content}]