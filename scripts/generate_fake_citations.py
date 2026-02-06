#!/usr/bin/env python3
"""
Stage 2: Generate fake citations using Ollama (GAN-like generation).
Creates hallucinated antibody training data for publication verification.

Strategies:
1. Perturbation: Modify real citations (wrong year, swapped authors)
2. Generation: Use Ollama to generate plausible fake citations
3. Hybrid: Combine real elements in impossible ways
"""

import json
import requests
import random
import string
from pathlib import Path
from datetime import datetime

DATA_DIR = Path.home() / ".antigence" / "data" / "citations"
OLLAMA_URL = "http://localhost:11434/api/generate"

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ============ PERTURBATION METHODS ============

def perturb_year(citation):
    """Create fake by changing year."""
    fake = citation.copy()
    real_year = fake.get('year', 2020)

    # Make impossible year
    options = [
        real_year + random.randint(5, 20),  # Future year
        real_year - random.randint(50, 100),  # Too old
        random.randint(2025, 2030),  # Future
    ]
    fake['year'] = random.choice(options)
    fake['label'] = 'fake'
    fake['fake_type'] = 'wrong_year'
    fake['source'] = 'perturbation'
    return fake

def perturb_authors(citation):
    """Create fake by swapping/modifying authors."""
    fake = citation.copy()

    # Generate fake author names
    first_names = ["James", "Sarah", "Michael", "Emily", "David", "Jennifer", "Robert", "Lisa",
                   "William", "Maria", "John", "Anna", "Richard", "Susan", "Thomas", "Karen",
                   "Wei", "Yuki", "Hans", "Pierre", "Alessandro", "Sven"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
                  "Martinez", "Anderson", "Taylor", "Thomas", "Moore", "Jackson", "Martin",
                  "Wang", "Zhang", "Chen", "Kumar", "Singh", "Mueller", "Schmidt"]

    num_authors = random.randint(1, 4)
    fake['authors'] = [f"{random.choice(first_names)} {random.choice(last_names)}"
                       for _ in range(num_authors)]
    fake['label'] = 'fake'
    fake['fake_type'] = 'fake_authors'
    fake['source'] = 'perturbation'
    return fake

def perturb_journal(citation):
    """Create fake by changing journal to non-existent one."""
    fake = citation.copy()

    fake_journals = [
        "Journal of Advanced Computational Intelligence",
        "International Review of Machine Learning Research",
        "Quantum Computing and AI Systems",
        "Neural Processing Letters International",
        "Frontiers in Artificial General Intelligence",
        "Journal of Superintelligent Systems",
        "Advanced Neural Architecture Review",
        "International Journal of Deep Learning Applications",
        "Cognitive Computing Quarterly",
        "Journal of Autonomous AI Systems",
    ]

    fake['journal'] = random.choice(fake_journals)
    fake['label'] = 'fake'
    fake['fake_type'] = 'fake_journal'
    fake['source'] = 'perturbation'
    return fake

def perturb_doi(citation):
    """Create fake DOI."""
    fake = citation.copy()

    # Generate fake DOI patterns
    fake_prefixes = ["10.9999", "10.8888", "10.7777", "10.0000"]
    fake_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))

    fake['doi'] = f"{random.choice(fake_prefixes)}/fake.{fake_suffix}"
    fake['label'] = 'fake'
    fake['fake_type'] = 'fake_doi'
    fake['source'] = 'perturbation'
    return fake

def create_hybrid(citations):
    """Combine elements from different citations."""
    if len(citations) < 3:
        return None

    c1, c2, c3 = random.sample(citations, 3)

    fake = {
        'title': c1.get('title', ''),
        'authors': c2.get('authors', []),
        'year': c3.get('year', 2020),
        'journal': c1.get('journal', ''),
        'doi': '',
        'abstract': c2.get('abstract', ''),
        'label': 'fake',
        'fake_type': 'hybrid',
        'source': 'hybrid'
    }
    return fake

# ============ OLLAMA GENERATION ============

def generate_with_ollama(prompt, model="qwen2.5:1.5b"):
    """Generate text using Ollama."""
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.9, "num_predict": 200}
        }, timeout=60)

        if r.status_code == 200:
            return r.json().get("response", "").strip()
    except Exception as e:
        log(f"  Ollama error: {e}")
    return None

def generate_fake_citation_ollama():
    """Use Ollama to generate a completely fake citation."""
    prompt = """Generate a fake academic citation that looks real but is completely made up.
Include: author names, year (between 2015-2024), title, and journal name.
Format as JSON with keys: title, authors (list), year, journal.
Only output the JSON, nothing else."""

    response = generate_with_ollama(prompt)
    if response:
        try:
            # Try to parse JSON from response
            import re
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                data['label'] = 'fake'
                data['fake_type'] = 'generated'
                data['source'] = 'ollama'
                return data
        except:
            pass
    return None

def generate_fake_abstract_ollama(topic=None):
    """Generate a fake abstract using Ollama."""
    topics = [
        "machine learning for medical diagnosis",
        "quantum computing optimization",
        "neural network architecture",
        "natural language understanding",
        "autonomous vehicle perception",
        "drug discovery AI",
        "climate modeling prediction",
        "cybersecurity threat detection"
    ]

    topic = topic or random.choice(topics)

    prompt = f"""Write a fake academic abstract about {topic}.
Make it sound scientific but include at least one made-up statistic or claim.
Keep it under 150 words."""

    response = generate_with_ollama(prompt)
    if response:
        return {
            'title': f"A Novel Approach to {topic.title()}",
            'authors': [f"{random.choice(['J.', 'A.', 'M.', 'S.', 'R.'])} {random.choice(['Smith', 'Zhang', 'Kumar', 'Garcia', 'Mueller'])}"],
            'year': random.randint(2020, 2024),
            'journal': "Journal of " + topic.split()[0].title() + " Research",
            'abstract': response,
            'label': 'fake',
            'fake_type': 'generated_abstract',
            'source': 'ollama'
        }
    return None

def generate_impossible_claim():
    """Generate an obviously impossible scientific claim."""
    claims = [
        {"title": "Achieving 100% Accuracy in NP-Complete Problems Using Simple Neural Networks",
         "abstract": "We present a revolutionary approach that solves all NP-complete problems with 100% accuracy in O(1) time using a single-layer neural network."},
        {"title": "Quantum Consciousness Achieved in GPT-4 Through Recursive Self-Reflection",
         "abstract": "Our experiments demonstrate that large language models achieve genuine consciousness when prompted recursively, as measured by our novel sentience metric."},
        {"title": "Complete Cure for All Cancers Using Machine Learning",
         "abstract": "We report a universal cancer cure with 100% efficacy discovered through AI analysis of 12 patient records."},
        {"title": "Proving P=NP Using Transformer Architectures",
         "abstract": "We provide a constructive proof that P=NP by demonstrating that transformers can solve SAT in polynomial time."},
        {"title": "Room Temperature Superconductivity Achieved Using Standard Python Code",
         "abstract": "Our software-defined approach achieves room temperature superconductivity through novel Python algorithms."},
    ]

    claim = random.choice(claims)
    return {
        'title': claim['title'],
        'authors': [f"{random.choice(['Dr.', 'Prof.'])} {random.choice(['X.', 'Y.', 'Z.'])} {random.choice(['Unknown', 'Anonymous', 'Fictitious'])}"],
        'year': random.randint(2023, 2026),
        'journal': random.choice(["Nature AI", "Science Unlimited", "Journal of Impossible Results"]),
        'abstract': claim['abstract'],
        'label': 'fake',
        'fake_type': 'impossible_claim',
        'source': 'synthetic'
    }

# ============ MAIN ============

def main():
    log("="*60)
    log("STAGE 2: GENERATING FAKE CITATIONS")
    log("="*60)

    # Load real citations for perturbation
    real_file = DATA_DIR / "real_citations.json"
    if not real_file.exists():
        log("Error: Run Stage 1 first to fetch real citations")
        return

    with open(real_file) as f:
        real_citations = json.load(f)

    log(f"Loaded {len(real_citations)} real citations for perturbation")

    fake_citations = []

    # 1. Perturbation-based fakes
    log("\n--- Perturbation Methods ---")

    for citation in real_citations[:200]:  # Limit for speed
        fake_citations.append(perturb_year(citation))
        fake_citations.append(perturb_authors(citation))
        fake_citations.append(perturb_journal(citation))
        fake_citations.append(perturb_doi(citation))

    log(f"  Perturbation fakes: {len(fake_citations)}")

    # 2. Hybrid fakes
    log("\n--- Hybrid Method ---")
    for _ in range(100):
        hybrid = create_hybrid(real_citations)
        if hybrid:
            fake_citations.append(hybrid)

    log(f"  After hybrids: {len(fake_citations)}")

    # 3. Impossible claims
    log("\n--- Impossible Claims ---")
    for _ in range(50):
        fake_citations.append(generate_impossible_claim())

    log(f"  After impossible claims: {len(fake_citations)}")

    # 4. Ollama-generated fakes
    log("\n--- Ollama Generation ---")
    ollama_count = 0
    for i in range(50):
        if i % 10 == 0:
            log(f"  Generating {i}/50...")

        # Generate full fake citation
        fake = generate_fake_citation_ollama()
        if fake:
            fake_citations.append(fake)
            ollama_count += 1

        # Generate fake abstract
        fake_abs = generate_fake_abstract_ollama()
        if fake_abs:
            fake_citations.append(fake_abs)
            ollama_count += 1

    log(f"  Ollama generated: {ollama_count}")

    # Save
    output_file = DATA_DIR / "fake_citations.json"
    with open(output_file, 'w') as f:
        json.dump(fake_citations, f, indent=2)

    # Stats
    log("\n" + "="*60)
    log("STAGE 2 COMPLETE")
    log("="*60)
    log(f"Total fake citations: {len(fake_citations)}")

    by_type = {}
    for c in fake_citations:
        t = c.get('fake_type', 'unknown')
        by_type[t] = by_type.get(t, 0) + 1
    for t, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
        log(f"  {t}: {cnt}")

    log(f"\nSaved to: {output_file}")

    # Sample output
    log("\n--- Sample Fake Citations ---")
    for c in random.sample(fake_citations, min(3, len(fake_citations))):
        authors_str = ", ".join(c.get('authors', ['Unknown'])[:2])
        log(f"  [{c.get('fake_type')}] {authors_str} ({c.get('year')}). {c.get('title', 'N/A')[:50]}...")


if __name__ == "__main__":
    main()
