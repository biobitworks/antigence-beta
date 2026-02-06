#!/usr/bin/env python3
"""
Stage 1: Fetch real citations from CrossRef and Semantic Scholar APIs.
Creates truthful antibody training data for publication verification.
"""

import json
import requests
import time
import random
from pathlib import Path
from datetime import datetime

DATA_DIR = Path.home() / ".antigence" / "data" / "citations"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ============ CROSSREF API ============

def fetch_crossref_citations(query, rows=100):
    """Fetch citations from CrossRef API."""
    url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": rows,
        "select": "DOI,title,author,published-print,container-title,abstract,type"
    }
    headers = {
        "User-Agent": "Antigence/1.0 (mailto:research@example.com)"
    }

    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code == 200:
            data = r.json()
            items = data.get("message", {}).get("items", [])
            return items
    except Exception as e:
        log(f"  CrossRef error: {e}")
    return []

def parse_crossref_item(item):
    """Parse CrossRef item into citation format."""
    try:
        # Extract authors
        authors = []
        for author in item.get("author", [])[:5]:
            name = f"{author.get('given', '')} {author.get('family', '')}".strip()
            if name:
                authors.append(name)

        # Extract year
        pub = item.get("published-print", item.get("published-online", {}))
        date_parts = pub.get("date-parts", [[None]])[0]
        year = date_parts[0] if date_parts else None

        # Extract other fields
        title = item.get("title", [""])[0] if item.get("title") else ""
        journal = item.get("container-title", [""])[0] if item.get("container-title") else ""
        doi = item.get("DOI", "")
        abstract = item.get("abstract", "")

        if title and authors and year:
            return {
                "title": title[:500],
                "authors": authors,
                "year": year,
                "journal": journal[:200],
                "doi": doi,
                "abstract": abstract[:1000] if abstract else "",
                "type": item.get("type", ""),
                "label": "real",
                "source": "crossref"
            }
    except:
        pass
    return None

# ============ SEMANTIC SCHOLAR API ============

def fetch_semantic_scholar(query, limit=100):
    """Fetch papers from Semantic Scholar API."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": "title,authors,year,venue,abstract,citationCount,externalIds"
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            return data.get("data", [])
    except Exception as e:
        log(f"  Semantic Scholar error: {e}")
    return []

def parse_s2_item(item):
    """Parse Semantic Scholar item into citation format."""
    try:
        authors = [a.get("name", "") for a in item.get("authors", [])[:5]]
        authors = [a for a in authors if a]

        title = item.get("title", "")
        year = item.get("year")
        venue = item.get("venue", "")
        abstract = item.get("abstract", "")

        # Get DOI if available
        ext_ids = item.get("externalIds", {})
        doi = ext_ids.get("DOI", "")

        if title and authors and year:
            return {
                "title": title[:500],
                "authors": authors,
                "year": year,
                "journal": venue[:200],
                "doi": doi,
                "abstract": abstract[:1000] if abstract else "",
                "citations": item.get("citationCount", 0),
                "label": "real",
                "source": "semantic_scholar"
            }
    except:
        pass
    return None

# ============ MAIN ============

def main():
    log("="*60)
    log("STAGE 1: FETCHING REAL CITATIONS")
    log("="*60)

    # Search queries for diverse citations
    queries = [
        # CS/ML
        "machine learning neural network",
        "deep learning transformer",
        "natural language processing",
        "computer vision object detection",
        "reinforcement learning",
        "software vulnerability security",
        "code analysis static dynamic",
        # Biology/Medicine
        "protein structure prediction",
        "gene expression analysis",
        "drug discovery machine learning",
        "clinical trial randomized",
        "cancer immunotherapy",
        # Physics/Math
        "quantum computing algorithm",
        "optimization gradient descent",
        # General Science
        "climate change model",
        "artificial intelligence ethics",
    ]

    all_citations = []
    seen_dois = set()

    # Fetch from CrossRef
    log("\n--- CrossRef API ---")
    for query in queries:
        log(f"  Query: {query}")
        items = fetch_crossref_citations(query, rows=50)

        for item in items:
            citation = parse_crossref_item(item)
            if citation and citation.get("doi") not in seen_dois:
                all_citations.append(citation)
                if citation.get("doi"):
                    seen_dois.add(citation["doi"])

        log(f"    Found {len(items)} items, total: {len(all_citations)}")
        time.sleep(1)  # Rate limiting

    log(f"\nCrossRef total: {len(all_citations)}")

    # Fetch from Semantic Scholar
    log("\n--- Semantic Scholar API ---")
    for query in queries[:10]:  # Limit S2 queries
        log(f"  Query: {query}")
        items = fetch_semantic_scholar(query, limit=50)

        for item in items:
            citation = parse_s2_item(item)
            if citation and citation.get("doi") not in seen_dois:
                all_citations.append(citation)
                if citation.get("doi"):
                    seen_dois.add(citation["doi"])

        log(f"    Found {len(items)} items, total: {len(all_citations)}")
        time.sleep(1)

    # Save
    output_file = DATA_DIR / "real_citations.json"
    with open(output_file, 'w') as f:
        json.dump(all_citations, f, indent=2)

    # Stats
    log("\n" + "="*60)
    log("STAGE 1 COMPLETE")
    log("="*60)
    log(f"Total real citations: {len(all_citations)}")
    log(f"With DOI: {len([c for c in all_citations if c.get('doi')])}")
    log(f"With abstract: {len([c for c in all_citations if c.get('abstract')])}")

    by_source = {}
    for c in all_citations:
        src = c.get('source', 'unknown')
        by_source[src] = by_source.get(src, 0) + 1
    for src, cnt in by_source.items():
        log(f"  {src}: {cnt}")

    log(f"\nSaved to: {output_file}")

    # Sample output
    log("\n--- Sample Citations ---")
    for c in random.sample(all_citations, min(3, len(all_citations))):
        authors_str = ", ".join(c['authors'][:2])
        if len(c['authors']) > 2:
            authors_str += " et al."
        log(f"  {authors_str} ({c['year']}). {c['title'][:60]}...")


if __name__ == "__main__":
    main()
