"""
Debug script to understand scanner behavior
"""

from code_preprocessor import CodePreprocessor
from embeddings import SimpleCodeEmbedder
from datasets import SAFE_PATTERNS, VULNERABLE_PATTERNS
import numpy as np

preprocessor = CodePreprocessor()
embedder = SimpleCodeEmbedder()

# Test case: SQL injection
test_sql_injection = """user_id = request.args.get('id')
query = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(query)
result = cursor.fetchone()"""

# Compare with first vulnerable pattern (should be identical)
vuln_pattern_0 = VULNERABLE_PATTERNS[0]['code']

print("=" * 70)
print("TEST CODE:")
print(test_sql_injection)
print("\n" + "=" * 70)
print("VULNERABLE PATTERN 0:")
print(vuln_pattern_0)
print("\n" + "=" * 70)
print("ARE THEY EQUAL?", test_sql_injection.strip() == vuln_pattern_0.strip())
print()

# Generate embeddings
test_emb = embedder.embed(test_sql_injection)
vuln_emb = embedder.embed(vuln_pattern_0)

# Check similarity
similarity = np.dot(test_emb, vuln_emb)
print(f"Cosine similarity: {similarity:.4f}")
print()

# Extract features from both
test_features = preprocessor.extract_features(test_sql_injection)
vuln_features = preprocessor.extract_features(vuln_pattern_0)

print("Test code features:")
for key in ['has_sql', 'has_sql_concat', 'has_sql_format', 'has_f_string', 'sql_keyword_count']:
    print(f"  {key}: {test_features.get(key)}")

print("\nVulnerable pattern features:")
for key in ['has_sql', 'has_sql_concat', 'has_sql_format', 'has_f_string', 'sql_keyword_count']:
    print(f"  {key}: {vuln_features.get(key)}")

# Check some safe patterns
print("\n" + "=" * 70)
print("COMPARING TO SAFE PATTERNS")
print("=" * 70)

for i in range(min(3, len(SAFE_PATTERNS))):
    safe_code = SAFE_PATTERNS[i]['code']
    safe_emb = embedder.embed(safe_code)
    safe_sim = np.dot(test_emb, safe_emb)
    print(f"Safe pattern {i}: similarity = {safe_sim:.4f}")
    print(f"  Description: {SAFE_PATTERNS[i].get('description', 'N/A')}")

# Check vulnerable patterns
print("\n" + "=" * 70)
print("COMPARING TO VULNERABLE PATTERNS")
print("=" * 70)

for i in range(min(5, len(VULNERABLE_PATTERNS))):
    vuln_code = VULNERABLE_PATTERNS[i]['code']
    vuln_emb = embedder.embed(vuln_code)
    vuln_sim = np.dot(test_emb, vuln_emb)
    print(f"Vulnerable pattern {i}: similarity = {vuln_sim:.4f}")
    print(f"  Type: {VULNERABLE_PATTERNS[i].get('vulnerability_type', 'N/A')}")
