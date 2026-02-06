"""
Debug B Cell classification
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from immunos_mcp.agents.bcell_agent import BCellAgent
from code_preprocessor import CodePreprocessor
from embeddings import SimpleCodeEmbedder
from datasets import SAFE_PATTERNS, VULNERABLE_PATTERNS

preprocessor = CodePreprocessor()
embedder = SimpleCodeEmbedder()
bcell = BCellAgent(agent_name="debug_bcell")

# Train on all patterns
print("Training B Cell...")
for pattern in SAFE_PATTERNS:
    antigen = preprocessor.to_antigen(
        code=pattern['code'],
        label='safe',
        metadata=pattern
    )
    embedding = embedder.embed(pattern['code'])
    bcell.add_pattern(antigen, embedding)

for pattern in VULNERABLE_PATTERNS:
    antigen = preprocessor.to_antigen(
        code=pattern['code'],
        label='vulnerable',
        metadata=pattern
    )
    embedding = embedder.embed(pattern['code'])
    bcell.add_pattern(antigen, embedding)

print(f"Trained with {len(bcell.patterns)} patterns")
print(f"Clones: {list(bcell.clones.keys())}")
for label, clone in bcell.clones.items():
    print(f"  {label}: {clone.size} patterns")
print()

# Test SQL injection
test_code = """user_id = request.args.get('id')
query = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(query)
result = cursor.fetchone()"""

test_antigen = preprocessor.to_antigen(test_code, label=None)
test_embedding = embedder.embed(test_code)

# Manually calculate affinities to see what's happening
print("=" * 70)
print("MANUAL AFFINITY CALCULATION")
print("=" * 70)

for class_label, clone in bcell.clones.items():
    print(f"\n{class_label.upper()} Clone ({clone.size} patterns):")
    affinities = []

    for i, pattern in enumerate(clone.patterns[:5]):  # Show first 5
        affinity = bcell.calculate_affinity(test_antigen, pattern, test_embedding)
        affinities.append(affinity)
        print(f"  Pattern {i}: affinity = {affinity:.4f}")

    # Calculate avidity
    avidity = clone.calculate_avidity(affinities)
    print(f"  → Avidity (first 5): {avidity:.4f}")

    # Calculate for all
    all_affinities = []
    for pattern in clone.patterns:
        affinity = bcell.calculate_affinity(test_antigen, pattern, test_embedding)
        all_affinities.append(affinity)

    all_avidity = clone.calculate_avidity(all_affinities)
    print(f"  → Avidity (all {len(all_affinities)}): {all_avidity:.4f}")
    print(f"  → Max affinity: {max(all_affinities):.4f}")
    print(f"  → Mean affinity: {sum(all_affinities)/len(all_affinities):.4f}")

# Now use the recognize method
print("\n" + "=" * 70)
print("B CELL RECOGNITION RESULT")
print("=" * 70)

result = bcell.recognize(test_antigen, test_embedding)
print(f"Predicted class: {result.predicted_class}")
print(f"Confidence: {result.confidence:.4f}")
print(f"Avidity scores: {result.avidity_scores}")
print(f"Explanation: {result.explanation}")
