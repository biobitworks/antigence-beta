"""
Code Security Scanner

Main scanner that coordinates B Cell and NK Cell agents for code security analysis.
Implements the immune system metaphor for detecting vulnerabilities.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from immunos_mcp.agents.bcell_agent import BCellAgent
from immunos_mcp.agents.nk_cell_agent import NKCellAgent
from immunos_mcp.agents.nk_cell_enhanced import EnhancedNKCellAgent
from immunos_mcp.core.antigen import Antigen
from immunos_mcp.core.protocols import RecognitionResult, AnomalyResult, Pattern

from examples.code_security_scanner.code_preprocessor import CodePreprocessor
from examples.code_security_scanner.embeddings import SimpleCodeEmbedder
from examples.code_security_scanner.datasets import SAFE_PATTERNS, VULNERABLE_PATTERNS


@dataclass
class SecurityResult:
    """Result of security scan"""
    code: str
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    is_vulnerable: bool
    confidence: float
    bcell_result: RecognitionResult
    nk_cell_result: AnomalyResult
    explanation: str
    vulnerability_types: List[str]
    recommendations: List[str]


class CodeSecurityScanner:
    """
    Code Security Scanner using Immune System Agents.

    Coordinates B Cell (pattern matching) and NK Cell (anomaly detection)
    agents to identify security vulnerabilities in Python code.
    """

    def __init__(self, use_enhanced_nk: bool = True):
        """
        Initialize the security scanner.

        Args:
            use_enhanced_nk: Whether to use Enhanced NK Cell (with adaptive thresholds)
        """
        # Initialize components
        self.preprocessor = CodePreprocessor()
        self.embedder = SimpleCodeEmbedder()

        # Initialize immune agents
        self.bcell = BCellAgent(agent_name="security_bcell")
        if use_enhanced_nk:
            self.nk_cell = EnhancedNKCellAgent(
                agent_name="security_nk_enhanced",
                detectors_per_class=15,
                threshold_method="min_distance"
            )
        else:
            self.nk_cell = NKCellAgent(agent_name="security_nk", num_detectors=100)

        self.use_enhanced_nk = use_enhanced_nk
        self.trained = False

    def train(self):
        """
        Train the scanner on safe and vulnerable code patterns.

        This represents the immune system learning what is "self" (safe code)
        and what patterns indicate threats (vulnerable code).
        """
        print("🧬 Training Code Security Scanner...")

        # Prepare training data
        safe_antigens = []
        vuln_antigens = []

        # Convert safe patterns to antigens
        print(f"  Loading {len(SAFE_PATTERNS)} safe code patterns (self)...")
        for pattern in SAFE_PATTERNS:
            antigen = self.preprocessor.to_antigen(
                code=pattern['code'],
                label='safe',
                metadata=pattern
            )
            # Merge metadata into features for pattern storage
            if antigen.features is None:
                antigen.features = {}
            antigen.features.update(antigen.metadata)

            # Generate embedding
            embedding = self.embedder.embed(pattern['code'])
            safe_antigens.append((antigen, embedding))

        # Convert vulnerable patterns to antigens
        print(f"  Loading {len(VULNERABLE_PATTERNS)} vulnerable patterns (non-self)...")
        for pattern in VULNERABLE_PATTERNS:
            antigen = self.preprocessor.to_antigen(
                code=pattern['code'],
                label='vulnerable',
                metadata=pattern
            )
            # Merge metadata into features for pattern storage
            if antigen.features is None:
                antigen.features = {}
            antigen.features.update(antigen.metadata)

            # Generate embedding
            embedding = self.embedder.embed(pattern['code'])
            vuln_antigens.append((antigen, embedding))

        # Train B Cell on both safe and vulnerable patterns
        print("  🦠 Training B Cell Agent (Pattern Matching)...")
        all_antigens = safe_antigens + vuln_antigens
        for antigen, embedding in all_antigens:
            self.bcell.add_pattern(antigen, embedding)

        # Train NK Cell on safe patterns only (learns "self")
        print("  🛡️  Training NK Cell Agent (Anomaly Detection on 'self')...")
        safe_only = [antigen for antigen, _ in safe_antigens]
        safe_embeddings = [embedding for _, embedding in safe_antigens]
        self.nk_cell.train_on_self(safe_only, safe_embeddings)

        self.trained = True
        print(f"✅ Training complete!")
        print(f"   B Cell: {len(all_antigens)} patterns learned across 2 classes")
        print(f"   NK Cell: Trained on {len(safe_antigens)} self patterns")
        print()

    def scan_code(self, code: str) -> SecurityResult:
        """
        Scan code for security vulnerabilities.

        Args:
            code: Python code string to analyze

        Returns:
            SecurityResult with comprehensive analysis
        """
        if not self.trained:
            raise RuntimeError("Scanner must be trained before scanning. Call train() first.")

        # Preprocess code
        antigen = self.preprocessor.to_antigen(code, label=None)
        embedding = self.embedder.embed(code)

        # Get B Cell classification with custom max-based avidity
        bcell_result = self._custom_bcell_classify(antigen, embedding)

        # Get NK Cell anomaly detection
        nk_result = self.nk_cell.detect_novelty(antigen, embedding)

        # Combine results to determine overall risk
        risk_level, is_vulnerable, confidence, explanation, vuln_types, recommendations = \
            self._analyze_results(code, bcell_result, nk_result)

        return SecurityResult(
            code=code,
            risk_level=risk_level,
            is_vulnerable=is_vulnerable,
            confidence=confidence,
            bcell_result=bcell_result,
            nk_cell_result=nk_result,
            explanation=explanation,
            vulnerability_types=vuln_types,
            recommendations=recommendations
        )

    def _custom_bcell_classify(
        self,
        antigen: Antigen,
        embedding: np.ndarray
    ) -> RecognitionResult:
        """
        Custom B Cell classification using MAX affinity (nearest neighbor).

        For security scanning, we want the highest match to dominate the decision.
        If code is very similar to a known vulnerable pattern, that should be decisive.

        Args:
            antigen: Antigen to classify
            embedding: Code embedding

        Returns:
            RecognitionResult with max-based classification
        """
        # Calculate max affinity and mean affinity for each clone
        max_affinities = {}
        mean_affinities = {}
        all_affinities = {}

        for class_label, clone in self.bcell.clones.items():
            affinities = []
            for pattern in clone.patterns:
                affinity = self.bcell.calculate_affinity(antigen, pattern, embedding)
                affinities.append(affinity)

            all_affinities[class_label] = affinities
            max_affinities[class_label] = max(affinities) if affinities else 0.0
            mean_affinities[class_label] = sum(affinities) / len(affinities) if affinities else 0.0

        # Decision: Use weighted combination of max (80%) and mean (20%)
        # This prioritizes strong matches but considers overall pattern similarity
        scores = {}
        for class_label in self.bcell.clones.keys():
            scores[class_label] = (
                0.8 * max_affinities[class_label] +
                0.2 * mean_affinities[class_label]
            )

        # Determine winner
        winner_class = max(scores, key=scores.get)
        winner_score = scores[winner_class]
        total_score = sum(scores.values())

        confidence = winner_score / total_score if total_score > 0 else 0.0

        # Find top matched patterns
        matched_patterns = []
        for pattern in self.bcell.clones[winner_class].patterns:
            affinity = self.bcell.calculate_affinity(antigen, pattern, embedding)
            # Copy features or create new dict
            pattern_features = pattern.features.copy() if pattern.features else {}
            pattern_features['_match_affinity'] = affinity

            matched_patterns.append(Pattern(
                pattern_id=pattern.pattern_id,
                class_label=pattern.class_label,
                example_data=pattern.example_data,
                embedding=pattern.embedding,
                features=pattern_features,
                creation_time=pattern.creation_time
            ))

        # Sort by affinity
        matched_patterns.sort(
            key=lambda p: p.features.get('_match_affinity', 0.0),
            reverse=True
        )

        return RecognitionResult(
            predicted_class=winner_class,
            confidence=confidence,
            is_uncertain=False,
            avidity_scores=scores,
            explanation=f"Max-based classification: {winner_class} (max={max_affinities[winner_class]:.3f}, mean={mean_affinities[winner_class]:.3f})",
            metadata={
                "max_affinities": max_affinities,
                "mean_affinities": mean_affinities,
                "matched_patterns": matched_patterns[:5]  # Top 5
            }
        )

    def _analyze_results(
        self,
        code: str,
        bcell_result: RecognitionResult,
        nk_result: AnomalyResult
    ) -> Tuple[str, bool, float, str, List[str], List[str]]:
        """
        Analyze B Cell and NK Cell results to determine overall security assessment.

        Args:
            code: Original code
            bcell_result: B Cell classification result
            nk_result: NK Cell anomaly detection result

        Returns:
            Tuple of (risk_level, is_vulnerable, confidence, explanation, vuln_types, recommendations)
        """
        bcell_says_vuln = (bcell_result.predicted_class == 'vulnerable')
        nk_says_anomaly = nk_result.is_anomaly

        vuln_types = []
        recommendations = []

        # Decision matrix based on agent agreement
        if bcell_says_vuln and nk_says_anomaly:
            # Both agents agree: HIGH RISK
            risk_level = 'high'
            is_vulnerable = True
            confidence = (bcell_result.confidence + nk_result.confidence) / 2

            explanation = (
                f"⚠️  BOTH AGENTS DETECTED THREAT\n"
                f"  B Cell: Identified as '{bcell_result.predicted_class}' "
                f"(confidence: {bcell_result.confidence:.2f})\n"
                f"  NK Cell: Detected as anomaly (score: {nk_result.anomaly_score:.2f})\n"
                f"  → This pattern matches known vulnerabilities AND exhibits unusual behavior"
            )

        elif bcell_says_vuln and not nk_says_anomaly:
            # B Cell detects, NK Cell doesn't: MEDIUM-HIGH RISK
            risk_level = 'medium'
            is_vulnerable = True
            confidence = bcell_result.confidence

            explanation = (
                f"⚠️  B CELL DETECTED VULNERABILITY\n"
                f"  B Cell: Identified as '{bcell_result.predicted_class}' "
                f"(confidence: {bcell_result.confidence:.2f})\n"
                f"  NK Cell: Appears similar to safe patterns (score: {nk_result.anomaly_score:.2f})\n"
                f"  → Known vulnerability pattern, but structure seems familiar"
            )

        elif not bcell_says_vuln and nk_says_anomaly:
            # NK Cell detects, B Cell doesn't: MEDIUM RISK
            risk_level = 'medium'
            is_vulnerable = True
            confidence = nk_result.confidence

            explanation = (
                f"⚠️  NK CELL DETECTED ANOMALY\n"
                f"  B Cell: Classified as '{bcell_result.predicted_class}' "
                f"(confidence: {bcell_result.confidence:.2f})\n"
                f"  NK Cell: Novel pattern detected (score: {nk_result.anomaly_score:.2f})\n"
                f"  → This doesn't match known vulnerabilities but shows unusual characteristics"
            )

        else:
            # Neither agent detects: LOW RISK
            risk_level = 'low'
            is_vulnerable = False
            confidence = (bcell_result.confidence + (1.0 - nk_result.anomaly_score)) / 2

            explanation = (
                f"✓ BOTH AGENTS APPROVE\n"
                f"  B Cell: Classified as '{bcell_result.predicted_class}' "
                f"(confidence: {bcell_result.confidence:.2f})\n"
                f"  NK Cell: Matches normal patterns (score: {nk_result.anomaly_score:.2f})\n"
                f"  → Code appears safe"
            )

        # Extract vulnerability types from B Cell metadata
        matched_patterns = bcell_result.metadata.get('matched_patterns', [])
        if bcell_says_vuln and matched_patterns:
            for pattern in matched_patterns[:3]:  # Top 3
                if pattern.features and 'vulnerability_type' in pattern.features:
                    vuln_type = pattern.features['vulnerability_type']
                    if vuln_type not in vuln_types:
                        vuln_types.append(vuln_type)

        # Generate recommendations
        if is_vulnerable:
            recommendations = self._generate_recommendations(code, vuln_types)

        return risk_level, is_vulnerable, confidence, explanation, vuln_types, recommendations

    def _generate_recommendations(self, code: str, vuln_types: List[str]) -> List[str]:
        """
        Generate security recommendations based on detected vulnerabilities.

        Args:
            code: Original code
            vuln_types: List of detected vulnerability types

        Returns:
            List of recommendations
        """
        recommendations = []

        for vuln_type in vuln_types:
            if vuln_type == 'sql_injection':
                recommendations.append(
                    "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))"
                )
            elif vuln_type == 'command_injection':
                recommendations.append(
                    "Use subprocess with list arguments (no shell=True): subprocess.run(['ls', '-la', dir])"
                )
            elif vuln_type == 'path_traversal':
                recommendations.append(
                    "Validate and sanitize file paths, use os.path.normpath() and check against allowed directories"
                )
            elif vuln_type == 'code_injection':
                recommendations.append(
                    "Never use eval() or exec() on user input. Use ast.literal_eval() for safe evaluation"
                )
            elif vuln_type == 'unsafe_deserialization':
                recommendations.append(
                    "Use safe serialization: json.loads() instead of pickle.loads()"
                )
            elif vuln_type == 'hardcoded_credentials':
                recommendations.append(
                    "Store credentials in environment variables: password = os.getenv('DB_PASSWORD')"
                )
            elif vuln_type == 'weak_crypto':
                recommendations.append(
                    "Use strong cryptography: bcrypt or PBKDF2 for passwords, AES for encryption"
                )
            elif vuln_type == 'ssrf':
                recommendations.append(
                    "Validate URLs against allowlist, use URL parsing to check scheme and host"
                )

        if not recommendations:
            recommendations.append("Review code against OWASP Top 10 security guidelines")

        return recommendations

    def batch_scan(self, code_list: List[str]) -> List[SecurityResult]:
        """
        Scan multiple code snippets.

        Args:
            code_list: List of Python code strings

        Returns:
            List of SecurityResult objects
        """
        results = []
        for code in code_list:
            results.append(self.scan_code(code))
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get scanner statistics.

        Returns:
            Dictionary of statistics
        """
        # Count detectors based on NK Cell type
        if self.trained:
            if self.use_enhanced_nk:
                # EnhancedNKCellAgent has detector_sets (dict of ClassDetectorSet)
                nk_detectors = sum(
                    len(detector_set.detectors)
                    for detector_set in self.nk_cell.detector_sets.values()
                )
            else:
                # Regular NKCellAgent has detectors (list)
                nk_detectors = len(self.nk_cell.detectors)
        else:
            nk_detectors = 0

        return {
            'trained': self.trained,
            'bcell_patterns': len(self.bcell.clones) if self.trained else 0,
            'nk_detectors': nk_detectors,
            'embedding_dim': self.embedder.TOTAL_DIM,
            'using_enhanced_nk': self.use_enhanced_nk,
        }


# Example usage
if __name__ == '__main__':
    # Create and train scanner
    scanner = CodeSecurityScanner(use_enhanced_nk=True)
    scanner.train()

    # Test case 1: SQL Injection
    print("=" * 70)
    print("TEST CASE 1: SQL Injection")
    print("=" * 70)
    test_code_1 = """
user_id = request.args.get('id')
query = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(query)
"""
    result = scanner.scan_code(test_code_1)
    print(f"Risk Level: {result.risk_level.upper()}")
    print(f"Vulnerable: {result.is_vulnerable}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"\n{result.explanation}")
    if result.recommendations:
        print(f"\nRecommendations:")
        for rec in result.recommendations:
            print(f"  • {rec}")

    print("\n")

    # Test case 2: Safe code
    print("=" * 70)
    print("TEST CASE 2: Safe Parameterized Query")
    print("=" * 70)
    test_code_2 = """
import sqlite3
conn = sqlite3.connect('database.db')
cursor = conn.cursor()
cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
"""
    result = scanner.scan_code(test_code_2)
    print(f"Risk Level: {result.risk_level.upper()}")
    print(f"Vulnerable: {result.is_vulnerable}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"\n{result.explanation}")

    print("\n")
    print("=" * 70)
    print(f"Scanner Statistics:")
    stats = scanner.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
