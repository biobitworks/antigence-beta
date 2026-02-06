"""
Code Embeddings

Generates vector embeddings from code without requiring LLM APIs.
Uses character frequencies, keyword counts, and structural features.
"""

import numpy as np
from typing import Dict, Any, List
from collections import Counter
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.code_security_scanner.code_preprocessor import CodePreprocessor


class SimpleCodeEmbedder:
    """
    Generates embeddings for Python code using simple statistical features.

    Does not require LLM APIs - uses character frequencies, keyword counts,
    and structural analysis to create ~326-dimensional embeddings.
    """

    # Standard embedding dimensions
    CHAR_FREQ_DIM = 256  # ASCII character frequencies
    KEYWORD_DIM = 40     # Python keyword frequencies
    DANGER_DIM = 20      # Dangerous function indicators
    STRUCTURE_DIM = 10   # Code structure metrics

    TOTAL_DIM = CHAR_FREQ_DIM + KEYWORD_DIM + DANGER_DIM + STRUCTURE_DIM  # 326

    # Python keywords to track
    PYTHON_KEYWORDS = [
        'def', 'class', 'if', 'elif', 'else', 'for', 'while', 'break', 'continue',
        'try', 'except', 'finally', 'with', 'as', 'import', 'from', 'return',
        'yield', 'lambda', 'async', 'await', 'pass', 'raise', 'assert', 'global',
        'nonlocal', 'del', 'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None',
        'print', 'len', 'range', 'str', 'int', 'float'
    ]

    # Dangerous functions to track
    DANGER_FUNCTIONS = [
        'eval', 'exec', 'compile', '__import__', 'pickle.loads', 'pickle.load',
        'os.system', 'os.popen', 'subprocess.call', 'subprocess.run', 'subprocess.Popen',
        'shell=True', 'yaml.load', 'yaml.unsafe_load',
        'md5', 'sha1', 'DES', 'RC4',
        'requests.get', 'requests.post', 'urllib.request', 'urlopen',
        'open(', '.read()', '.write()'
    ]

    def __init__(self):
        """Initialize the embedder."""
        self.preprocessor = CodePreprocessor()

    def embed(self, code: str) -> np.ndarray:
        """
        Generate embedding vector for code.

        Args:
            code: Python code string

        Returns:
            numpy array of shape (TOTAL_DIM,)
        """
        # Extract features first
        features = self.preprocessor.extract_features(code)

        # Build embedding from different feature types
        char_freq = self._char_frequency_vector(code)
        keyword_freq = self._keyword_frequency_vector(code)
        danger_indicators = self._danger_indicator_vector(code, features)
        structure_metrics = self._structure_metric_vector(features)

        # Concatenate all feature vectors
        embedding = np.concatenate([
            char_freq,
            keyword_freq,
            danger_indicators,
            structure_metrics
        ])

        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _char_frequency_vector(self, code: str) -> np.ndarray:
        """
        Generate character frequency vector (256-dim for ASCII).

        Args:
            code: Python code string

        Returns:
            numpy array of character frequencies
        """
        freq = np.zeros(256)

        if not code:
            return freq

        # Count character occurrences
        for char in code:
            char_code = ord(char)
            if 0 <= char_code < 256:
                freq[char_code] += 1

        # Normalize by code length
        total = len(code)
        if total > 0:
            freq = freq / total

        return freq

    def _keyword_frequency_vector(self, code: str) -> np.ndarray:
        """
        Generate Python keyword frequency vector (40-dim).

        Args:
            code: Python code string

        Returns:
            numpy array of keyword frequencies
        """
        freq = np.zeros(len(self.PYTHON_KEYWORDS))

        # Count keyword occurrences
        code_lower = code.lower()
        tokens = self.preprocessor.extract_code_tokens(code)
        token_count = len(tokens) if tokens else 1

        for i, keyword in enumerate(self.PYTHON_KEYWORDS):
            # Count whole word matches
            import re
            matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', code_lower))
            freq[i] = matches / token_count

        return freq

    def _danger_indicator_vector(self, code: str, features: Dict[str, Any]) -> np.ndarray:
        """
        Generate danger indicator vector (20-dim).

        Args:
            code: Python code string
            features: Extracted features from preprocessor

        Returns:
            numpy array of danger indicators
        """
        indicators = np.zeros(self.DANGER_DIM)

        # Binary indicators from features
        indicators[0] = float(features.get('has_eval', False))
        indicators[1] = float(features.get('has_exec', False))
        indicators[2] = float(features.get('has_pickle', False))
        indicators[3] = float(features.get('has_system', False))
        indicators[4] = float(features.get('has_shell', False))
        indicators[5] = float(features.get('has_sql', False))
        indicators[6] = float(features.get('has_sql_concat', False))
        indicators[7] = float(features.get('has_sql_format', False))
        indicators[8] = float(features.get('has_sql_percent', False))
        indicators[9] = float(features.get('has_string_concat', False))
        indicators[10] = float(features.get('has_f_string', False))
        indicators[11] = float(features.get('has_md5', False))
        indicators[12] = float(features.get('has_sha1', False))
        indicators[13] = float(features.get('has_des', False))
        indicators[14] = float(features.get('has_password_literal', False))
        indicators[15] = float(features.get('has_api_key_literal', False))
        indicators[16] = float(features.get('has_secret_literal', False))
        indicators[17] = float(features.get('has_open', False))
        indicators[18] = float(features.get('has_requests', False))
        indicators[19] = float(features.get('has_urllib', False))

        return indicators

    def _structure_metric_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Generate code structure metric vector (10-dim).

        Args:
            features: Extracted features from preprocessor

        Returns:
            numpy array of structure metrics
        """
        metrics = np.zeros(self.STRUCTURE_DIM)

        # Normalize metrics to 0-1 range
        metrics[0] = min(features.get('length', 0) / 1000, 1.0)  # Length (normalized to 1000 chars)
        metrics[1] = min(features.get('line_count', 0) / 100, 1.0)  # Lines (normalized to 100)
        metrics[2] = min(features.get('danger_keyword_count', 0) / 10, 1.0)  # Danger keywords
        metrics[3] = min(features.get('sql_keyword_count', 0) / 10, 1.0)  # SQL keywords
        metrics[4] = min(features.get('python_keyword_count', 0) / 20, 1.0)  # Python keywords
        metrics[5] = min(features.get('parentheses_depth', 0) / 10, 1.0)  # Parentheses depth
        metrics[6] = min(features.get('bracket_depth', 0) / 5, 1.0)  # Bracket depth
        metrics[7] = min(features.get('brace_depth', 0) / 5, 1.0)  # Brace depth

        # Calculate entropy (normalized to 0-8 bits)
        entropy = self.preprocessor.calculate_entropy(features.get('normalized_code', ''))
        metrics[8] = min(entropy / 8.0, 1.0)

        # Complexity indicator (combination of depths)
        total_depth = (features.get('parentheses_depth', 0) +
                      features.get('bracket_depth', 0) +
                      features.get('brace_depth', 0))
        metrics[9] = min(total_depth / 20, 1.0)

        return metrics

    def batch_embed(self, code_list: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple code snippets.

        Args:
            code_list: List of Python code strings

        Returns:
            numpy array of shape (len(code_list), TOTAL_DIM)
        """
        embeddings = []
        for code in code_list:
            embeddings.append(self.embed(code))

        return np.array(embeddings)

    def similarity(self, code1: str, code2: str) -> float:
        """
        Calculate cosine similarity between two code snippets.

        Args:
            code1: First Python code string
            code2: Second Python code string

        Returns:
            Similarity score between 0 and 1
        """
        emb1 = self.embed(code1)
        emb2 = self.embed(code2)

        # Cosine similarity
        similarity = np.dot(emb1, emb2)

        return float(similarity)


# Example usage
if __name__ == '__main__':
    embedder = SimpleCodeEmbedder()

    # Example 1: Safe code
    safe_code = """
    import sqlite3
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    """

    # Example 2: Vulnerable code
    vuln_code = """
    user_id = request.args.get('id')
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    """

    print("=== Code Embedding Demo ===\n")

    # Generate embeddings
    safe_emb = embedder.embed(safe_code)
    vuln_emb = embedder.embed(vuln_code)

    print(f"Safe code embedding shape: {safe_emb.shape}")
    print(f"Safe code embedding (first 10 dims): {safe_emb[:10]}\n")

    print(f"Vulnerable code embedding shape: {vuln_emb.shape}")
    print(f"Vulnerable code embedding (first 10 dims): {vuln_emb[:10]}\n")

    # Calculate similarities
    safe_to_safe = embedder.similarity(safe_code, safe_code)
    safe_to_vuln = embedder.similarity(safe_code, vuln_code)
    vuln_to_vuln = embedder.similarity(vuln_code, vuln_code)

    print(f"Safe to Safe similarity: {safe_to_safe:.3f}")
    print(f"Safe to Vulnerable similarity: {safe_to_vuln:.3f}")
    print(f"Vulnerable to Vulnerable similarity: {vuln_to_vuln:.3f}")

    print("\n✓ Embeddings generated successfully!")
    print(f"  Total embedding dimension: {embedder.TOTAL_DIM}")
    print(f"  - Character frequencies: {embedder.CHAR_FREQ_DIM}")
    print(f"  - Keyword frequencies: {embedder.KEYWORD_DIM}")
    print(f"  - Danger indicators: {embedder.DANGER_DIM}")
    print(f"  - Structure metrics: {embedder.STRUCTURE_DIM}")
