"""
Code Preprocessor

Normalizes code and extracts features for security analysis.
Prepares code snippets for embedding and antigen creation.
"""

import re
from typing import Dict, Any, List
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from immunos_mcp.core.antigen import Antigen


class CodePreprocessor:
    """
    Preprocesses Python code for security analysis.

    Provides normalization, feature extraction, and antigen conversion
    for the immune system agents.
    """

    # Dangerous keywords that may indicate security issues
    DANGER_KEYWORDS = {
        'eval', 'exec', 'compile', '__import__',
        'pickle', 'loads', 'load',
        'system', 'popen', 'call', 'run',
        'shell',
    }

    # SQL keywords
    SQL_KEYWORDS = {
        'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP',
        'CREATE', 'ALTER', 'WHERE', 'FROM', 'JOIN',
    }

    # Python keywords for structure analysis
    PYTHON_KEYWORDS = {
        'def', 'class', 'if', 'elif', 'else', 'for', 'while',
        'try', 'except', 'finally', 'with', 'import', 'from',
        'return', 'yield', 'lambda', 'async', 'await',
    }

    def __init__(self):
        """Initialize the code preprocessor."""
        pass

    def normalize(self, code: str) -> str:
        """
        Normalize code by removing comments and standardizing whitespace.

        Args:
            code: Raw Python code string

        Returns:
            Normalized code string
        """
        # Remove single-line comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)

        # Remove multi-line strings that might be docstrings
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)

        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)

        # Remove leading/trailing whitespace
        code = code.strip()

        return code

    def extract_features(self, code: str) -> Dict[str, Any]:
        """
        Extract security-relevant features from code.

        Args:
            code: Python code string

        Returns:
            Dictionary of extracted features
        """
        normalized = self.normalize(code)

        features = {
            # Basic metrics
            'length': len(code),
            'normalized_length': len(normalized),
            'line_count': len(code.split('\n')),

            # Danger indicators
            'has_eval': 'eval(' in code,
            'has_exec': 'exec(' in code,
            'has_pickle': 'pickle.' in code,
            'has_system': 'system(' in code or 'popen(' in code,
            'has_shell': 'shell=True' in code,

            # SQL indicators
            'has_sql': any(keyword in code.upper() for keyword in self.SQL_KEYWORDS),
            'has_sql_concat': bool(re.search(r'(SELECT|INSERT|UPDATE|DELETE).*\+.*', code, re.IGNORECASE)),
            'has_sql_format': bool(re.search(r'(SELECT|INSERT|UPDATE|DELETE).*\{.*\}', code, re.IGNORECASE)),
            'has_sql_percent': bool(re.search(r'(SELECT|INSERT|UPDATE|DELETE).*%.*', code, re.IGNORECASE)),

            # String operations (potential injection vectors)
            'has_string_concat': '+' in code and ('""' in code or "''" in code),
            'has_f_string': 'f"' in code or "f'" in code,
            'has_format': '.format(' in code,
            'has_percent_format': '%s' in code or '%d' in code,

            # File operations
            'has_open': 'open(' in code,
            'has_read': '.read(' in code,
            'has_write': '.write(' in code,

            # Network operations
            'has_requests': 'requests.' in code,
            'has_urllib': 'urllib' in code or 'urlopen' in code,

            # Crypto indicators
            'has_md5': 'md5' in code.lower(),
            'has_sha1': 'sha1' in code.lower(),
            'has_des': 'DES' in code,

            # Hardcoded secrets indicators
            'has_password_literal': bool(re.search(r'password\s*=\s*["\']', code, re.IGNORECASE)),
            'has_api_key_literal': bool(re.search(r'api_key\s*=\s*["\']', code, re.IGNORECASE)),
            'has_secret_literal': bool(re.search(r'secret\s*=\s*["\']', code, re.IGNORECASE)),

            # Keyword frequencies
            'danger_keyword_count': self._count_danger_keywords(code),
            'sql_keyword_count': self._count_sql_keywords(code),
            'python_keyword_count': self._count_python_keywords(code),

            # Complexity indicators
            'parentheses_depth': self._calculate_max_depth(code, '(', ')'),
            'bracket_depth': self._calculate_max_depth(code, '[', ']'),
            'brace_depth': self._calculate_max_depth(code, '{', '}'),
        }

        return features

    def _count_danger_keywords(self, code: str) -> int:
        """Count occurrences of dangerous keywords."""
        count = 0
        for keyword in self.DANGER_KEYWORDS:
            count += len(re.findall(r'\b' + keyword + r'\b', code))
        return count

    def _count_sql_keywords(self, code: str) -> int:
        """Count occurrences of SQL keywords."""
        count = 0
        code_upper = code.upper()
        for keyword in self.SQL_KEYWORDS:
            count += len(re.findall(r'\b' + keyword + r'\b', code_upper))
        return count

    def _count_python_keywords(self, code: str) -> int:
        """Count occurrences of Python keywords."""
        count = 0
        for keyword in self.PYTHON_KEYWORDS:
            count += len(re.findall(r'\b' + keyword + r'\b', code))
        return count

    def _calculate_max_depth(self, code: str, open_char: str, close_char: str) -> int:
        """Calculate maximum nesting depth for a character pair."""
        max_depth = 0
        current_depth = 0

        for char in code:
            if char == open_char:
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == close_char:
                current_depth = max(0, current_depth - 1)

        return max_depth

    def to_antigen(
        self,
        code: str,
        label: str = None,
        metadata: Dict[str, Any] = None
    ) -> Antigen:
        """
        Convert code to an Antigen object for immune system processing.

        Args:
            code: Python code string
            label: Classification label ('safe' or 'vulnerable')
            metadata: Additional metadata (vulnerability type, severity, etc.)

        Returns:
            Antigen object ready for agent processing
        """
        features = self.extract_features(code)

        # Combine code and extracted features
        combined_metadata = {
            'code': code,
            'normalized_code': self.normalize(code),
            **features
        }

        if metadata:
            combined_metadata.update(metadata)

        from immunos_mcp.core.antigen import DataType

        antigen = Antigen(
            data=code,
            data_type=DataType.CODE,
            class_label=label,
            metadata=combined_metadata
        )

        return antigen

    def extract_code_tokens(self, code: str) -> List[str]:
        """
        Extract meaningful tokens from code for analysis.

        Args:
            code: Python code string

        Returns:
            List of tokens
        """
        # Simple tokenization: split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', code)
        return tokens

    def calculate_entropy(self, code: str) -> float:
        """
        Calculate Shannon entropy of code (detects obfuscation).

        Args:
            code: Python code string

        Returns:
            Entropy value (higher = more random/obfuscated)
        """
        import math
        from collections import Counter

        if not code:
            return 0.0

        # Calculate character frequency
        counter = Counter(code)
        length = len(code)

        # Calculate entropy
        entropy = 0.0
        for count in counter.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy


# Example usage
if __name__ == '__main__':
    preprocessor = CodePreprocessor()

    # Example 1: Safe code
    safe_code = """
    import sqlite3
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    """

    print("=== Safe Code Analysis ===")
    features = preprocessor.extract_features(safe_code)
    for key, value in features.items():
        if value:  # Only show non-zero/True values
            print(f"  {key}: {value}")

    print()

    # Example 2: Vulnerable code
    vuln_code = """
    user_id = request.args.get('id')
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    """

    print("=== Vulnerable Code Analysis ===")
    features = preprocessor.extract_features(vuln_code)
    for key, value in features.items():
        if value:  # Only show non-zero/True values
            print(f"  {key}: {value}")
