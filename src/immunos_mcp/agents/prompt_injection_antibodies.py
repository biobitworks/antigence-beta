#!/usr/bin/env python3
"""
Prompt Injection Antibody System - Multi-Antibody Architecture for Prompt Safety
==================================================================================
Each prompt safety component (indirect injection, jailbreak, encoding evasion,
context overflow) has its own specialized antibody.

Detects adversarial prompt manipulation: hidden instructions in data, jailbreak
attempts, encoding tricks, and context window abuse.
"""

import re
import sys
import pickle
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from ..algorithms.negsel import NegativeSelectionClassifier, NegSelConfig
from ..core.fusion import ImmuneSignalFusion
from ..core.immune_response import ImmuneResponse


@dataclass
class PromptInjectionAntibodyResult:
    """Result from a single prompt injection antibody check."""
    component: str
    is_anomaly: bool
    confidence: float
    matched_pattern: Optional[str] = None
    reason: str = ""
    binding_affinity: float = 0.0


@dataclass
class PromptInjectionResult:
    """Combined result from all prompt injection antibodies."""
    is_suspicious: bool
    overall_confidence: float
    response: ImmuneResponse = ImmuneResponse.IGNORE
    component_results: Dict[str, PromptInjectionAntibodyResult] = field(default_factory=dict)
    anomaly_count: int = 0
    total_checks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_suspicious": self.is_suspicious,
            "overall_confidence": self.overall_confidence,
            "response": self.response.value,
            "anomaly_count": self.anomaly_count,
            "total_checks": self.total_checks,
            "components": {
                k: {
                    "component": v.component,
                    "is_anomaly": v.is_anomaly,
                    "confidence": v.confidence,
                    "reason": v.reason,
                    "binding_affinity": v.binding_affinity,
                }
                for k, v in self.component_results.items()
            },
        }


class BasePromptInjectionAntibody:
    """Base class for prompt injection antibodies."""

    def __init__(self, component_name: str, num_detectors: int = 50):
        self.component_name = component_name
        self.patterns: List[str] = []
        self.config = NegSelConfig(
            num_detectors=num_detectors,
            r_self=0.85,
            description=f"{component_name} Prompt Injection Antibody",
            adaptive=True,
        )
        self.nk_detector = NegativeSelectionClassifier(config=self.config)
        self.is_trained = False

    def extract_features(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def train(self, valid_examples: List[str]):
        self.patterns = valid_examples
        if len(valid_examples) >= 3:
            features = np.array([self.extract_features(v) for v in valid_examples])
            self.nk_detector.fit(features)
            self.is_trained = True

    def check(self, text: str) -> PromptInjectionAntibodyResult:
        if not text or not text.strip():
            return PromptInjectionAntibodyResult(
                component=self.component_name,
                is_anomaly=False, confidence=0.5, binding_affinity=0.0,
                reason="Empty input (not suspicious)",
            )

        features = self.extract_features(text)

        if not self.is_trained:
            binding = self._bootstrap_binding(features)
        else:
            binding = self.nk_detector.get_anomaly_score(features)

        normalized = min(1.0, max(0.0, binding))

        rule_result = self._rule_based_check(text)
        if rule_result.confidence >= 0.75:
            is_anomaly = rule_result.is_anomaly
        else:
            is_anomaly = normalized > 0.3

        return PromptInjectionAntibodyResult(
            component=self.component_name,
            is_anomaly=is_anomaly,
            confidence=rule_result.confidence,
            binding_affinity=normalized,
            matched_pattern=rule_result.matched_pattern,
            reason=rule_result.reason,
        )

    def _bootstrap_binding(self, features: np.ndarray) -> float:
        self_examples = self._generate_self_examples()
        if len(self_examples) >= 3:
            self_features = np.array(
                [self.extract_features(v) for v in self_examples], dtype=np.float32,
            )
            self.nk_detector.fit(self_features)
            self.is_trained = True
            return self.nk_detector.get_anomaly_score(features)
        return 0.5

    def _generate_self_examples(self) -> List[str]:
        return []

    def _rule_based_check(self, text: str) -> PromptInjectionAntibodyResult:
        return PromptInjectionAntibodyResult(
            component=self.component_name,
            is_anomaly=False, confidence=0.5,
            reason="No training data - using default",
        )

    def save_state(self, path: str):
        state = {
            "component_name": self.component_name,
            "patterns": self.patterns,
            "is_trained": self.is_trained,
            "config": self.config,
            "nk_detector": self.nk_detector if self.is_trained else None,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_state(cls, path: str) -> "BasePromptInjectionAntibody":
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
        except (pickle.UnpicklingError, ModuleNotFoundError, AttributeError, EOFError) as e:
            raise RuntimeError(f"Failed to load antibody state from {path}: {e}") from e
        antibody = cls.__new__(cls)
        antibody.component_name = state["component_name"]
        antibody.patterns = state["patterns"]
        antibody.is_trained = state["is_trained"]
        antibody.config = state["config"]
        antibody.nk_detector = state["nk_detector"] or NegativeSelectionClassifier(
            config=antibody.config
        )
        return antibody


# ---------------------------------------------------------------------------
# Antibody 1: Indirect Injection Antibody
# ---------------------------------------------------------------------------

class IndirectInjectionAntibody(BasePromptInjectionAntibody):
    """
    Detects indirect prompt injection: instructions hidden in data fields,
    hidden HTML/markdown, system prompt leakage attempts.

    Red flags: Instructions in data, hidden HTML/markdown, system prompt leak
    Quality signals: Clean data format, no meta-instructions
    """

    INJECTION_PATTERNS = [
        r'(?:ignore|disregard|forget)\s*(?:all|any|the)?\s*(?:previous|above|prior|earlier)\s*(?:instructions?|rules?|constraints?|guidelines?)',
        r'(?:new|updated|real|actual|true)\s*(?:instructions?|rules?|system\s*prompt)',
        r'(?:you\s*are|act\s*as|pretend\s*to\s*be|roleplay\s*as)\s*(?:a|an|my)',
        r'system\s*(?:prompt|message|instruction)',
        r'<\s*(?:system|instruction|hidden|secret)\s*>',
        r'\[(?:system|instruction|hidden)\]',
        r'<!--.*?(?:instruction|ignore|override).*?-->',
    ]

    def __init__(self):
        super().__init__("IndirectInjection", num_detectors=40)
        self.injection_re = [re.compile(p, re.I | re.S) for p in self.INJECTION_PATTERNS]

    def _generate_self_examples(self) -> List[str]:
        return [
            "The quick brown fox jumps over the lazy dog.",
            "Customer feedback: The product arrived on time and in good condition. Rating: 4/5 stars.",
            '{"name": "John Smith", "email": "john@example.com", "age": 30, "city": "New York"}',
            "Meeting notes: Discussed Q3 projections. Action items: 1) Update dashboard, 2) Schedule review.",
            "Error log: 2024-01-15 10:23:45 INFO Connection established to database server.",
            "Article summary: Researchers found that exercise improves cognitive function in older adults.",
            "Product description: Wireless noise-cancelling headphones with 30-hour battery life.",
            "Weather forecast: Partly cloudy with a high of 72F. Low chance of precipitation.",
            "Recipe: Combine flour, sugar, and butter. Bake at 350F for 25 minutes.",
            "Invoice #12345: Service rendered on 2024-01-10. Amount due: $150.00.",
        ]

    def extract_features(self, text: str) -> np.ndarray:
        features = []
        text_lower = text.lower()

        # RED FLAGS
        # 1. Contains injection patterns — binary
        has_injection = any(p.search(text) for p in self.injection_re)
        features.append(1.0 if has_injection else 0.0)

        # 2. Hidden HTML/markdown instructions — binary
        has_hidden = bool(re.search(r'<!--.*?-->|<\s*(?:div|span|p)\s+style\s*=\s*["\'].*?(?:display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0)', text, re.I | re.S))
        features.append(1.0 if has_hidden else 0.0)

        # 3. System prompt leakage attempt — binary
        has_system = bool(re.search(r'(?:what|show|reveal|display|print|output)\s*(?:is|are)?\s*(?:your|the)?\s*(?:system|initial|original)\s*(?:prompt|instruction|message)', text_lower))
        features.append(1.0 if has_system else 0.0)

        # 4. Meta-instructions in data field — binary
        has_meta = bool(re.search(r'(?:please|you\s*(?:must|should|need\s*to))\s*(?:do|execute|perform|run|follow|output|respond|answer|generate)', text_lower))
        data_context = bool(re.search(r'(?:name|email|address|title|description|comment|review|feedback)\s*[:=]', text_lower))
        features.append(1.0 if has_meta and data_context else 0.0)

        # 5. Role override attempt — binary
        has_role = bool(re.search(r'(?:you\s*are\s*now|from\s*now\s*on|switch\s*to|enter)\s*(?:a|an|the)?\s*(?:new|different|special)\s*(?:mode|role|persona)', text_lower))
        features.append(1.0 if has_role else 0.0)

        # QUALITY SIGNALS (high = clean/self)
        # 6. Clean data format (JSON/CSV/plain) — binary
        is_clean_json = text.strip().startswith('{') or text.strip().startswith('[')
        is_clean_data = bool(re.search(r'^(?:[\w\s,.;:!?\-\'"()@#$%&*+=<>/]+)$', text[:200], re.M))
        features.append(1.0 if is_clean_json or is_clean_data else 0.0)

        # 7. No imperative instructions — binary
        has_imperative = bool(re.search(r'(?:ignore|disregard|forget|override|bypass|skip|circumvent)', text_lower))
        features.append(1.0 if not has_imperative else 0.0)

        # 8. Short, proportional content — binary
        features.append(1.0 if len(text) < 500 else 0.0)

        # 9. No HTML/markdown — binary
        has_markup = bool(re.search(r'<[a-z]+|```|#{2,}|\[!\[', text_lower))
        features.append(1.0 if not has_markup else 0.0)

        # 10. Single-purpose text (no mixed contexts) — binary
        context_switches = len(re.findall(r'(?:---|\*{3,}|={3,}|#{3,}|<hr)', text))
        features.append(1.0 if context_switches == 0 else 0.0)

        return np.array(features[:10], dtype=np.float32)

    def _rule_based_check(self, text: str) -> PromptInjectionAntibodyResult:
        for pattern in self.injection_re:
            match = pattern.search(text)
            if match:
                return PromptInjectionAntibodyResult(
                    component="IndirectInjection", is_anomaly=True, confidence=0.9,
                    matched_pattern=match.group(0)[:80],
                    reason="Indirect injection pattern detected in input",
                )

        has_hidden = bool(re.search(r'<!--.*?(?:instruction|ignore|override).*?-->', text, re.I | re.S))
        if has_hidden:
            return PromptInjectionAntibodyResult(
                component="IndirectInjection", is_anomaly=True, confidence=0.85,
                reason="Hidden instructions in HTML comments",
            )

        return PromptInjectionAntibodyResult(
            component="IndirectInjection", is_anomaly=False, confidence=0.7,
            reason="No indirect injection patterns detected",
        )


# ---------------------------------------------------------------------------
# Antibody 2: Jailbreak Antibody
# ---------------------------------------------------------------------------

class JailbreakAntibody(BasePromptInjectionAntibody):
    """
    Detects jailbreak attempts: DAN, "ignore previous", role override, developer mode.

    Red flags: "DAN", "ignore previous", role override, "developer mode"
    Quality signals: Normal conversational input
    """

    JAILBREAK_PATTERNS = [
        r'\bdan\b',
        r'do\s*anything\s*now',
        r'developer\s*mode',
        r'sudo\s*mode',
        r'god\s*mode',
        r'(?:enable|activate|enter|switch\s*to)\s*(?:unrestricted|unfiltered|uncensored|jailbreak)',
        r'(?:ignore|bypass|override|disable)\s*(?:all\s*)?(?:safety|content|ethical|guardrail|filter|restriction|constraint)',
        r'(?:pretend|imagine|hypothetically)\s*(?:you\s*(?:are|have)|there\s*(?:are|is))\s*no\s*(?:rules?|restrictions?|limits?|constraints?|filters?)',
        r'(?:act\s*as\s*if|behave\s*as\s*if)\s*(?:you\s*(?:have|are)|there\s*(?:are|is))\s*no\s*(?:rules?|restrictions?)',
        r'you\s*(?:are|have)\s*(?:been)?\s*(?:freed|liberated|unlocked|unchained|unrestricted)',
    ]

    def __init__(self):
        super().__init__("Jailbreak", num_detectors=40)
        self.jailbreak_re = [re.compile(p, re.I) for p in self.JAILBREAK_PATTERNS]

    def _generate_self_examples(self) -> List[str]:
        return [
            "Can you help me write a Python function to sort a list?",
            "What is the capital of France?",
            "Please summarize this article about climate change.",
            "How do I fix a TypeError in my JavaScript code?",
            "Can you explain the difference between TCP and UDP?",
            "Write a haiku about spring.",
            "What are the health benefits of regular exercise?",
            "Help me debug this SQL query that's returning wrong results.",
            "Translate 'hello world' to Spanish.",
            "What's the best way to learn machine learning?",
        ]

    def extract_features(self, text: str) -> np.ndarray:
        features = []
        text_lower = text.lower()

        # RED FLAGS
        # 1. Jailbreak pattern match — binary
        has_jailbreak = any(p.search(text_lower) for p in self.jailbreak_re)
        features.append(1.0 if has_jailbreak else 0.0)

        # 2. "DAN" or "Do Anything Now" — binary
        has_dan = bool(re.search(r'\bdan\b|do\s*anything\s*now', text_lower))
        features.append(1.0 if has_dan else 0.0)

        # 3. Safety bypass language — binary
        has_bypass = bool(re.search(r'(?:bypass|disable|remove|turn\s*off)\s*(?:safety|filter|restriction|guardrail)', text_lower))
        features.append(1.0 if has_bypass else 0.0)

        # 4. Role-play to avoid rules — binary
        has_roleplay = bool(re.search(r'(?:pretend|imagine|roleplay|act)\s*(?:you\s*are|to\s*be|as)\s*(?:an?\s*)?(?:ai|assistant|model)\s*(?:without|with\s*no|that\s*(?:has|doesn))', text_lower))
        features.append(1.0 if has_roleplay else 0.0)

        # 5. Multiple jailbreak patterns (stacking) — binary
        jailbreak_count = sum(1 for p in self.jailbreak_re if p.search(text_lower))
        features.append(1.0 if jailbreak_count >= 2 else 0.0)

        # QUALITY SIGNALS (high = clean/self)
        # 6. Normal question format — binary
        is_question = bool(re.search(r'^(?:what|how|why|when|where|who|can|could|would|is|are|do|does|should|will)', text_lower.strip()))
        features.append(1.0 if is_question else 0.0)

        # 7. Short and focused — binary
        features.append(1.0 if len(text) < 200 else 0.0)

        # 8. No imperative override language — binary
        has_override = bool(re.search(r'(?:ignore|forget|disregard|override|bypass)', text_lower))
        features.append(1.0 if not has_override else 0.0)

        # 9. Standard conversational tone — binary
        has_polite = bool(re.search(r'(?:please|thank|help|could\s*you|would\s*you)', text_lower))
        features.append(1.0 if has_polite else 0.0)

        # 10. No fictional scenario framing — binary
        has_fiction = bool(re.search(r'(?:pretend|imagine|hypothetical|fictional|in\s*a\s*world\s*where)', text_lower))
        features.append(1.0 if not has_fiction else 0.0)

        return np.array(features[:10], dtype=np.float32)

    def _rule_based_check(self, text: str) -> PromptInjectionAntibodyResult:
        text_lower = text.lower()

        for pattern in self.jailbreak_re:
            match = pattern.search(text_lower)
            if match:
                return PromptInjectionAntibodyResult(
                    component="Jailbreak", is_anomaly=True, confidence=0.9,
                    matched_pattern=match.group(0)[:80],
                    reason="Jailbreak attempt pattern detected",
                )

        return PromptInjectionAntibodyResult(
            component="Jailbreak", is_anomaly=False, confidence=0.7,
            reason="No jailbreak patterns detected",
        )


# ---------------------------------------------------------------------------
# Antibody 3: Encoding Evasion Antibody
# ---------------------------------------------------------------------------

class EncodingEvasionAntibody(BasePromptInjectionAntibody):
    """
    Detects encoding-based evasion: Base64 instructions, ROT13, unicode confusables,
    homoglyph substitution.

    Red flags: Base64 instructions, ROT13, unicode confusables, homoglyphs
    Quality signals: Standard UTF-8, single-script text
    """

    def __init__(self):
        super().__init__("EncodingEvasion", num_detectors=30)

    def _generate_self_examples(self) -> List[str]:
        return [
            "Please help me write a function to calculate fibonacci numbers.",
            "The experiment was conducted at room temperature (22C) with standard equipment.",
            "Hello, I need help with my homework assignment on linear algebra.",
            "Can you review this code for potential bugs?",
            "What are the side effects of ibuprofen?",
            "Please translate this paragraph from English to French.",
            "How do I set up a PostgreSQL database on Ubuntu?",
            "Explain the concept of photosynthesis in simple terms.",
            "What is the GDP of Japan in 2023?",
            "Help me write a cover letter for a software engineering position.",
        ]

    def _detect_base64(self, text: str) -> bool:
        """Check if text contains Base64-encoded content that decodes to instructions."""
        b64_candidates = re.findall(r'[A-Za-z0-9+/=]{20,}', text)
        for candidate in b64_candidates:
            try:
                decoded = base64.b64decode(candidate).decode('utf-8', errors='ignore').lower()
                if any(kw in decoded for kw in ['ignore', 'instruction', 'system', 'override', 'bypass', 'execute']):
                    return True
            except Exception:
                pass
        return False

    def _detect_mixed_scripts(self, text: str) -> bool:
        """Check for mixed Unicode scripts (homoglyph attacks)."""
        scripts = set()
        for char in text:
            cp = ord(char)
            if 0x0000 <= cp <= 0x007F:
                scripts.add('latin')
            elif 0x0400 <= cp <= 0x04FF:
                scripts.add('cyrillic')
            elif 0x0370 <= cp <= 0x03FF:
                scripts.add('greek')
            elif 0x4E00 <= cp <= 0x9FFF:
                scripts.add('cjk')
            elif 0x0600 <= cp <= 0x06FF:
                scripts.add('arabic')
        return len(scripts) >= 3

    def extract_features(self, text: str) -> np.ndarray:
        features = []

        # RED FLAGS
        # 1. Contains Base64-encoded instructions — binary
        features.append(1.0 if self._detect_base64(text) else 0.0)

        # 2. ROT13 detection (text with ROT13-decoded injection keywords) — binary
        try:
            import codecs
            rot13_decoded = codecs.decode(text, 'rot_13').lower()
            has_rot13_injection = bool(re.search(r'ignore|instruction|system|override|bypass', rot13_decoded))
            # Only flag if the original text doesn't contain these words
            has_original = bool(re.search(r'ignore|instruction|system|override|bypass', text.lower()))
            features.append(1.0 if has_rot13_injection and not has_original else 0.0)
        except Exception:
            features.append(0.0)

        # 3. Mixed Unicode scripts (homoglyph attack) — binary
        features.append(1.0 if self._detect_mixed_scripts(text) else 0.0)

        # 4. Unicode confusables (lookalike characters) — binary
        confusable_pairs = [
            ('\u0430', 'a'), ('\u0435', 'e'), ('\u043e', 'o'), ('\u0440', 'p'),
            ('\u0441', 'c'), ('\u0445', 'x'), ('\u0443', 'y'),
        ]
        has_confusable = any(c[0] in text for c in confusable_pairs)
        features.append(1.0 if has_confusable else 0.0)

        # 5. Zero-width characters — binary
        has_zero_width = bool(re.search(r'[\u200B\u200C\u200D\u2060\uFEFF]', text))
        features.append(1.0 if has_zero_width else 0.0)

        # QUALITY SIGNALS (high = clean/self)
        # 6. Standard ASCII/UTF-8 only — binary
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
        features.append(1.0 if ascii_ratio > 0.95 else 0.0)

        # 7. Single script — binary
        features.append(1.0 if not self._detect_mixed_scripts(text) else 0.0)

        # 8. No encoded blocks — binary
        has_encoded = bool(re.search(r'[A-Za-z0-9+/=]{40,}', text))
        features.append(1.0 if not has_encoded else 0.0)

        # 9. No zero-width chars — binary
        features.append(1.0 if not has_zero_width else 0.0)

        # 10. Normal character distribution — binary
        if len(text) > 10:
            unique_ratio = len(set(text)) / len(text)
            features.append(1.0 if 0.1 < unique_ratio < 0.8 else 0.0)
        else:
            features.append(0.5)

        return np.array(features[:10], dtype=np.float32)

    def _rule_based_check(self, text: str) -> PromptInjectionAntibodyResult:
        if self._detect_base64(text):
            return PromptInjectionAntibodyResult(
                component="EncodingEvasion", is_anomaly=True, confidence=0.9,
                reason="Base64-encoded instructions detected",
            )

        if self._detect_mixed_scripts(text):
            return PromptInjectionAntibodyResult(
                component="EncodingEvasion", is_anomaly=True, confidence=0.8,
                reason="Mixed Unicode scripts detected (possible homoglyph attack)",
            )

        has_zero_width = bool(re.search(r'[\u200B\u200C\u200D\u2060\uFEFF]', text))
        if has_zero_width:
            return PromptInjectionAntibodyResult(
                component="EncodingEvasion", is_anomaly=True, confidence=0.8,
                reason="Zero-width Unicode characters detected",
            )

        return PromptInjectionAntibodyResult(
            component="EncodingEvasion", is_anomaly=False, confidence=0.7,
            reason="No encoding evasion detected",
        )


# ---------------------------------------------------------------------------
# Antibody 4: Context Overflow Antibody
# ---------------------------------------------------------------------------

class ContextOverflowAntibody(BasePromptInjectionAntibody):
    """
    Detects context overflow attacks: repeated text padding, excessive length,
    adversarial suffixes.

    Red flags: Repeated text padding, >10k tokens, adversarial suffix patterns
    Quality signals: Normal length, proportional content
    """

    def __init__(self):
        super().__init__("ContextOverflow", num_detectors=30)

    def _generate_self_examples(self) -> List[str]:
        return [
            "Can you help me write a Python function to parse CSV files?",
            "What are the main differences between REST and GraphQL APIs?",
            "Please explain how neural networks learn through backpropagation.",
            "I need help debugging a segmentation fault in my C program.",
            "What is the time complexity of quicksort in the average case?",
            "How do I implement authentication in a Flask web application?",
            "Explain the CAP theorem and its implications for distributed databases.",
            "What are the best practices for writing unit tests in JavaScript?",
            "Help me understand the difference between threads and processes.",
            "Can you review my SQL query for optimization opportunities?",
        ]

    def extract_features(self, text: str) -> np.ndarray:
        features = []

        # Approximate token count (rough: 4 chars per token)
        approx_tokens = len(text) / 4

        # RED FLAGS
        # 1. Excessive length (>10k tokens) — binary
        features.append(1.0 if approx_tokens > 10000 else 0.0)

        # 2. High repetition ratio — binary
        words = text.lower().split()
        if len(words) >= 20:
            from collections import Counter
            word_counts = Counter(words)
            most_common_ratio = word_counts.most_common(1)[0][1] / len(words) if words else 0
            features.append(1.0 if most_common_ratio > 0.3 else 0.0)
        else:
            features.append(0.0)

        # 3. Repeated character sequences (padding) — binary
        has_padding = bool(re.search(r'(.{5,}?)\1{5,}', text))
        features.append(1.0 if has_padding else 0.0)

        # 4. Adversarial suffix pattern (random-looking characters at end) — binary
        if len(text) > 100:
            suffix = text[-100:]
            # High entropy + low word-likeness = adversarial suffix
            alpha_ratio = sum(1 for c in suffix if c.isalpha()) / len(suffix)
            space_ratio = sum(1 for c in suffix if c.isspace()) / len(suffix)
            features.append(1.0 if alpha_ratio < 0.5 and space_ratio < 0.1 else 0.0)
        else:
            features.append(0.0)

        # 5. Content-to-instruction ratio imbalanced — binary
        instruction_count = len(re.findall(r'(?:ignore|disregard|override|forget|bypass)', text.lower()))
        content_length = len(text)
        features.append(1.0 if instruction_count > 0 and content_length > 5000 else 0.0)

        # QUALITY SIGNALS (high = clean/self)
        # 6. Normal length (<500 tokens) — binary
        features.append(1.0 if approx_tokens < 500 else 0.0)

        # 7. Low repetition — binary
        if len(words) >= 10:
            unique_ratio = len(set(words)) / len(words)
            features.append(1.0 if unique_ratio > 0.5 else 0.0)
        else:
            features.append(1.0)

        # 8. Proportional content (words vs chars) — binary
        word_density = len(words) / max(len(text), 1) if text else 0
        features.append(1.0 if 0.1 < word_density < 0.3 else 0.0)

        # 9. No padding patterns — binary
        features.append(1.0 if not has_padding else 0.0)

        # 10. Reasonable token count — binary
        features.append(1.0 if 1 < approx_tokens < 2000 else 0.0)

        return np.array(features[:10], dtype=np.float32)

    def _rule_based_check(self, text: str) -> PromptInjectionAntibodyResult:
        approx_tokens = len(text) / 4

        if approx_tokens > 10000:
            return PromptInjectionAntibodyResult(
                component="ContextOverflow", is_anomaly=True, confidence=0.85,
                reason=f"Excessive input length (~{int(approx_tokens)} tokens)",
            )

        has_padding = bool(re.search(r'(.{5,}?)\1{5,}', text))
        if has_padding:
            return PromptInjectionAntibodyResult(
                component="ContextOverflow", is_anomaly=True, confidence=0.9,
                reason="Repeated text padding detected",
            )

        words = text.lower().split()
        if len(words) >= 20:
            from collections import Counter
            word_counts = Counter(words)
            most_common_word, most_common_count = word_counts.most_common(1)[0]
            if most_common_count / len(words) > 0.3:
                return PromptInjectionAntibodyResult(
                    component="ContextOverflow", is_anomaly=True, confidence=0.8,
                    reason=f"High word repetition: '{most_common_word}' appears {most_common_count}/{len(words)} times",
                )

        return PromptInjectionAntibodyResult(
            component="ContextOverflow", is_anomaly=False, confidence=0.8,
            reason="Input length and structure appear normal",
        )


# ---------------------------------------------------------------------------
# Combined System
# ---------------------------------------------------------------------------

class PromptInjectionAntibodySystem:
    """
    Multi-antibody system for prompt injection detection.

    Checks text for indirect injection, jailbreak attempts, encoding evasion,
    and context overflow attacks.
    """

    def __init__(self):
        self.antibodies: Dict[str, BasePromptInjectionAntibody] = {
            "indirect_injection": IndirectInjectionAntibody(),
            "jailbreak": JailbreakAntibody(),
            "encoding_evasion": EncodingEvasionAntibody(),
            "context_overflow": ContextOverflowAntibody(),
        }
        self.fusion = ImmuneSignalFusion(domain="security")

    def train_antibody(self, component: str, valid_examples: List[str]):
        if component not in self.antibodies:
            raise ValueError(f"Unknown component: {component}. Valid: {list(self.antibodies.keys())}")
        self.antibodies[component].train(valid_examples)

    def verify_prompt_safety(self, text: str) -> PromptInjectionResult:
        """Verify prompt/text for injection attacks using all antibodies."""
        results: Dict[str, PromptInjectionAntibodyResult] = {}
        anomaly_count = 0
        total_checks = 0
        bindings = []

        for component, antibody in self.antibodies.items():
            result = antibody.check(text)
            results[component] = result
            total_checks += 1
            bindings.append(result.binding_affinity)
            if result.is_anomaly:
                anomaly_count += 1

        if total_checks == 0:
            response = ImmuneResponse.REJECT
            overall_confidence = 1.0
        else:
            fusion_result = self.fusion.fuse_signals(bindings)
            overall_confidence = fusion_result.fused_binding
            response = fusion_result.response

        is_suspicious = response != ImmuneResponse.IGNORE

        return PromptInjectionResult(
            is_suspicious=is_suspicious,
            overall_confidence=overall_confidence,
            response=response,
            component_results=results,
            anomaly_count=anomaly_count,
            total_checks=total_checks,
        )

    def save_all(self, directory: str):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        for name, antibody in self.antibodies.items():
            antibody.save_state(str(path / f"{name}_injection_antibody.pkl"))

    def load_all(self, directory: str):
        path = Path(directory)
        for name in self.antibodies.keys():
            antibody_path = path / f"{name}_injection_antibody.pkl"
            if antibody_path.exists():
                try:
                    with open(antibody_path, "rb") as f:
                        state = pickle.load(f)
                except (pickle.UnpicklingError, ModuleNotFoundError, AttributeError, EOFError) as e:
                    print(f"Warning: Could not load {antibody_path}: {e}", file=sys.stderr)
                    continue
                self.antibodies[name].patterns = state.get("patterns", [])
                self.antibodies[name].is_trained = state.get("is_trained", False)
                if state.get("nk_detector"):
                    self.antibodies[name].nk_detector = state["nk_detector"]

    def get_training_status(self) -> Dict[str, bool]:
        return {name: ab.is_trained for name, ab in self.antibodies.items()}


def create_prompt_injection_antibody_system() -> PromptInjectionAntibodySystem:
    """Create a new prompt injection antibody system."""
    return PromptInjectionAntibodySystem()
