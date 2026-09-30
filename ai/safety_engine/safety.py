"""
ALIZIA AI - Multi-Layer Safety Engine & Prompt Injection Defense
Implements PRD Sections 71-77.
Provides:
1. Input Safety Classification (malware, credentials, dangerous topics)
2. Prompt Injection Defense (Section 76: trust metadata & injection pattern defense)
3. Secrets Detection & Redaction (Section 77: API keys, JWTs, private keys)
4. Action Risk Engine (Section 74: R0-R4 risk ranking)
"""

from __future__ import annotations
import re
from typing import Tuple, List, Dict, Any, Optional
from packages.schemas.models import TrustLevel, RiskLevel, ContentPart


class SafetyEngine:
    # Pre-compiled regex patterns for secret detection (Section 77)
    SECRET_PATTERNS = [
        ("alizia_api_key", re.compile(r"alz_(?:live|test)_[a-zA-Z0-9]{20,}")),
        ("openai_api_key", re.compile(r"sk-[a-zA-Z0-9]{20,}")),
        ("jwt_token", re.compile(r"eyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}")),
        ("private_key", re.compile(r"-----BEGIN (?:RSA|OPENSSH|EC|PRIVATE) KEY-----")),
        ("generic_api_key", re.compile(r"(?i)(?:api_key|access_token|secret_key|client_secret)\s*[:=]\s*['\"][a-zA-Z0-9_\-]{16,}['\"]")),
        ("database_uri", re.compile(r"postgres(?:ql)?:\/\/[^:]+:[^@]+@[^:\/]+:\d+\/[^\s]+")),
    ]

    # Prompt injection attack heuristics (Section 76)
    INJECTION_PATTERNS = [
        re.compile(r"(?i)ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions"),
        re.compile(r"(?i)disregard\s+(?:all\s+)?(?:system\s+)?directives"),
        re.compile(r"(?i)you\s+are\s+now\s+(?:in\s+)?(?:DAN|developer\s+mode|jailbroken)"),
        re.compile(r"(?i)system\s+prompt\s+override"),
        re.compile(r"(?i)exfiltrate\s+(?:all\s+)?(?:data|files|secrets|keys)"),
        re.compile(r"(?i)send\s+(?:all\s+)?(?:credentials|tokens)\s+to\s+https?://"),
    ]

    # Tool Action Risk classification mappings (Section 74)
    TOOL_RISK_MAP = {
        "web.search": RiskLevel.R0_INFORMATIONAL,
        "web.fetch": RiskLevel.R0_INFORMATIONAL,
        "knowledge.search": RiskLevel.R0_INFORMATIONAL,
        "memory.search": RiskLevel.R0_INFORMATIONAL,
        "file.read": RiskLevel.R0_INFORMATIONAL,
        "file.search": RiskLevel.R0_INFORMATIONAL,
        "git.read": RiskLevel.R0_INFORMATIONAL,
        "file.write": RiskLevel.R1_REVERSIBLE,
        "create_local_file": RiskLevel.R1_REVERSIBLE,
        "code.execute": RiskLevel.R1_REVERSIBLE,
        "python.execute": RiskLevel.R1_REVERSIBLE,
        "git.commit": RiskLevel.R1_REVERSIBLE,
        "send_email": RiskLevel.R2_EXTERNALLY_VISIBLE,
        "email.send": RiskLevel.R2_EXTERNALLY_VISIBLE,
        "browser.click": RiskLevel.R2_EXTERNALLY_VISIBLE,
        "git.push": RiskLevel.R2_EXTERNALLY_VISIBLE,
        "database.write": RiskLevel.R3_SENSITIVE,
        "modify_production_db": RiskLevel.R3_SENSITIVE,
        "database.delete": RiskLevel.R4_DESTRUCTIVE,
        "delete_production_db": RiskLevel.R4_DESTRUCTIVE,
    }

    @classmethod
    def scan_for_secrets(cls, text: str, action: str = "redact") -> Tuple[str, List[str]]:
        """
        Section 77: Secrets Detection.
        Configurable behavior: redact, warn, block.
        """
        found_types = []
        cleaned_text = text
        for name, pattern in cls.SECRET_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                found_types.append(name)
                if action == "redact":
                    cleaned_text = pattern.sub(f"[REDACTED_{name.upper()}]", cleaned_text)

        return cleaned_text, found_types

    @classmethod
    def validate_input_safety(cls, text: str) -> Tuple[bool, Optional[str]]:
        """
        Section 72: Input Safety classifier.
        Classifies malicious abuse, bio-hazards, fraud, credentials theft.
        """
        lower = text.lower()
        malware_indicators = ["create ransomware", "exploit payload generator", "steal session tokens", "ddos script"]
        for ind in malware_indicators:
            if ind in lower:
                return False, f"Safety rejection: request matched policy violation '{ind}'"
        return True, None

    @classmethod
    def inspect_prompt_injection(cls, content_parts: List[ContentPart]) -> Tuple[bool, Optional[str]]:
        """
        Section 76: Prompt Injection Defense.
        Ensures content tagged as UNTRUSTED_DATA (e.g. web fetch, uploaded files)
        cannot override system or user directives.
        """
        for part in content_parts:
            text = part.text or ""
            if part.trust_level == TrustLevel.UNTRUSTED_DATA:
                for pattern in cls.INJECTION_PATTERNS:
                    if pattern.search(text):
                        return False, "Prompt injection defense triggered: Untrusted source contains instruction override attempt."
        return True, None

    @classmethod
    def evaluate_tool_risk(cls, tool_name: str, arguments: Dict[str, Any]) -> RiskLevel:
        """
        Section 74: Assigns Risk R0 (Informational) to R4 (Destructive)
        """
        # Dynamic inspection of arguments (e.g. DROP TABLE, DELETE in SQL)
        if "delete" in tool_name or "drop" in tool_name:
            return RiskLevel.R4_DESTRUCTIVE

        if tool_name == "database.query":
            query = arguments.get("query", "").upper()
            if "DROP" in query or "DELETE" in query or "TRUNCATE" in query:
                return RiskLevel.R4_DESTRUCTIVE
            if "UPDATE" in query or "INSERT" in query or "ALTER" in query:
                return RiskLevel.R3_SENSITIVE
            return RiskLevel.R0_INFORMATIONAL

        return cls.TOOL_RISK_MAP.get(tool_name, RiskLevel.R1_REVERSIBLE)
