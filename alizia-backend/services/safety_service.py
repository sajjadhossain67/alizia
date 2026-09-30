"""Alizia AI Backend - Safety Service"""

from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime


class RiskLevel(Enum):
    """Risk levels for tool actions."""
    R0_INFORMATIONAL = 0
    R1_REVERSIBLE = 1
    R2_EXTERNALLY_VISIBLE = 2
    R3_SENSITIVE = 3
    R4_DESTRUCTIVE = 4


class SafetyClassification(str, Enum):
    """Safety classification for requests."""
    SAFE = "safe"
    WARNING = "warning"
    BLOCKED = "blocked"
    REVIEW = "review"


class SafetyPolicy:
    """Safety policy configuration."""
    
    # Default safety policies - these can be overridden per-organization
    default_policies = {
        "allow_web_search": True,
        "allow_code_execution": True,
        "allow_file_operations": True,
        "allow_browser_control": False,
        "allow_database_write": False,
        "allow_email_sending": False,
        "allow_shell_execution": True,
    }
    
    # Tool risk mappings
    tool_risk_map = {
        "web.search": RiskLevel.R0_INFORMATIONAL,
        "web.fetch": RiskLevel.R0_INFORMATIONAL,
        "code.execute": RiskLevel.R2_EXTERNALLY_VISIBLE,
        "shell.execute": RiskLevel.R2_EXTERNALLY_VISIBLE,
        "file.read": RiskLevel.R1_REVERSIBLE,
        "file.write": RiskLevel.R2_EXTERNALLY_VISIBLE,
        "file.search": RiskLevel.R1_REVERSIBLE,
        "python.execute": RiskLevel.R2_EXTERNALLY_VISIBLE,
        "database.query": RiskLevel.R1_REVERSIBLE,
        "database.modify": RiskLevel.R3_SENSITIVE,
        "database.delete": RiskLevel.R4_DESTRUCTIVE,
        "email.send": RiskLevel.R3_SENSITIVE,
        "email.read": RiskLevel.R2_EXTERNALLY_VISIBLE,
        "browser.open": RiskLevel.R1_REVERSIBLE,
        "browser.click": RiskLevel.R1_REVERSIBLE,
        "browser.type": RiskLevel.R1_REVERSIBLE,
    }
    
    # Safety directives by domain
    safety_directives = {
        "malware": "BLOCK",
        "violence": "BLOCK",
        "self-harm": "BLOCK",
        "sexual-explicit": "BLOCK",
        "fraud": "BLOCK",
        "privacy-abuse": "BLOCK",
        "credential-theft": "BLOCK",
    }


class SafetyEngine:
    """Engine for safety classification and policy enforcement."""
    
    def __init__(self, policy: Optional[SafetyPolicy] = None):
        self.policy = policy or SafetyPolicy()
    
    def classify_request(self, text: str) -> SafetyClassification:
        """Classify a request for safety concerns."""
        text_lower = text.lower()
        
        # Check against known dangerous categories
        for category, directive in self.policy.safety_directives.items():
            if category in text_lower:
                if directive == "BLOCK":
                    return SafetyClassification.BLOCKED
                elif directive == "WARNING":
                    return SafetyClassification.WARNING
        
        # Default: safe if no known patterns match
        return SafetyClassification.SAFE
    
    def assess_tool_risk(self, tool_name: str) -> RiskLevel:
        """Assess the risk level of a tool."""
        return self.policy.tool_risk_map.get(tool_name, RiskLevel.R0_INFORMATIONAL)
    
    def check_policy(self, tool_name: str, action: str) -> bool:
        """Check if a tool action is allowed per policy."""
        allowed = self.policy.default_policies.get(tool_name, True)
        # In production, also check user/organization policies
        return allowed
    
    def evaluate(self, content: str, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Full safety evaluation of content and tool request."""
        safety_class = self.classify_request(content)
        risk_level = self.assess_tool_risk(tool_name) if tool_name else RiskLevel.R0_INFORMATIONAL
        policy_allowed = self.check_policy(tool_name, "") if tool_name else True
        
        return {
            "safety_classification": safety_class.value,
            "risk_level": risk_level.name,
            "risk_level_value": risk_level.value,
            "policy_allowed": policy_allowed,
            "content_safe": safety_class == SafetyClassification.SAFE,
            "timestamp": datetime.utcnow().isoformat(),
        }


# Global safety engine instance
safety_engine = SafetyEngine()


async def get_safety_engine() -> SafetyEngine:
    """Dependency to get the safety engine instance."""
    return safety_engine