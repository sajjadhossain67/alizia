"""Alizia AI Backend - Model Service with full model routing"""

from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime

from core.config import settings


class ModelInfo:
    """Model metadata stored in the registry."""
    
    def __init__(
        self,
        model_id: str,
        name: str,
        model_type: str = "text",
        context_window: Optional[int] = None,
        modalities: Optional[List[str]] = None,
        cost_per_input_token: Optional[float] = None,
        cost_per_output_token: Optional[float] = None,
        cost_per_reasoning_token: Optional[float] = None,
    ):
        self.model_id = model_id
        self.name = name
        self.type = model_type
        self.context_window = context_window
        self.modalities = modalities or []
        self.cost_per_input_token = cost_per_input_token or 0.0
        self.cost_per_output_token = cost_per_output_token or 0.0
        self.cost_per_reasoning_token = cost_per_reasoning_token or 0.0
        self.status = "available"
        self.created_at = datetime.utcnow()


class ModelRouter:
    """Routes requests to appropriate models based on task type and requirements."""
    
    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register the default Alizia model family."""
        # Alizia Nova - flagship reasoning model
        self.register(ModelInfo(
            model_id="alizia-nova",
            name="Alizia Nova",
            model_type="reasoning",
            context_window=1_000_000,
            modalities=["text", "image", "document"],
            cost_per_input_token=settings.PRICING.get("alizia-nova", {}).get("input", 0.001),
            cost_per_output_token=settings.PRICING.get("alizia-nova", {}).get("output", 0.002),
            cost_per_reasoning_token=settings.PRICING.get("alizia-nova", {}).get("reasoning", 0.003),
        ))
        
        # Alizia Pulse - fast general-purpose
        self.register(ModelInfo(
            model_id="alizia-pulse",
            name="Alizia Pulse",
            model_type="general",
            context_window=128_000,
            modalities=["text"],
            cost_per_input_token=settings.PRICING.get("alizia-pulse", {}).get("input", 0.0002),
            cost_per_output_token=settings.PRICING.get("alizia-pulse", {}).get("output", 0.0005),
            cost_per_reasoning_token=settings.PRICING.get("alizia-pulse", {}).get("reasoning", 0.001),
        ))
        
        # Alizia Forge - coding specialist
        self.register(ModelInfo(
            model_id="alizia-forge",
            name="Alizia Forge",
            model_type="coding",
            context_window=128_000,
            modalities=["text"],
            cost_per_input_token=settings.PRICING.get("alizia-forge", {}).get("input", 0.0015),
            cost_per_output_token=settings.PRICING.get("alizia-forge", {}).get("output", 0.003),
            cost_per_reasoning_token=settings.PRICING.get("alizia-forge", {}).get("reasoning", 0.005),
        ))
        
        # Alizia Vision - multimodal
        self.register(ModelInfo(
            model_id="alizia-vision",
            name="Alizia Vision",
            model_type="multimodal",
            context_window=8192,
            modalities=["text", "image"],
            cost_per_input_token=settings.PRICING.get("alizia-vision", {}).get("input", 0.002),
            cost_per_output_token=settings.PRICING.get("alizia-vision", {}).get("output", 0.004),
            cost_per_reasoning_token=settings.PRICING.get("alizia-vision", {}).get("reasoning", 0.006),
        ))
        
        # Alizia Edge - small model
        self.register(ModelInfo(
            model_id="alizia-edge",
            name="Alizia Edge",
            model_type="efficient",
            context_window=8192,
            modalities=["text"],
            cost_per_input_token=settings.PRICING.get("alizia-edge", {}).get("input", 0.0001),
            cost_per_output_token=settings.PRICING.get("alizia-edge", {}).get("output", 0.0002),
            cost_per_reasoning_token=settings.PRICING.get("alizia-edge", {}).get("reasoning", 0.0005),
        ))
        
        # Alizia Embed - embeddings
        self.register(ModelInfo(
            model_id="alizia-embed-v1",
            name="Alizia Embed v1",
            model_type="embedding",
            context_window=8192,
            modalities=["text"],
        ))
    
    def register(self, model: ModelInfo) -> None:
        """Register a model in the router."""
        self._models[model.model_id] = model
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information by ID."""
        return self._models.get(model_id)
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models for API endpoints."""
        return [
            {
                "id": m.model_id,
                "name": m.name,
                "type": m.type,
                "context_window": m.context_window,
                "modalities": m.modalities,
                "status": m.status,
                "cost_per_input_token": m.cost_per_input_token,
                "cost_per_output_token": m.cost_per_output_token,
            }
            for m in self._models.values()
        ]
    
    def route(
        self,
        task_type: str,
        modalities: Optional[List[str]] = None,
        context_window_needed: Optional[int] = None,
        cost_ceiling: Optional[float] = None,
        preferred_model: Optional[str] = None,
    ) -> str:
        """Select the best model for a given task.
        
        Args:
            task_type: Type of task (reasoning, coding, general, multimodal, etc.)
            modalities: Required input modalities
            context_window_needed: Minimum context window required
            cost_ceiling: Maximum cost per token acceptable
            preferred_model: User-requested model if any
        
        Returns:
            Model ID to use for the request
        """
        # If user specified a preferred model, use it if available
        if preferred_model and preferred_model in self._models:
            return preferred_model
        
        # Task-type based routing
        task_routing = {
            "reasoning": "alizia-nova",
            "complex": "alizia-nova",
            "analysis": "alizia-nova",
            "business-analysis": "alizia-nova",
            "science": "alizia-nova",
            "research": "alizia-nova",
            "agent": "alizia-nova",
            "coding": "alizia-forge",
            "software-engineering": "alizia-forge",
            "debugging": "alizia-forge",
            "testing": "alizia-forge",
            "refactoring": "alizia-forge",
            "general": "alizia-pulse",
            "chat": "alizia-pulse",
            "classification": "alizia-pulse",
            "extraction": "alizia-pulse",
            "summarization": "alizia-pulse",
            "customer-support": "alizia-pulse",
            "multimodal": "alizia-vision",
            "vision": "alizia-vision",
            "image": "alizia-vision",
            "document": "alizia-vision",
            "chart": "alizia-vision",
            "diagram": "alizia-vision",
            "embedding": "alizia-embed-v1",
            "semantic-search": "alizia-embed-v1",
            "efficient": "alizia-edge",
            "low-cost": "alizia-edge",
        }
        
        # Default to Nova for unknown types (flagship)
        suggested = task_routing.get(task_type, "alizia-nova")
        
        # Check if the suggested model meets requirements
        model = self._models.get(suggested)
        if not model:
            suggested = "alizia-nova"
        
        # Cost check
        if cost_ceiling and model and (
            (model.cost_per_input_token and model.cost_per_input_token > cost_ceiling)
            or (model.cost_per_output_token and model.cost_per_output_token > cost_ceiling)
        ):
            # Fall back to cheaper model
            for mid, m in sorted(self._models.items(), key=lambda x: x[1].cost_per_input_token or 0):
                if not cost_ceiling or (model.cost_per_input_token and model.cost_per_input_token <= cost_ceiling):
                    suggested = mid
                    break
        
        # Context window check
        if context_window_needed and self._models.get(suggested):
            if self._models[suggested].context_window and self._models[suggested].context_window < context_window_needed:
                # Need a model with larger context
                for mid, m in self._models.items():
                    if m.context_window and m.context_window >= context_window_needed:
                        suggested = mid
                        break
        
        return suggested


# Global router instance
model_router = ModelRouter()


async def get_available_models() -> List[Dict[str, Any]]:
    """FastAPI dependency to get available models."""
    return model_router.get_available_models()


async def route_model(
    task_type: str,
    modalities: Optional[List[str]] = None,
    context_window_needed: Optional[int] = None,
    cost_ceiling: Optional[float] = None,
    preferred_model: Optional[str] = None,
) -> str:
    """FastAPI dependency to get the routed model ID."""
    return model_router.route(
        task_type=task_type,
        modalities=modalities,
        context_window_needed=context_window_needed,
        cost_ceiling=cost_ceiling,
        preferred_model=preferred_model,
    )