from __future__ import annotations

from typing import Dict, Any
from abc import ABC, abstractmethod


class BaseStage(ABC):
    """
    Base class for all workflow stages.

    Enforces a consistent interface for stage execution.
    """

    @abstractmethod
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the stage.

        Each stage must implement this method.
        """
        pass