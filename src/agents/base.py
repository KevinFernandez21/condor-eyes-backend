"""Definiciones compartidas para construir agentes AgentScope."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from bus.hub import Topic

if TYPE_CHECKING:
    from agentscope.agent import Agent
    from agentscope.model import ChatModelBase
    from agentscope.tool import Toolkit


class AgentKind(StrEnum):
    """Tipo de agente definido por la arquitectura."""

    TOOL = "tool"
    REACT = "react"


@dataclass(frozen=True, slots=True)
class AgentRoute:
    """Contrato estático de un agente dentro de la ruta multiagente."""

    name: str
    kind: AgentKind
    purpose: str
    singleton: bool
    consumes: tuple[Topic, ...]
    publishes: tuple[Topic, ...]

    def build(self, model: ChatModelBase, toolkit: Toolkit | None = None) -> Agent:
        """Construye el agente con la API pública de AgentScope 2.x."""
        from agentscope.agent import Agent

        return Agent(
            name=self.name,
            system_prompt=self.purpose,
            model=model,
            toolkit=toolkit,
        )
