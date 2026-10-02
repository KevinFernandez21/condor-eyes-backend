"""Adaptador del bus sobre la mensajería pública de AgentScope 2.x.

AgentScope 2.x ya no ofrece ``MsgHub``: la unidad de mensajería es ``Msg`` y
los agentes reciben observaciones con ``Agent.observe``. Este adaptador:

- codifica cada ``MetadataEnvelope`` validado como ``Msg`` (con el envelope
  completo en ``metadata``) y lo decodifica al entregarlo, de modo que el
  contrato sobre el que viaja es el de AgentScope;
- reenvía el ``Msg`` a los agentes AgentScope adjuntos (``attach_agent``),
  típicamente los ReAct (``event`` y ``supervisor``) que razonan sobre la
  metadata;
- no aplica reglas de negocio, deduplicación ni reintentos.

``agentscope`` se importa de forma diferida: cargarlo cuesta segundos y los
módulos que solo necesitan el contrato no deberían pagarlo.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol

from .hub import (
    InvalidEnvelopeError,
    MetadataEnvelope,
    Topic,
    envelope_from_dict,
    envelope_to_dict,
)
from .memory import InMemoryHub

if TYPE_CHECKING:
    from agentscope.message import Msg

logger = logging.getLogger(__name__)

ENVELOPE_KEY = "condor_envelope"
"""Clave de ``Msg.metadata`` que transporta el envelope serializado."""


class MsgObserver(Protocol):
    """Lo mínimo que el hub exige a un agente AgentScope adjunto."""

    name: str

    async def observe(self, msgs: Any = None) -> None:
        """Recibe observaciones sin generar respuesta (``Agent.observe``)."""


class AgentScopeHub(InMemoryHub):
    """``MetadataHub`` cuyo transporte de mensajes es ``agentscope.message.Msg``."""

    def __init__(self, *, queue_size: int = 1024) -> None:
        super().__init__(queue_size=queue_size)
        self._observers: defaultdict[Topic, list[MsgObserver]] = defaultdict(list)
        self.stats.update({"observed": 0, "observer_errors": 0})

    @staticmethod
    def to_msg(topic: Topic, envelope: MetadataEnvelope) -> Msg:
        """Codifica un envelope como ``Msg`` de AgentScope."""
        from agentscope.message import Msg, TextBlock

        data = envelope_to_dict(topic, envelope)
        return Msg(
            name=envelope.source,
            role="assistant",
            content=[TextBlock(text=json.dumps(data, separators=(",", ":")))],
            id=envelope.event_id,
            metadata={"topic": data["topic"], ENVELOPE_KEY: data},
        )

    @staticmethod
    def from_msg(msg: Msg) -> tuple[Topic, MetadataEnvelope]:
        """Decodifica un ``Msg`` producido por ``to_msg``."""
        data = msg.metadata.get(ENVELOPE_KEY)
        if not isinstance(data, dict):
            raise InvalidEnvelopeError(
                "El Msg no contiene un envelope de Condor Eye "
                f"(falta metadata['{ENVELOPE_KEY}'])"
            )
        return envelope_from_dict(data)

    def attach_agent(self, agent: MsgObserver, topics: Iterable[Topic]) -> None:
        """Hace que ``agent.observe`` reciba los ``Msg`` de los tópicos dados."""
        for topic in topics:
            self._observers[Topic(topic)].append(agent)

    async def _dispatch(self, topic: Topic, message: MetadataEnvelope) -> None:
        msg = self.to_msg(topic, message)
        topic, decoded = self.from_msg(msg)
        await super()._dispatch(topic, decoded)
        for agent in list(self._observers[topic]):
            try:
                await agent.observe(msg)
            except Exception:  # un agente roto no debe frenar al resto
                self.stats["observer_errors"] += 1
                logger.exception(
                    "El agente '%s' falló al observar %s", agent.name, topic
                )
            else:
                self.stats["observed"] += 1
