"""Adaptador del bus sobre la mensajería pública de AgentScope 2.x.

AgentScope 2.x ya no ofrece ``MsgHub``: la unidad de mensajería es ``Msg`` y
los agentes reciben observaciones con ``Agent.observe``. Este adaptador:

- codifica cada ``MetadataEnvelope`` validado como ``Msg`` (con el envelope
  completo en ``metadata``) y lo decodifica al entregarlo, de modo que el
  contrato sobre el que viaja es el de AgentScope;
- reenvía el ``Msg`` a los agentes AgentScope adjuntos (``attach_agent``),
  típicamente los ReAct (``event`` y ``supervisor``) que razonan sobre la
  metadata. Cada agente tiene su cola acotada y su tarea: un agente lento
  nunca bloquea a quien publica (ni al hilo del pipeline);
- no aplica reglas de negocio, deduplicación ni reintentos.

``agentscope`` se importa de forma diferida: cargarlo cuesta segundos y los
módulos que solo necesitan el contrato no deberían pagarlo.
"""

from __future__ import annotations

import asyncio
import json
import logging
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


class _ObserverPump:
    """Cola acotada y tarea propia que entrega ``Msg`` a un agente."""

    def __init__(self, agent: MsgObserver, size: int, stats: dict[str, int]) -> None:
        self.agent = agent
        self.queue: asyncio.Queue[Msg | None] = asyncio.Queue(maxsize=size)
        self._stats = stats
        self._task: asyncio.Task[None] | None = None

    def offer(self, msg: Msg) -> None:
        if self._task is None:
            self._task = asyncio.create_task(
                self._run(), name=f"observer:{self.agent.name}"
            )
        if self.queue.full():  # el contexto del agente es "última lectura gana"
            self.queue.get_nowait()
            self.queue.task_done()
            self._stats["observer_dropped"] += 1
            logger.warning(
                "Cola saturada del agente '%s'; se descartó un Msg", self.agent.name
            )
        self.queue.put_nowait(msg)

    async def _run(self) -> None:
        while True:
            msg = await self.queue.get()
            try:
                if msg is None:
                    return
                await self.agent.observe(msg)
            except Exception:
                self._stats["observer_errors"] += 1
                logger.exception("El agente '%s' falló al observar", self.agent.name)
            else:
                self._stats["observed"] += 1
            finally:
                self.queue.task_done()

    async def stop(self) -> None:
        if self._task is None:
            return
        await self.queue.join()
        self.queue.put_nowait(None)
        await self._task
        self._task = None


class AgentScopeHub(InMemoryHub):
    """``MetadataHub`` cuyo transporte de mensajes es ``agentscope.message.Msg``."""

    def __init__(
        self,
        *,
        queue_size: int = 1024,
        history_size: int = 1000,
        observer_queue_size: int = 256,
    ) -> None:
        super().__init__(queue_size=queue_size, history_size=history_size)
        self._observer_queue_size = observer_queue_size
        self._pumps: dict[int, _ObserverPump] = {}
        self._topic_pumps: dict[Topic, list[_ObserverPump]] = {}
        self.stats.update({"observed": 0, "observer_errors": 0, "observer_dropped": 0})

    @staticmethod
    def to_msg(
        topic: Topic, envelope: MetadataEnvelope, *, validate: bool = True
    ) -> Msg:
        """Codifica un envelope como ``Msg`` de AgentScope.

        ``validate=False`` omite la validación para envelopes que ``publish``
        ya validó.
        """
        from agentscope.message import Msg, TextBlock

        data = envelope_to_dict(topic, envelope, validate=validate)
        return Msg(
            name=envelope.source,
            role="assistant",
            content=[TextBlock(text=json.dumps(data, separators=(",", ":")))],
            id=envelope.event_id,
            metadata={"topic": data["topic"], ENVELOPE_KEY: data},
        )

    @staticmethod
    def from_msg(msg: Msg) -> tuple[Topic, MetadataEnvelope]:
        """Decodifica un ``Msg`` (entrada externa: siempre se valida)."""
        data = msg.metadata.get(ENVELOPE_KEY)
        if not isinstance(data, dict):
            raise InvalidEnvelopeError(
                "El Msg no contiene un envelope de Condor Eye "
                f"(falta metadata['{ENVELOPE_KEY}'])"
            )
        return envelope_from_dict(data)

    def attach_agent(self, agent: MsgObserver, topics: Iterable[Topic]) -> None:
        """Hace que ``agent.observe`` reciba los ``Msg`` de los tópicos dados."""
        pump = self._pumps.setdefault(
            id(agent),
            _ObserverPump(agent, self._observer_queue_size, self.stats),
        )
        for topic in topics:
            peers = self._topic_pumps.setdefault(Topic(topic), [])
            if pump not in peers:
                peers.append(pump)

    async def _dispatch(self, topic: Topic, message: MetadataEnvelope) -> None:
        await super()._dispatch(topic, message)
        pumps = self._topic_pumps.get(topic)
        if pumps:
            msg = self.to_msg(topic, message, validate=False)
            for pump in pumps:
                pump.offer(msg)

    async def flush(self) -> None:
        """Espera a que los agentes adjuntos procesen todo lo encolado."""
        await asyncio.gather(*(p.queue.join() for p in self._pumps.values()))

    async def close(self) -> None:
        """Cierra el hub y vacía a los agentes adjuntos antes de terminar."""
        await super().close()
        await asyncio.gather(*(p.stop() for p in self._pumps.values()))
