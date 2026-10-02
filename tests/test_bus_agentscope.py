"""Pruebas del adaptador del bus sobre la mensajería de AgentScope 2.x."""

from unittest.mock import MagicMock

import pytest

from bus import (
    InvalidEnvelopeError,
    InvalidTopicError,
    MetadataEnvelope,
    Topic,
)
from bus.agentscope_hub import AgentScopeHub


def env(event_id: str = "e1") -> MetadataEnvelope:
    return MetadataEnvelope(
        source="tracker",
        payload={"track_id": 7},
        stream_id="cam-01",
        event_id=event_id,
        correlation_id="corr-1",
    )


def test_envelope_is_encoded_as_an_agentscope_msg():
    from agentscope.message import Msg

    msg = AgentScopeHub.to_msg(Topic.TRACKS, env())

    assert isinstance(msg, Msg)
    assert msg.id == "e1"
    assert msg.name == "tracker"
    assert msg.metadata["topic"] == "vision.tracks"
    assert msg.metadata["condor_envelope"]["correlation_id"] == "corr-1"


def test_msg_roundtrip_returns_the_same_envelope():
    original = env()
    topic, restored = AgentScopeHub.from_msg(
        AgentScopeHub.to_msg(Topic.TRACKS, original)
    )
    assert topic is Topic.TRACKS
    assert restored == original


def test_foreign_msg_is_rejected_with_clear_error():
    from agentscope.message import Msg, TextBlock

    foreign = Msg(name="x", role="assistant", content=[TextBlock(text="hola")])
    with pytest.raises(InvalidEnvelopeError, match="envelope de Condor Eye"):
        AgentScopeHub.from_msg(foreign)


async def test_subscribers_receive_envelopes_through_msg_transport():
    hub = AgentScopeHub()
    sub = hub.subscribe(Topic.TRACKS)
    original = env()
    await hub.publish(Topic.TRACKS, original)
    await hub.close()
    assert [m async for m in sub] == [original]


async def test_invalid_input_is_rejected_before_building_a_msg():
    hub = AgentScopeHub()
    with pytest.raises(InvalidTopicError):
        await hub.publish("video.frames", env())  # type: ignore[arg-type]
    with pytest.raises(InvalidEnvelopeError, match="binarios"):
        await hub.publish(
            Topic.EVENTS, MetadataEnvelope(source="x", payload={"frame": b"raw"})
        )


async def test_attached_real_agent_observes_msgs_of_its_topics_only():
    from agentscope.agent import Agent

    agent = Agent(name="event", system_prompt="reglas", model=MagicMock())
    hub = AgentScopeHub()
    hub.attach_agent(agent, [Topic.TRACKS])

    await hub.publish(Topic.TRACKS, env("t1"))
    await hub.publish(Topic.HEALTH, env("h1"))

    assert len(agent.state.context) == 1
    assert agent.state.context[0].id == "t1"
    assert hub.stats["observed"] == 1


async def test_observer_failure_does_not_block_other_subscribers():
    class Broken:
        name = "broken"

        async def observe(self, msgs=None):
            raise RuntimeError("fallo del agente")

    hub = AgentScopeHub()
    hub.attach_agent(Broken(), [Topic.EVENTS])
    sub = hub.subscribe(Topic.EVENTS)

    await hub.publish(Topic.EVENTS, env("e1"))
    await hub.close()

    assert [m.event_id async for m in sub] == ["e1"]
    assert hub.stats["observer_errors"] == 1
