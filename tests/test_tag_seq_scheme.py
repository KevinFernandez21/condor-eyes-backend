"""El esquema de `seq` del firmware sobrevive a reinicios frente al anti-replay de #16.

`TagSeqModel` replica en Python la lógica de `firmware/xiao_c6_tag/xiao_c6_tag.ino`:
al arrancar lee de NVS el inicio del siguiente bloque, lo usa como `seq` inicial y
reserva `seq + BLOCK`; cada vez que `seq` alcanza el final del bloque reservado,
reserva otro. El valor guardado siempre es mayor que cualquier `seq` ya emitido.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path

from location import (
    InMemoryPersonnelRepository,
    LocationService,
    PersonnelRecord,
    Pseudonymizer,
    RejectReason,
    load_config,
)

TAG = "58E6C515220A"
BLOCK = 1000
T0 = 1767268800.0
LOCATION_CFG = Path(__file__).resolve().parents[1] / "configs" / "location.toml"


class TagSeqModel:
    def __init__(self, nvs: dict[str, int], block: int = BLOCK) -> None:
        self.nvs = nvs
        self.block = block
        self.seq = nvs.get("seqblk", 0)
        self.block_end = self.seq + block
        nvs["seqblk"] = self.block_end

    def next(self) -> int:
        if self.seq >= self.block_end:
            self.block_end += self.block
            self.nvs["seqblk"] = self.block_end
        value = self.seq
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        return value


class Harness:
    def __init__(self) -> None:
        self.t = T0
        pseudo = Pseudonymizer(b"k")
        repo = InMemoryPersonnelRepository(pseudo)
        repo.enroll(TAG, PersonnelRecord("p", "P", frozenset({"lobby"})))
        self.svc = LocationService(
            load_config(LOCATION_CFG),
            repo,
            pseudo,
            clock=lambda: datetime.fromtimestamp(self.t, UTC),
        )

    def send(self, seq: int) -> bool:
        self.t += 0.1
        raw = json.dumps(
            {
                "v": 1,
                "ch": "ble",
                "tag": TAG,
                "node": "N0001",
                "rssi": -60,
                "ts": round(self.t * 1000),
                "seq": seq,
            }
        )
        return self.svc.ingest_json(raw).accepted


def run(h: Harness, tag: TagSeqModel, n: int) -> list[bool]:
    return [h.send(tag.next()) for _ in range(n)]


def test_seq_continues_across_reboots_and_all_are_accepted():
    h, nvs = Harness(), {}
    last = -1
    for boot_len in (250, 1, 999, 1000, 37):  # reinicios en puntos arbitrarios
        tag = TagSeqModel(nvs)
        first = tag.seq
        assert first > last
        assert all(run(h, tag, boot_len))
        last = tag.seq - 1
    assert not h.svc.stats


def test_many_random_reboots_never_rejected():
    h, nvs, rng = Harness(), {}, random.Random(7)
    for _ in range(200):
        assert all(run(h, TagSeqModel(nvs), rng.randint(1, 2500)))
    assert not h.svc.stats


def test_saved_block_is_always_ahead_of_emitted_seq():
    nvs: dict[str, int] = {}
    tag = TagSeqModel(nvs)
    for _ in range(5000):
        s = tag.next()
        assert nvs["seqblk"] > s


def test_old_random_start_scheme_is_rejected_after_reboot():
    """Documenta el bug: reiniciar con un seq aleatorio bajo se toma como replay."""
    h = Harness()
    for s in range(5_000_000, 5_000_020):
        assert h.send(s)
    assert h.send(100) is False
    assert h.svc.stats[RejectReason.REPLAY] == 1
