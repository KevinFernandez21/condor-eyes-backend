"""El esquema de `seq` del firmware sobrevive a reinicios frente al anti-replay de #16.

`TagSeqModel` replica en Python la lógica de `firmware/xiao_c6_tag/xiao_c6_tag.ino`:
al arrancar lee de NVS el inicio del siguiente bloque, lo usa como `seq` inicial y
reserva `seq + BLOCK`; cada vez que `seq` alcanza el final del bloque reservado,
reserva otro. El valor guardado siempre es mayor que cualquier `seq` ya emitido.
"""

from __future__ import annotations

import json
import random
import re
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
NVS_KEY = "seqblk"
INO = (
    Path(__file__).resolve().parents[1] / "firmware" / "xiao_c6_tag" / "xiao_c6_tag.ino"
)
T0 = 1767268800.0
LOCATION_CFG = Path(__file__).resolve().parents[1] / "configs" / "location.toml"


class TagSeqModel:
    def __init__(self, nvs: dict[str, int], block: int = BLOCK) -> None:
        self.nvs = nvs
        self.block = block
        self.seq = nvs.get(NVS_KEY, 0)
        self.block_end = self.seq + block
        nvs[NVS_KEY] = self.block_end

    def next(self) -> int:
        if self.seq >= self.block_end:
            self.block_end += self.block
            self.nvs[NVS_KEY] = self.block_end
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
        assert nvs[NVS_KEY] > s


def test_old_random_start_scheme_is_rejected_after_reboot():
    """Documenta el bug: reiniciar con un seq aleatorio bajo se toma como replay."""
    h = Harness()
    for s in range(5_000_000, 5_000_020):
        assert h.send(s)
    assert h.send(100) is False
    assert h.svc.stats[RejectReason.REPLAY] == 1


def test_model_constants_match_firmware_source():
    src = INO.read_text(encoding="utf-8")
    block = re.search(r"#define\s+SEQ_BLOCK\s+(\d+)", src)
    assert block is not None and int(block.group(1)) == BLOCK
    assert f'"{NVS_KEY}"' in src  # misma clave NVS que el modelo
    assert 'prefs.getUInt("seqblk", 0)' in src  # primer arranque desde 0


def test_boot_line_reports_seq_start_with_matching_args():
    src = INO.read_text(encoding="utf-8")
    call = re.search(r'Serial\.printf\("BOOT(.*?)\\n",(.*?)\);', src, re.S)
    assert call is not None
    fmt, args = call.group(1), call.group(2)
    assert "seq_start=%lu" in fmt
    assert len(re.findall(r"%[0-9]*\w+", fmt.replace("%%", ""))) == len(
        [a for a in re.split(r",(?![^()]*\))", args) if a.strip()]
    )
