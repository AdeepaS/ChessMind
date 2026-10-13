"""Audited embedded resources, not a general policy for Stockfish builds.

Day 7 Step 4D.1/4D.2 verification of the project Windows executable:
* 103,046,300-byte MinGW64 Stockfish 19, x86-64-avx512icl.
* NNUE payload at file offset 3,189,792, length 98,511,183 bytes; its
  independently calculated SHA-256 is recorded below.
* UCI config/target_config select nn-1a298aa575a0.nnue; no EvalFileSmall.
* The read-only `eval` command successfully uses that network.
* sf_19/src/nnue/network.cpp loads the embedded default before external files:
  https://github.com/official-stockfish/Stockfish/blob/sf_19/src/nnue/network.cpp

The full executable digest binds this evidence to these exact bytes. Unknown
builds require independent verification. Files must remain unchanged throughout
the engine lifetime. Tablebase validation remains the identity builder's job.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddedResourceAttestation:
    executable_sha256: str
    option: str
    value: str
    network_sha256: str


VERIFIED_EMBEDDED_RESOURCES = (
    EmbeddedResourceAttestation(
        "45bc8e4969147db9c2eb533810637994619bff0eacc81ccfd9854394901bcbd0",
        "evalfile",
        "nn-1a298aa575a0.nnue",
        "1a298aa575a085434d29027978dc36867fe9c5bcea9376654b7a8eba1e52dfc2",
    ),
)


def verified_embedded_resource(executable_sha256: str, option: str,
                               value: str) -> bool:
    return any(record.executable_sha256 == executable_sha256
               and record.option == option and record.value == value
               for record in VERIFIED_EMBEDDED_RESOURCES)
