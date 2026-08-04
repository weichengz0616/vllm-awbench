# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KVRequestPolicyMetadata:
    workflow_id: str | None = None
    program_id: str | None = None
    agent_id: str | None = None
    op_id: str | None = None
    agent_steps_to_execution: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_extra_args(
        cls, extra_args: dict[str, Any] | None
    ) -> KVRequestPolicyMetadata:
        if not extra_args:
            return cls()
        raw = extra_args.get("awbench_meta")
        if not isinstance(raw, dict):
            return cls()

        steps_by_agent = (
            raw.get("agent_steps_to_execution")
            or raw.get("agent_next_call_distance")
            or {}
        )
        if not isinstance(steps_by_agent, dict):
            steps_by_agent = {}

        return cls(
            workflow_id=_as_str(raw.get("template_id"))
            or _as_str(raw.get("workflow_id")),
            program_id=_as_str(raw.get("program_id")),
            agent_id=_as_str(raw.get("agent_id")),
            op_id=_as_str(raw.get("op_id")),
            agent_steps_to_execution={
                str(k): float(v)
                for k, v in steps_by_agent.items()
                if _is_number(v)
            },
        )

    def step_for_agent(self, agent_id: str | None) -> float | None:
        if agent_id is None:
            return None
        return self.agent_steps_to_execution.get(agent_id)

    @property
    def workflow_key(self) -> str | None:
        return self.workflow_id


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
