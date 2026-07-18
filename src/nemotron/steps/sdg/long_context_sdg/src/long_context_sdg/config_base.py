"""Shared strict configuration base used by independent pipeline stages."""

from pydantic import BaseModel, ConfigDict


class StrictConfigModel(BaseModel):
    """Reject misspelled or obsolete configuration keys."""

    model_config = ConfigDict(extra="forbid")
