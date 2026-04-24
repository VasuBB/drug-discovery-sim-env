from __future__ import annotations

from pydantic import BaseModel, Field


class TargetSnapshotRow(BaseModel):
    disease: str = Field(min_length=1)
    target: str = Field(min_length=1)
    target_class: str = Field(min_length=1)
    druggability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class CompoundSnapshotRow(BaseModel):
    smiles: str = Field(min_length=1)
    qed: float = Field(ge=0.0, le=1.0)


class LiteratureSnapshotRow(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    abstract: str = Field(min_length=1)
    year: int = Field(ge=1900, le=2100)
