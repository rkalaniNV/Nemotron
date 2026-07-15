"""Persona formatting (Nemotron-Personas). Reused from the reference.

Domain-agnostic: point the outer config at any locale (en_IN for the legal use
case). The generator passes whatever persona record the seed provides.
"""

from __future__ import annotations

import random
from datetime import date, datetime
from typing import Any, Dict, List


def format_persona_for_prompt(persona: Dict[str, Any]) -> str:
    def f(key: str) -> str:
        v = persona.get(key)
        return str(v) if v is not None else ""

    state = f("state") or f("region")
    age = f("age")
    if not age and f("birth_date"):
        try:
            bd = datetime.fromisoformat(str(persona["birth_date"])).date()
            age = str((date.today() - bd).days // 365)
        except Exception:
            age = ""

    name = " ".join(p for p in [f("first_name"), f("last_name")] if p) or "Unknown"
    demo: List[str] = []
    if age:
        demo.append(f"Age: {age}")
    for key, label in [("sex", "Sex"), ("marital_status", "Marital Status"),
                       ("occupation", "Occupation"), ("education_level", "Education Level")]:
        if f(key):
            demo.append(f"{label}: {f(key)}")
    loc = ", ".join(x for x in [f("city"), state, f("country")] if x)
    if loc:
        demo.append(f"Location: {loc}")

    sections = [f"Name: {name}"]
    if demo:
        sections.append("Demographics:\n" + "\n".join(demo))

    facets = [
        ("persona", "Overview"), ("professional_persona", "Professional"),
        ("finance_persona", "Finance"), ("healthcare_persona", "Healthcare"),
        ("sports_persona", "Sports"), ("arts_persona", "Arts"),
        ("travel_persona", "Travel"), ("culinary_persona", "Culinary"),
        ("skills_and_expertise", "Skills & Expertise"),
        ("hobbies_and_interests", "Hobbies & Interests"),
        ("career_goals_and_ambitions", "Career Goals"),
    ]
    random.shuffle(facets)
    for key, label in facets:
        if f(key):
            sections.append(f"{label}: {f(key).rstrip('.') + '.'}")
    return "\n\n".join(sections)
