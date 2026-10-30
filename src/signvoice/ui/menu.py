from __future__ import annotations

import streamlit as st

from signvoice.ui.compat import stretch
from signvoice.ui.registry import Section

ACTIVE_KEY = "active_section"


def _select(key: str) -> None:
    st.session_state[ACTIVE_KEY] = key


def render_menu(sections: list[Section]) -> str | None:
    """Draw the menu and return the key of the active section (None if nothing is usable)."""
    enabled_keys = [s.key for s in sections if s.enabled]
    active = st.session_state.get(ACTIVE_KEY)
    if active not in enabled_keys:
        active = enabled_keys[0] if enabled_keys else None
        st.session_state[ACTIVE_KEY] = active

    _, centre, _ = st.columns([1, 6, 1])
    with centre:
        columns = st.columns(len(sections)) if sections else []
        for column, section in zip(columns, sections):
            with column:
                st.button(
                    f"{section.icon}  {section.label}",
                    key=f"menu_{section.key}",
                    disabled=not section.enabled,
                    type="primary" if section.key == active else "secondary",
                    help=section.blurb if section.enabled else f"{section.blurb} (not built yet)",
                    on_click=_select,
                    args=(section.key,),
                    **stretch(st.button),
                )
                if not section.enabled:
                    st.markdown(
                        "<div style='text-align:center;color:gray;font-size:0.8rem'>Coming soon</div>",
                        unsafe_allow_html=True,
                    )
    return active