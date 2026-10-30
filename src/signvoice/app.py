#Run from the repo root:    streamlit run src/signvoice/app.py


from __future__ import annotations

import sys
from pathlib import Path

try:
    import signvoice 
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from signvoice.ui import menu, registry


def main() -> None:
    st.set_page_config(page_title="Sign-to-Voice", page_icon="🤟", layout="wide")
    st.markdown("<h1 style='text-align:center;margin-bottom:0'>Sign-to-Voice</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:gray'>Sign language query assistant: vision, LLM, voice</p>",
                unsafe_allow_html=True)

    sections = registry.get_sections()
    active = menu.render_menu(sections)

    for module, trace in registry.load_errors().items():
        with st.expander(f"Plug-in failed to load: {module}", expanded=True):
            st.code(trace)

    st.divider()
    chosen = next((s for s in sections if s.key == active), None)
    if chosen is None:
        st.warning("No sections are available yet.")
    else:
        chosen.render()


if __name__ == "__main__":
    main()