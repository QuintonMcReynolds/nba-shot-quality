from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("streamlit")
if not (ROOT / "artifacts" / "metrics.json").exists():
    pytest.skip("run scripts/run_pipeline.py first", allow_module_level=True)

from streamlit.testing.v1 import AppTest  # noqa: E402


def test_app_renders_and_switches_team():
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180).run()
    assert not at.exception
    assert at.metric[0].label == "Offense xPPS"
    at.selectbox(key="team").select("DEN").run()
    assert at.metric[0].value != ""
    assert not at.exception
