from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_all_dashboard_pages_render():
    app_path = Path(__file__).parents[1] / "app/streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    for page in ["Product Matcher", "Batch Matching", "Human Review", "Model Evaluation", "Technical Architecture"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
