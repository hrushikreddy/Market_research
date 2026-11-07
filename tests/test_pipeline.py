from pathlib import Path

from market_delay.config import load_settings
from market_delay.pipeline import run_demo


def test_demo_pipeline_writes_expected_outputs(tmp_path: Path) -> None:
    settings = load_settings("config/demo.yaml")
    settings.project.output_dir = tmp_path
    settings.analysis.bootstrap_samples = 100
    settings.validation.initial_train_size = 250
    result = run_demo(settings)
    assert (result / "feature_table.csv").exists()
    assert (result / "inference.csv").exists()
    assert (result / "forecast_metrics.csv").exists()
    assert (result / "run_manifest.json").exists()
