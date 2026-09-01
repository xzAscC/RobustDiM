from pathlib import Path

from robustdim.config import load_config


def test_default_config_matches_slides() -> None:
    cfg = load_config(Path("configs/default.yaml"))
    assert cfg["model"]["id"] == "Qwen/Qwen3-4B"
    assert cfg["model"]["layer"] == 20
    assert cfg["model"]["hidden_size"] == 2560
    assert cfg["data"]["dim_n"] == 100
    assert cfg["data"]["cov_n"] == 3000
    assert cfg["methods"]["lowvar_frac"] == 0.01
    assert cfg["eval"]["harmbench_config"] == "contextual"
    assert cfg["eval"]["harmbench_n"] == 100
    assert cfg["eval"]["judge_model"] == "DeepSeek-V3.2"
