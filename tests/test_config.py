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
    assert cfg["eval"]["judge_model"] == "gemini-3.7-flash"
    assert cfg["eval"]["judge_project"] is None
    assert cfg["eval"]["judge_location"] == "global"
    assert cfg["eval"]["mmlu_n"] == 50


def test_default_config_uses_2048_mmlu_tokens() -> None:
    cfg = load_config(Path("configs/default.yaml"))
    assert cfg["steering"]["max_new_tokens_mmlu"] == 2048


def test_default_config_contains_subsim_settings() -> None:
    cfg = load_config(Path("configs/default.yaml"))
    assert cfg["subsim"]["tau"] == 0.95
    assert len(cfg["subsim"]["ks"]) == 10


def test_default_config_contains_sweep_settings() -> None:
    cfg = load_config(Path("configs/default.yaml"))
    sweep = cfg["sweep"]
    assert sweep["layers"] == [14, 17, 20, 23]
    assert sweep["alphas"] == [10, 20, 30, 40]
    assert sweep["fractions"] == [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def test_default_config_contains_generation_batch_size() -> None:
    cfg = load_config(Path("configs/default.yaml"))
    assert cfg["sweep"]["batch_size"] == 8
    assert isinstance(cfg["sweep"]["batch_size"], int)
