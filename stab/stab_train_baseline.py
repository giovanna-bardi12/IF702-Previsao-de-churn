#Baseline fixo do STab. Na raiz: python stab/stab_train_baseline.py

import json

from stab_train import MODEL_DIR, VARIATION, train_run

BASELINE_PARAMS = {
    "lr": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 64,
    "d_model": 16,
    "n_heads": 4,
    "n_layers": 1,
    "dropout": 0.1,
}


def main():
    result = train_run(BASELINE_PARAMS, "baseline")
    path = MODEL_DIR / "baseline" / "config.json"
    path.write_text(json.dumps({"variation": VARIATION, **result}, indent=2), encoding="utf-8")
    print(f"STab baseline: melhor KS={result['val_best_ks_model']['ks']:.4f}")
    print(f"Checkpoints e configuração salvos em {path.parent}")


if __name__ == "__main__":
    main()
