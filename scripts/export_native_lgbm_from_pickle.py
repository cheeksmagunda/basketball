#!/usr/bin/env python3
"""Write native LightGBM .txt + sidecar .json from existing training pickles.

Run once after pulling broken pickles, or when migrating a dev machine.
Requires the same environment as production inference: ``pip install -r requirements.txt``.

Usage (repo root)::

    python scripts/export_native_lgbm_from_pickle.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError:
        print("Install deps first: pip install -r requirements.txt", file=sys.stderr)
        return 1

    rs_pkl = REPO / "lgbm_model.pkl"
    if not rs_pkl.is_file():
        print(f"Missing {rs_pkl}", file=sys.stderr)
        return 1
    with rs_pkl.open("rb") as f:
        rs = pickle.load(f)
    if rs.get("bundle_version") != 2 or "model_baseline" not in rs or "model_spike" not in rs:
        print("lgbm_model.pkl: expected bundle_version 2 with model_baseline and model_spike", file=sys.stderr)
        return 1
    base_txt = REPO / "lgbm_baseline.txt"
    spike_txt = REPO / "lgbm_spike.txt"
    rs["model_baseline"].booster_.save_model(str(base_txt))
    rs["model_spike"].booster_.save_model(str(spike_txt))
    (REPO / "lgbm_model.json").write_text(
        json.dumps(
            {
                "format": "lightgbm_native",
                "bundle_version": 2,
                "features": rs["features"],
                "baseline_file": base_txt.name,
                "spike_file": spike_txt.name,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote lgbm_model.json, {base_txt.name}, {spike_txt.name}")

    for pkl_name, txt_name, json_name in [
        ("boost_model.pkl", "boost_model.txt", "boost_model.json"),
        ("drafts_model.pkl", "drafts_model.txt", "drafts_model.json"),
    ]:
        pkl = REPO / pkl_name
        if not pkl.is_file():
            print(f"Skip {pkl_name}: not found")
            continue
        with pkl.open("rb") as f:
            b = pickle.load(f)
        if "model" not in b or "features" not in b:
            print(f"Skip {pkl_name}: invalid bundle", file=sys.stderr)
            continue
        txt = REPO / txt_name
        b["model"].booster_.save_model(str(txt))
        (REPO / json_name).write_text(
            json.dumps(
                {"format": "lightgbm_native", "features": b["features"], "model_file": txt.name},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {json_name}, {txt_name}")

    arr = np.zeros((1, len(rs["features"])))
    mb = lgb.Booster(model_file=str(base_txt.resolve()))
    ms = lgb.Booster(model_file=str(spike_txt.resolve()))
    pred = float(mb.predict(arr)[0] + max(0.0, float(ms.predict(arr)[0])))
    print(f"RS booster smoke predict: {pred:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
