
# -*- coding: utf-8 -*-
"""
Generate 13 independent confusion-matrix figures on the Hunan-Plant dataset
using a fixed shared training seed = 43 for all methods.

Formal source scripts:
    1) run_ajse_hunan_fixedsplit_5runs.py
       - 12 comparison methods
    2) run_grkanet_full_5runs_final.py
       - Ours (GR-KANet)

Protocol:
    - Fixed split seed: 42
    - Shared training seed for ALL methods: 43
    - One independent PNG per method
    - Count-based confusion matrices in the same style as the user's original figure
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


METHOD_ORDER: List[Tuple[str, str]] = [
    ("01", "SVM"),
    ("02", "XGBoost"),
    ("03", "1D-CNN"),
    ("04", "SMOTE-GBDT"),
    ("05", "SMOTE-ENN-XGBoost"),
    ("06", "AdaBoost-TCNN"),
    ("07", "Lightweight ResNet"),
    ("08", "MTF-GhostNetV2"),
    ("09", "Graph-enhanced DGA"),
    ("10", "DE-IPFL"),
    ("11", "Knowledge-filtered DSL"),
    ("12", "Knowledge-CapsNet"),
    ("13", "Ours (GR-KANet)"),
]


def resolve_existing_path(path_text: str) -> Path:
    raw = Path(path_text).expanduser()
    candidates = [
        raw,
        Path.cwd() / raw,
        Path(__file__).resolve().parent / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    attempted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Cannot find file: {path_text}\nAttempted:\n{attempted}"
    )


def import_module_from_path(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Python module from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def safe_filename(method_name: str) -> str:
    name = method_name.replace("Ours (GR-KANet)", "GR-KANet")
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
    return name.strip("_")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    return {
        "Acc (%)": 100.0 * accuracy_score(y_true, y_pred),
        "Macro-F1 (%)": 100.0 * f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "Macro-P (%)": 100.0 * precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "Macro-R (%)": 100.0 * recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    output_path: Path,
    method_name: str,
    show_title: bool,
) -> np.ndarray:
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(labels)),
    )

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    image = ax.imshow(
        cm,
        cmap="Blues",
        interpolation="nearest",
    )

    # 不要 colorbar
    # fig.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_yticklabels(labels, fontsize=9)

    # 不要坐标轴标题
    # ax.set_xlabel("Predicted label", fontsize=10)
    # ax.set_ylabel("True label", fontsize=10)

    # 一般拼图时也不需要每个子图都写标题
    if show_title:
        ax.set_title(method_name, fontsize=11, pad=6)

    vmax = float(cm.max()) if cm.size else 0.0
    threshold = vmax / 2.0

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            value = int(cm[row, col])
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > threshold else "black",
            )

    ax.set_ylim(len(labels) - 0.5, -0.5)

    # 让边缘更紧凑，方便后续拼接
    fig.tight_layout(pad=0.2)
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    return cm


def make_comparison_args(
    argparse_module,
    data_path: Path,
    output_dir: Path,
    selected_seed: int,
    cpu: bool,
    verbose: bool,
):
    return argparse_module.Namespace(
        data=str(data_path),
        label_col="故障编码",
        outdir=str(output_dir),
        cpu=cpu,
        torch_epochs=150,
        patience=80,
        lr=1e-3,
        batch_size=64,
        boost_rounds=5,
        boost_epochs=40,
        verbose=verbose,
        seed=selected_seed,
    )


def run_comparison_methods(
    comp,
    split,
    device: torch.device,
    run_args,
    selected_seed: int,
) -> Dict[str, np.ndarray]:
    predictions: Dict[str, np.ndarray] = {}
    num_classes = len(split.display_labels)

    comp.set_seed(selected_seed)
    predictions["SVM"] = comp.run_svm(split, selected_seed)

    comp.set_seed(selected_seed)
    predictions["XGBoost"] = comp.run_xgboost(split, selected_seed)

    comp.set_seed(selected_seed)
    _, predictions["1D-CNN"] = comp.train_torch_classifier(
        comp.CNN1D(num_classes),
        split,
        device,
        epochs=run_args.torch_epochs,
        patience=run_args.patience,
        lr=run_args.lr,
        batch_size=run_args.batch_size,
        weighted_sampler=True,
        class_weighted_ce=True,
        verbose=run_args.verbose,
    )

    comp.set_seed(selected_seed)
    predictions["SMOTE-GBDT"] = comp.run_smote_gbdt(split, selected_seed)

    comp.set_seed(selected_seed)
    predictions["SMOTE-ENN-XGBoost"] = comp.run_smoteenn_xgboost(split, selected_seed)

    comp.set_seed(selected_seed)
    predictions["AdaBoost-TCNN"] = comp.run_adaboost_tcnn(split, device, run_args)

    comp.set_seed(selected_seed)
    _, predictions["Lightweight ResNet"] = comp.train_torch_classifier(
        comp.LightweightResNet1D(num_classes, channels=16),
        split,
        device,
        epochs=100,
        patience=30,
        lr=5e-4,
        batch_size=32,
        weighted_sampler=False,
        class_weighted_ce=True,
        verbose=run_args.verbose,
    )

    comp.set_seed(selected_seed)
    predictions["MTF-GhostNetV2"] = comp.run_mtf_ghostnet(split, device, run_args)

    comp.set_seed(selected_seed)
    predictions["Graph-enhanced DGA"] = comp.run_graph_enhanced_dga(split, selected_seed)

    comp.set_seed(selected_seed)
    predictions["DE-IPFL"] = comp.run_de_ipfl(split, device, run_args)

    comp.set_seed(selected_seed)
    predictions["Knowledge-filtered DSL"] = comp.run_knowledge_filtered_dsl(split, selected_seed)

    comp.set_seed(selected_seed)
    _, predictions["Knowledge-CapsNet"] = comp.train_torch_classifier(
        comp.KnowledgeCapsNet(num_classes),
        split,
        device,
        epochs=run_args.torch_epochs,
        patience=run_args.patience,
        lr=run_args.lr,
        batch_size=run_args.batch_size,
        weighted_sampler=True,
        class_weighted_ce=True,
        verbose=run_args.verbose,
    )

    return predictions


def make_grkanet_args(
    argparse_module,
    selected_seed: int,
    cpu: bool,
    verbose: bool,
):
    return argparse_module.Namespace(
        split_seed=42,
        n_runs=1,
        seeds=str(selected_seed),
        epochs=150,
        patience=80,
        lr=1e-3,
        batch_size=64,
        balance_mode="both",
        dgam_mode="energy",
        alpha_init=2.0,
        dropout=0.0,
        channels=32,
        kan_hidden=8,
        kan_latent=64,
        mask_update_every=5,
        dgam_ema=0.8,
        cpu=cpu,
        verbose=verbose,
        seed=selected_seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 13 Hunan-Plant confusion matrices with a shared training seed of 43."
        )
    )
    parser.add_argument(
        "--data",
        type=str,
        default="胖虎电厂数据未加测试集.xlsx",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="故障编码",
    )
    parser.add_argument(
        "--comparison-script",
        type=str,
        default="run_ajse_hunan_fixedsplit_5runs.py",
    )
    parser.add_argument(
        "--grkanet-script",
        type=str,
        default="run_grkanet_full_5runs_final.py",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=43,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="hunan_seed43_confusion_matrices",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--show-title",
        action="store_true",
        help="Show method name above each figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = resolve_existing_path(args.data)
    comparison_path = resolve_existing_path(args.comparison_script)
    grkanet_path = resolve_existing_path(args.grkanet_script)

    output_dir = Path(args.outdir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    comp = import_module_from_path(
        "fixed43_hunan_comparison",
        comparison_path,
    )
    grkanet = import_module_from_path(
        "fixed43_hunan_grkanet",
        grkanet_path,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Dataset: {data_path}")
    print(f"Fixed split seed: {args.split_seed}")
    print(f"Shared training seed for all methods: {args.train_seed}")
    print(f"Comparison script: {comparison_path}")
    print(f"GR-KANet script: {grkanet_path}")

    comparison_split = comp.load_and_split_data(
        data_path,
        label_col=args.label_col,
        seed=args.split_seed,
    )
    grkanet_split = grkanet.load_fixed_split(
        str(data_path),
        label_col=args.label_col,
        split_seed=args.split_seed,
    )

    if not np.array_equal(comparison_split.y_train, grkanet_split.y_train):
        raise RuntimeError("Training labels differ between the two formal scripts.")
    if not np.array_equal(comparison_split.y_val, grkanet_split.y_val):
        raise RuntimeError("Validation labels differ between the two formal scripts.")
    if not np.array_equal(comparison_split.y_test, grkanet_split.y_test):
        raise RuntimeError("Test labels differ between the two formal scripts.")
    if not np.allclose(comparison_split.X_test, grkanet_split.X_test, atol=1e-7):
        raise RuntimeError("Standardized test data differ between the two formal scripts.")

    comparison_args = make_comparison_args(
        argparse,
        data_path,
        output_dir,
        args.train_seed,
        args.cpu,
        args.verbose,
    )
    predictions = run_comparison_methods(
        comp=comp,
        split=comparison_split,
        device=device,
        run_args=comparison_args,
        selected_seed=args.train_seed,
    )

    grkanet_args = make_grkanet_args(
        argparse,
        args.train_seed,
        args.cpu,
        args.verbose,
    )
    grkanet_result, best_epoch, best_val_f1 = grkanet.train_one_seed(
        split=grkanet_split,
        seed=args.train_seed,
        device=device,
        args=grkanet_args,
    )
    predictions["Ours (GR-KANet)"] = grkanet_result.y_pred

    print(
        "GR-KANet | "
        f"seed={args.train_seed} | "
        f"best_epoch={best_epoch} | "
        f"best_val_macro_f1={100.0 * best_val_f1:.2f}% | "
        f"Acc={100.0 * grkanet_result.acc:.2f} | "
        f"Macro-F1={100.0 * grkanet_result.macro_f1:.2f}"
    )

    y_true = comparison_split.y_test
    labels = list(comparison_split.display_labels)

    prediction_table = pd.DataFrame({
        "True index": y_true,
        "True label": [labels[int(value)] for value in y_true],
    })
    metric_rows = []

    for order, method in METHOD_ORDER:
        if method not in predictions:
            raise KeyError(f"Missing predictions for method: {method}")

        y_pred = np.asarray(predictions[method], dtype=int)
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"{method}: prediction shape {y_pred.shape} does not match true-label shape {y_true.shape}."
            )

        file_stem = f"{order}_{safe_filename(method)}"
        png_path = output_dir / f"{file_stem}_confusion_matrix.png"
        csv_path = output_dir / f"{file_stem}_confusion_matrix.csv"

        cm = save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            output_path=png_path,
            method_name=method,
            show_title=args.show_title,
        )

        pd.DataFrame(cm, index=labels, columns=labels).to_csv(
            csv_path,
            encoding="utf-8-sig",
        )

        metrics = compute_metrics(y_true, y_pred)
        metrics.update({
            "Order": int(order),
            "Method": method,
            "Split seed": args.split_seed,
            "Training seed": args.train_seed,
        })
        metric_rows.append(metrics)

        prediction_table[f"{method} index"] = y_pred
        prediction_table[f"{method} label"] = [labels[int(value)] for value in y_pred]

        print(
            f"{method:28s} | "
            f"seed={args.train_seed} | "
            f"Acc={metrics['Acc (%)']:.2f} | "
            f"Macro-F1={metrics['Macro-F1 (%)']:.2f} | "
            f"saved={png_path.name}"
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values("Order")
    metrics_df = metrics_df[
        [
            "Order",
            "Method",
            "Split seed",
            "Training seed",
            "Acc (%)",
            "Macro-F1 (%)",
            "Macro-P (%)",
            "Macro-R (%)",
        ]
    ]
    metrics_df.to_csv(
        output_dir / "seed43_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )
    prediction_table.to_csv(
        output_dir / "seed43_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    readme_lines = [
        "Hunan-Plant seed-43 confusion matrices",
        f"Dataset: {data_path}",
        f"Fixed split seed: {args.split_seed}",
        f"Shared training seed for all methods: {args.train_seed}",
        "",
        f"Comparison script: {comparison_path}",
        f"GR-KANet script: {grkanet_path}",
        "",
        "Figures:",
    ]
    for order, method in METHOD_ORDER:
        readme_lines.append(
            f"{order}: {method} -> "
            f"{order}_{safe_filename(method)}_confusion_matrix.png"
        )

    (output_dir / "README.txt").write_text(
        "\n".join(readme_lines),
        encoding="utf-8",
    )

    print(f"\nSaved 13 independent PNG figures to: {output_dir}")
    print(f"Metrics: {output_dir / 'seed43_metrics.csv'}")
    print(f"Predictions: {output_dir / 'seed43_test_predictions.csv'}")


if __name__ == "__main__":
    main()
