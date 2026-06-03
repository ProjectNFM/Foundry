"""Generate figures for the AJILE12 tokenizer ablation experiments.

Groups: AJILE12_TOKENIZER_ABLATION, AJILE12_TOKENIZER_ABLATION_POSE

Outputs:
    ajile12_tokenizer_ablation_bar.pdf       — behavior AUROC bar chart
    ajile12_tokenizer_ablation_pose_bar.pdf   — pose R² bar chart
    summary_table.csv                         — cross-group summary
    ajile12_tokenizer_ablation_*_curves.pdf   — training curves (optional)

Usage:
    uv run scripts/analysis/tokenizer_ablation.py
    uv run scripts/analysis/tokenizer_ablation.py --plot-only
    uv run scripts/analysis/tokenizer_ablation.py --no-curves
"""

from __future__ import annotations

from common import (
    EXPERIMENT_GROUPS,
    generate_summary_table,
    load_or_fetch_group,
    make_base_parser,
    plot_group_bar_chart,
    plot_training_curves,
    setup_dirs_and_api,
)

GROUPS = ["AJILE12_TOKENIZER_ABLATION", "AJILE12_TOKENIZER_ABLATION_POSE"]


def main():
    parser = make_base_parser("AJILE12 tokenizer ablation figures")
    args = parser.parse_args()
    api = setup_dirs_and_api(args)

    dfs = {}
    for name in GROUPS:
        group = EXPERIMENT_GROUPS[name]
        df = load_or_fetch_group(
            group, api, args.cache_dir, plot_only=args.plot_only
        )
        if df is None:
            continue
        dfs[name] = df
        plot_group_bar_chart(df, group, args.output_dir)
        if not args.no_curves and api is not None:
            plot_training_curves(api, df, group, args.output_dir)

    if dfs:
        generate_summary_table(dfs, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
