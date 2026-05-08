"""Generate figures for the AJILE_NEW_ALLTOKS extended tokenizer sweep.

Groups: AJILE_NEW_ALLTOKS

Outputs:
    ajile_new_alltoks_bar.pdf           — behavior AUROC bar chart
    ajile_new_alltoks_curves.pdf        — training curves (optional)

Usage:
    uv run scripts/analysis/alltoks_sweep.py
    uv run scripts/analysis/alltoks_sweep.py --plot-only
    uv run scripts/analysis/alltoks_sweep.py --no-curves
"""

from __future__ import annotations

from common import (
    EXPERIMENT_GROUPS,
    load_or_fetch_group,
    make_base_parser,
    plot_group_bar_chart,
    plot_training_curves,
    setup_dirs_and_api,
)

GROUP = "AJILE_NEW_ALLTOKS"


def main():
    parser = make_base_parser("AJILE_NEW_ALLTOKS sweep figures")
    args = parser.parse_args()
    api = setup_dirs_and_api(args)

    group = EXPERIMENT_GROUPS[GROUP]
    df = load_or_fetch_group(
        group, api, args.cache_dir, plot_only=args.plot_only
    )
    if df is None:
        return

    plot_group_bar_chart(df, group, args.output_dir)
    if not args.no_curves and api is not None:
        plot_training_curves(api, df, group, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
