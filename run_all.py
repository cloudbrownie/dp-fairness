#!/usr/bin/env python3
# Orchestrator: run all v2 analysis drivers in dependency order.
# No flags. Each driver uses its own argparse defaults (data/results in,
# data/analysis/<artifact>/ out).

from __future__ import annotations

import logging
import sys
import traceback

from analysis import (
    make_failures,
    make_pareto,
    make_plateau,
    make_ratios,
    make_raw_summary,
    make_risk,
    make_side_effects,
    make_variant_tests,
    make_wilcoxon,
    plot_eps_sweep,
    plot_forest,
    plot_headline_bar,
    plot_headline_delta,
    plot_heatmap,
    plot_scatter,
)


PIPELINE = [
    ("make_failures", make_failures),
    ("make_raw_summary", make_raw_summary),
    ("make_ratios", make_ratios),
    ("make_risk", make_risk),
    ("make_wilcoxon", make_wilcoxon),
    ("make_side_effects", make_side_effects),
    ("make_variant_tests", make_variant_tests),
    ("make_plateau", make_plateau),
    ("make_pareto", make_pareto),
    ("plot_headline_bar", plot_headline_bar),
    ("plot_headline_delta", plot_headline_delta),
    ("plot_eps_sweep", plot_eps_sweep),
    ("plot_forest", plot_forest),
    ("plot_heatmap", plot_heatmap),
    ("plot_scatter", plot_scatter),
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    failed: list[tuple[str, str]] = []
    for name, mod in PIPELINE:
        logging.info("=== %s ===", name)
        try:
            old_argv = sys.argv
            sys.argv = [name]
            try:
                mod.main()
            finally:
                sys.argv = old_argv
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            if code != 0:
                failed.append((name, f"SystemExit({code})"))
        except Exception as e:
            tb = traceback.format_exc()
            logging.error("FAILED %s: %s\n%s", name, e, tb)
            failed.append((name, str(e)))

    if failed:
        print("\n=== FAILED STEPS ===")
        for name, msg in failed:
            print(f"  {name}: {msg}")
        sys.exit(1)
    print("\n=== all analysis steps completed ===")


if __name__ == "__main__":
    main()
