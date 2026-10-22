"""Summarize the pipeline's ten evaluation seeds."""
import argparse
import csv
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--episodes-per-seed", type=int, default=50)
    args = parser.parse_args()
    with open(args.csv_path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 10 or len({row["seed"] for row in rows}) != 10:
        raise ValueError("Expected ten distinct evaluation seeds.")
    rates = [float(row["success_rate_percent"]) for row in rows]
    times = [float(row["evaluation_time_seconds"]) for row in rows]
    print(f"seeds={len(rows)} episodes_per_seed={args.episodes_per_seed} total_episodes={len(rows) * args.episodes_per_seed}")
    print(f"success_rate_mean={statistics.mean(rates):.2f}%")
    print(f"success_rate_sample_std={statistics.stdev(rates):.2f}%")
    print(f"reported={statistics.mean(rates):.2f} +/- {statistics.stdev(rates):.2f}%")
    print(f"evaluation_time_mean_seconds={statistics.mean(times):.3f}")
    print(f"evaluation_time_total_seconds={sum(times):.3f}")


if __name__ == "__main__":
    main()
