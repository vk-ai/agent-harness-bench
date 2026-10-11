"""Render results/*.json into the README results table (between RESULTS markers)."""
import glob, json, re

ORDER = ["parallel_tools", "sequential_tools", "structured_output", "context_full", "context_compressed", "ALL"]
lines = ["| model | category | n | success [95% CI] | latency s [CI] | tokens [CI] | est. compute $/task |",
         "|---|---|---|---|---|---|---|"]
extra = ["", "| model | emitted parallel calls | schema-valid 1st try | schema-valid after repair | prompt cache off -> on (s) | wall clock |",
         "|---|---|---|---|---|---|"]
for f in sorted(glob.glob("results/*.json")):
    s = json.load(open(f))["summary"]
    c = s["categories"]
    for k in ORDER:
        v = c[k]
        lines.append(f"| {s['model'] if k == 'parallel_tools' else ''} | {k} | {v['n']} | {v['success']:.2f} "
                     f"[{v['success_ci95'][0]:.2f}, {v['success_ci95'][1]:.2f}] | {v['latency_s']:.1f} "
                     f"[{v['latency_ci95'][0]:.1f}, {v['latency_ci95'][1]:.1f}] | {v['tokens']:.0f} "
                     f"[{v['tokens_ci95'][0]:.0f}, {v['tokens_ci95'][1]:.0f}] | {v['est_compute_usd_per_task']:.4f} |")
    pc = s.get("prompt_cache_experiment", {})
    p, j = c["parallel_tools"], c["structured_output"]
    extra.append(f"| {s['model']} | {p['emitted_parallel_calls']:.2f} {p['emitted_parallel_calls_ci95']} | "
                 f"{j['schema_valid_first_try']:.2f} | {j['schema_valid_after_repair']:.2f} | "
                 f"{pc.get('cache_off', {}).get('mean_latency_s')} -> {pc.get('cache_on', {}).get('mean_latency_s')} "
                 f"({pc.get('cache_on', {}).get('mean_cached_tokens', 0):.0f} cached tok) | {s['wall_clock_s'] / 60:.0f} min |")
table = "\n".join(lines + extra)
r = open("README.md").read()
r = re.sub(r"<!-- RESULTS -->.*<!-- /RESULTS -->", "<!-- RESULTS -->\n" + table + "\n<!-- /RESULTS -->", r, flags=re.S)
open("README.md", "w").write(r)
