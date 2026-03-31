#!/usr/bin/env python3
"""Run KDA/GDN pytest suites with FLA cache enabled and verify Triton autotune cache output."""

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

DEFAULT_TESTS = [
    "tests/ops/test_kda.py",
    "tests/ops/test_gated_delta.py",
]


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_cache_dir() -> Path:
    return get_repo_root() / ".cache" / "pytest_kda_gdn_autotune"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pytest KDA/GDN suites with FLA_DISABLE_CACHE=0 and check whether "
            "Triton emits *.autotune.json files."
        )
    )
    parser.add_argument(
        "--cache-dir",
        default=str(default_cache_dir()),
        help="Dedicated Triton cache directory for this run.",
    )
    parser.add_argument(
        "--test",
        action="append",
        dest="tests",
        help="Additional pytest target. If omitted, uses the default KDA and GDN test files.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra pytest arguments. Prefix them with '--', e.g. -- -k chunk -x",
    )
    return parser.parse_args()


def build_env(repo_root: Path, cache_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["FLA_DISABLE_CACHE"] = "0"
    env["TRITON_CACHE_DIR"] = str(cache_dir)
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{repo_root}:{pythonpath}" if pythonpath else str(repo_root)
    return env


def normalize_pytest_args(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def list_autotune_files(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.rglob("*.autotune.json"))


def kernel_name_from_autotune_file(path: Path) -> str:
    suffix = ".autotune.json"
    name = path.name
    if name.endswith(suffix):
        return name[:-len(suffix)]
    return name


def run_preflight_extract_style_chunk_kda(repo_root: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    code = textwrap.dedent(
        """
        import os

        import torch

        from fla.ops.kda import chunk_kda

        print(f"preflight FLA_DISABLE_CACHE={os.environ.get('FLA_DISABLE_CACHE')}")
        print(f"preflight TRITON_CACHE_DIR={os.environ.get('TRITON_CACHE_DIR')}")

        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available for extract-style chunk_kda preflight")

        torch.manual_seed(0)
        device = "cuda"
        dtype = torch.bfloat16
        B, T, H, D = 1, 8192, 32, 128

        q = torch.rand(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        k = torch.rand(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        v = torch.rand(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        g = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        A_log = torch.randn(H, dtype=torch.float32, device=device).requires_grad_(True)
        dt_bias = torch.randn(H * D, dtype=torch.float32, device=device).requires_grad_(True)
        beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid().requires_grad_(True)
        h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=device).requires_grad_(True)
        do = torch.randn_like(v)
        dht = torch.randn_like(h0)

        tri, tri_ht = chunk_kda(
            q=q.clone(),
            k=k.clone(),
            v=v.clone(),
            g=g.clone().float(),
            beta=beta.clone(),
            A_log=A_log.clone(),
            dt_bias=dt_bias.clone(),
            scale=None,
            initial_state=h0.clone(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False,
            safe_gate=True,
            lower_bound=-5,
        )
        ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)

        tri0, tri_ht0 = chunk_kda(
            q=q.clone(),
            k=k.clone(),
            v=v.clone(),
            g=g.clone(),
            beta=beta.clone(),
            A_log=A_log.clone(),
            dt_bias=dt_bias.clone(),
            scale=None,
            initial_state=h0.clone(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-5,
        )
        ((tri0 * do).sum() + (tri_ht0 * dht).sum()).backward()

        print(f"preflight output shape={tuple(tri.shape)}")
        print(f"preflight final_state shape={tuple(tri_ht.shape)}")
        print(f"preflight second output shape={tuple(tri0.shape)}")
        print(f"preflight second final_state shape={tuple(tri_ht0.shape)}")
        print(f"preflight dq norm={q.grad.float().norm().item():.6f}")
        print(f"preflight dk norm={k.grad.float().norm().item():.6f}")
        print(f"preflight dv norm={v.grad.float().norm().item():.6f}")
        """
    ).strip()
    cmd = [sys.executable, "-c", code]
    return subprocess.run(cmd, cwd=repo_root, env=env, check=False)


def main() -> int:
    args = parse_args()
    repo_root = get_repo_root()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    tests = args.tests or DEFAULT_TESTS
    pytest_args = normalize_pytest_args(args.pytest_args)

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(repo_root, cache_dir)
    preflight_before = list_autotune_files(cache_dir)

    print("Running extract_triton_autotune_cache-style chunk_kda preflight")
    preflight_result = run_preflight_extract_style_chunk_kda(repo_root, env)
    preflight_after = list_autotune_files(cache_dir)
    preflight_new = [path for path in preflight_after if path not in preflight_before]

    print(f"preflight exit code: {preflight_result.returncode}")
    print(f"preflight autotune json delta: {len(preflight_after) - len(preflight_before)}")
    if preflight_new:
        print("preflight new autotune files:")
        for path in preflight_new:
            print(f"  {path.relative_to(cache_dir)}")
        print("preflight generated autotune cache files, exiting before pytest")
        return 0
    else:
        print("preflight new autotune files: none")
    if preflight_result.returncode != 0:
        return preflight_result.returncode

    cmd = [sys.executable, "-m", "pytest", *tests, *pytest_args]

    print(f"Repo root: {repo_root}")
    print(f"Cache dir: {cache_dir}")
    print(f"FLA_DISABLE_CACHE={env['FLA_DISABLE_CACHE']}")
    print(f"TRITON_CACHE_DIR={env['TRITON_CACHE_DIR']}")
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=repo_root, env=env, check=False)

    autotune_files = list_autotune_files(cache_dir)
    unique_kernels = sorted({kernel_name_from_autotune_file(path) for path in autotune_files})

    print(f"\npytest exit code: {result.returncode}")
    print(f"autotune json files: {len(autotune_files)}")
    print(f"unique kernels: {len(unique_kernels)}")

    if autotune_files:
        print("\nFound autotune files:")
        for path in autotune_files[:50]:
            print(f"  {path.relative_to(cache_dir)}")
        if len(autotune_files) > 50:
            print(f"  ... ({len(autotune_files) - 50} more)")
    else:
        print("\nNo *.autotune.json files were generated.")

    if result.returncode != 0:
        return result.returncode
    if not autotune_files:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
