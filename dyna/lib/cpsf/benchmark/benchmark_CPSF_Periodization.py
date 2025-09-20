# Run as (examples):
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_Periodization --mode all --N 2 --W 2 --device auto --iters 50 --warmup 10
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_Periodization --mode window --N 3 --W 2 --device cuda --iters 100 --warmup 20 --cache off
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_Periodization --mode iter_packed --N 2 --start 0 --max 3 --pack_target 20000 --device cpu --iters 50
#
# For full stats:
# > python -m dyna.lib.cpsf.benchmark.benchmark_CPSF_Periodization --mode all --N 2 --W 2 --device cuda --iters 50 --warmup 10 --profile --verify_devices

import argparse, time, torch, psutil, os
from torch.profiler import profile, ProfilerActivity, schedule

from ..periodization import CPSFPeriodization


def _fmt_bytes(x: int) -> str:
    u = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    v = float(x)
    while v >= 1024 and i < len(u) - 1:
        v /= 1024
        i += 1
    return f"{v:.2f} {u[i]}"


def _pick_device(sel: str) -> torch.device:
    if sel == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dev = torch.device(sel)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("WARN: CUDA not available, falling back to CPU.")
        return torch.device("cpu")

    return dev


def _sync(dev: torch.device):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _cuda_metrics_begin(dev: torch.device):
    if dev.type != "cuda":
        return None

    torch.cuda.reset_peak_memory_stats()
    mem0 = torch.cuda.memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    return (mem0, start, end)


def _cuda_metrics_end(dev: torch.device, token):
    if dev.type != "cuda":
        return None, None, None

    mem0, start, end = token
    end.record()
    torch.cuda.synchronize()
    dt_ms = start.elapsed_time(end)
    mem1 = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    alloc_delta = max(0, mem1 - mem0)

    return dt_ms, peak, alloc_delta


def _time_block(dev: torch.device, fn):
    if dev.type == "cuda":
        tok = _cuda_metrics_begin(dev)
        out = fn()
        dt_ms, peak, alloc = _cuda_metrics_end(dev, tok)
        return dt_ms, peak, alloc, out
    else:
        t0 = time.perf_counter()
        out = fn()
        dt_ms = (time.perf_counter() - t0) * 1e3
        return dt_ms, None, None, out


def _profile_block(dev: torch.device, steps: int, body):
    acts = [ProfilerActivity.CPU] + (
        [ProfilerActivity.CUDA] if dev.type == "cuda" else []
    )
    sch = schedule(wait=1, warmup=1, active=max(steps, 3), repeat=1)

    with profile(
        activities=acts, schedule=sch, record_shapes=False, profile_memory=True
    ) as prof:
        for _ in range(max(steps, 3)):
            body()
            prof.step()

    print(
        prof.key_averages().table(
            sort_by=(
                "self_cuda_time_total" if dev.type == "cuda" else "cpu_time_total"
            ),
            row_limit=15,
        )
    )


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=[
            "window",
            "shell",
            "iter_shells",
            "pack_offsets",
            "iter_packed",
            "all",
        ],
        default="all",
    )
    ap.add_argument(
        "--N",
        type=int,
        default=2,
    )
    ap.add_argument(
        "--W",
        type=int,
        default=2,
        help="Radius for window/shell.",
    )
    ap.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start radius for iter_*.",
    )
    ap.add_argument(
        "--max",
        dest="maxr",
        type=int,
        default=3,
        help="Max radius for iter_* and pack_offsets.",
    )
    ap.add_argument(
        "--pack_target",
        type=int,
        default=20000,
        help="Target rows per pack for iter_packed.",
    )
    ap.add_argument(
        "--idtype",
        choices=["i64", "i32", "i16", "i8"],
        default="i64",
        help="Integer dtype for offsets.",
    )
    ap.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=50,
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=10,
    )
    ap.add_argument(
        "--cache",
        choices=["on", "off"],
        default="off",
    )
    ap.add_argument(
        "--verify_devices",
        action="store_true",
    )
    ap.add_argument(
        "--profile",
        action="store_true",
    )
    args = ap.parse_args()

    dev = _pick_device(args.device)
    try:
        torch.set_default_device(dev.type)
    except Exception:
        pass

    idtype = {
        "i64": torch.int64,
        "i32": torch.int32,
        "i16": torch.int16,
        "i8": torch.int8,
    }[args.idtype]

    gen = CPSFPeriodization(
        enable_cache=(args.cache == "on"),
        dtype=idtype,
    )

    N = args.N
    W = args.W
    start = args.start
    maxr = args.maxr
    pack_target = args.pack_target

    print(
        f"Device={dev.type}"
        f", dtype={idtype}"
        f", N={N}"
        f", W={W}"
        f", start={start}"
        f", max={maxr}"
        f", pack_target={pack_target}"
        f", cache={args.cache}"
        f", warmup={args.warmup}"
        f", iters={args.iters}"
    )

    if N < 1:
        raise SystemExit("CPSFPeriodization requires N >= 1 (complex dimension).")
    if W < 0 or start < 0 or (maxr is not None and maxr < 0):
        raise SystemExit("Radii must be >= 0.")
    if args.mode in ("iter_shells", "iter_packed", "all") and (
        maxr is not None and start > maxr
    ):
        print("WARN: start > max; iter_* ranges will be empty.")

    def window_size(N, W):
        if W == 0:
            return 1
        tw = (W << 1) + 1
        return int(pow(tw, 2 * N))

    def shell_size(N, W):
        if W == 0:
            return 1
        tw = (W << 1) + 1
        tm = (W << 1) - 1
        return int(pow(tw, 2 * N) - pow(tm, 2 * N))

    def bench_window():
        M = window_size(N, W)
        _sync(dev)
        for _ in range(args.warmup):
            out = gen.window(N=N, W=W, device=dev)
            _ = int(out.shape[0])
        _sync(dev)

        if args.profile:
            _profile_block(dev, steps=5, body=lambda: gen.window(N=N, W=W, device=dev))

        times, peak_max, alloc_max = [], 0, 0
        for _ in range(args.iters):
            dt_ms, peak, alloc, out = _time_block(
                dev, lambda: gen.window(N=N, W=W, device=dev)
            )
            _ = int(out.shape[0])
            times.append(dt_ms)
            if peak:
                peak_max = max(peak_max, peak)
            if alloc:
                alloc_max = max(alloc_max, alloc)

        avg = sum(times) / len(times)
        thr = M / (avg / 1e3)

        print("\n=== CPSFPeriodization.window ===")
        print(f"M={M:,} points per call")
        print(f"Avg time/iter: {avg:.3f} ms")
        print(f"Throughput:    {thr:,.0f} pts/s")
        if dev.type == "cuda":
            print(f"CUDA peak mem (max):   {_fmt_bytes(peak_max)}")
            print(f"CUDA alloc Δ (max):    {_fmt_bytes(alloc_max)}")

    def bench_shell():
        Mw = shell_size(N, W)

        _sync(dev)

        for _ in range(args.warmup):
            out = gen.shell(N=N, W=W, device=dev)
            _ = int(out.shape[0])

        _sync(dev)

        if args.profile:
            _profile_block(dev, steps=5, body=lambda: gen.shell(N=N, W=W, device=dev))

        times, peak_max, alloc_max = [], 0, 0
        for _ in range(args.iters):
            dt_ms, peak, alloc, out = _time_block(
                dev, lambda: gen.shell(N=N, W=W, device=dev)
            )
            _ = int(out.shape[0])
            times.append(dt_ms)
            if peak:
                peak_max = max(peak_max, peak)
            if alloc:
                alloc_max = max(alloc_max, alloc)

        avg = sum(times) / len(times)
        thr = Mw / (avg / 1e3)

        print("\n=== CPSFPeriodization.shell ===")
        print(f"M_w={Mw:,} points per shell")
        print(f"Avg time/iter: {avg:.3f} ms")
        print(f"Throughput:    {thr:,.0f} pts/s")
        if dev.type == "cuda":
            print(f"CUDA peak mem (max):   {_fmt_bytes(peak_max)}")
            print(f"CUDA alloc Δ (max):    {_fmt_bytes(alloc_max)}")

    def bench_iter_shells():
        if maxr is None:
            raise SystemExit("iter_shells benchmark requires finite --max.")

        total_pts = sum(shell_size(N, w) for w in range(start, maxr + 1))

        _sync(dev)

        for _ in range(args.warmup):
            for w, S in gen.iter_shells(
                N=N, start_radius=start, max_radius=maxr, device=dev
            ):
                _ = int(S.shape[0])

        _sync(dev)

        if args.profile:

            def _drain_iter_packed():
                for _a, _b, P in gen.iter_packed(
                    N=N,
                    target_points_per_pack=pack_target,
                    start_radius=start,
                    max_radius=maxr,
                    device=dev,
                ):
                    _ = int(P.shape[0])

            _profile_block(dev, steps=3, body=_drain_iter_packed)

        times, peak_max, alloc_max = [], 0, 0
        for _ in range(args.iters):

            def body():
                tot = 0
                for w, S in gen.iter_shells(
                    N=N, start_radius=start, max_radius=maxr, device=dev
                ):
                    tot += S.shape[0]
                return tot

            dt_ms, peak, alloc, tot = _time_block(dev, body)
            assert int(tot) == total_pts
            times.append(dt_ms)
            if peak:
                peak_max = max(peak_max, peak)
            if alloc:
                alloc_max = max(alloc_max, alloc)

        avg = sum(times) / len(times)
        thr = total_pts / (avg / 1e3)

        print("\n=== CPSFPeriodization.iter_shells ===")
        print(f"Total points (start..max): {total_pts:,}")
        print(f"Avg time/pass: {avg:.3f} ms")
        print(f"Throughput:    {thr:,.0f} pts/s")
        if dev.type == "cuda":
            print(f"CUDA peak mem (max):   {_fmt_bytes(peak_max)}")
            print(f"CUDA alloc Δ (max):    {_fmt_bytes(alloc_max)}")

    def bench_pack_offsets():
        if maxr is None:
            raise SystemExit("pack_offsets benchmark requires finite --max.")

        total_pts = window_size(N, maxr)

        _sync(dev)

        for _ in range(args.warmup):
            off, lengths = gen.pack_offsets(N=N, max_radius=maxr, device=dev)
            _ = int(off.shape[0])

        _sync(dev)

        if args.profile:
            _profile_block(
                dev,
                steps=3,
                body=lambda: gen.pack_offsets(N=N, max_radius=maxr, device=dev),
            )

        times, peak_max, alloc_max = [], 0, 0
        for _ in range(args.iters):
            dt_ms, peak, alloc, res = _time_block(
                dev, lambda: gen.pack_offsets(N=N, max_radius=maxr, device=dev)
            )
            off, lengths = res
            assert int(off.shape[0]) == total_pts
            times.append(dt_ms)
            if peak:
                peak_max = max(peak_max, peak)
            if alloc:
                alloc_max = max(alloc_max, alloc)

        avg = sum(times) / len(times)
        thr = total_pts / (avg / 1e3)

        print("\n=== CPSFPeriodization.pack_offsets ===")
        print(f"Total points:  {total_pts:,}")
        print(f"Avg time/call: {avg:.3f} ms")
        print(f"Throughput:    {thr:,.0f} pts/s")
        if dev.type == "cuda":
            print(f"CUDA peak mem (max):   {_fmt_bytes(peak_max)}")
            print(f"CUDA alloc Δ (max):    {_fmt_bytes(alloc_max)}")

    def bench_iter_packed():
        if maxr is None:
            raise SystemExit("iter_packed benchmark requires finite --max.")

        total_pts = sum(shell_size(N, w) for w in range(start, maxr + 1))

        _sync(dev)

        for _ in range(args.warmup):
            for a, b, P in gen.iter_packed(
                N=N,
                target_points_per_pack=pack_target,
                start_radius=start,
                max_radius=maxr,
                device=dev,
            ):
                _ = int(P.shape[0])

        _sync(dev)

        if args.profile:

            def body():
                for a, b, P in gen.iter_packed(
                    N=N,
                    target_points_per_pack=pack_target,
                    start_radius=start,
                    max_radius=maxr,
                    device=dev,
                ):
                    _ = int(P.shape[0])

            _profile_block(dev, steps=3, body=body)

        times, peak_max, alloc_max = [], 0, 0
        for _ in range(args.iters):

            def body():
                tot = 0
                for a, b, P in gen.iter_packed(
                    N=N,
                    target_points_per_pack=pack_target,
                    start_radius=start,
                    max_radius=maxr,
                    device=dev,
                ):
                    tot += P.shape[0]
                return tot

            dt_ms, peak, alloc, tot = _time_block(dev, body)
            assert int(tot) == total_pts
            times.append(dt_ms)

            if peak:
                peak_max = max(peak_max, peak)
            if alloc:
                alloc_max = max(alloc_max, alloc)

        avg = sum(times) / len(times)
        thr = total_pts / (avg / 1e3)

        print("\n=== CPSFPeriodization.iter_packed ===")
        print(f"Total points (start..max): {total_pts:,}")
        print(f"Avg time/pass: {avg:.3f} ms")
        print(f"Throughput:    {thr:,.0f} pts/s")
        if dev.type == "cuda":
            print(f"CUDA peak mem (max):   {_fmt_bytes(peak_max)}")
            print(f"CUDA alloc Δ (max):    {_fmt_bytes(alloc_max)}")

    if args.verify_devices:
        w0 = gen.window(N=N, W=W, device=dev)
        s0 = gen.shell(N=N, W=W, device=dev)
        ok_w = w0.device.type == dev.type
        ok_s = s0.device.type == dev.type

        print(f"verify(window): in={dev.type}, out={w0.device.type} -> {ok_w}")
        print(f"verify(shell):  in={dev.type}, out={s0.device.type} -> {ok_s}")

        del w0, s0

    modes = (
        [args.mode]
        if args.mode != "all"
        else ["window", "shell", "iter_shells", "pack_offsets", "iter_packed"]
    )

    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for m in modes:
        if m == "window":
            bench_window()
        elif m == "shell":
            bench_shell()
        elif m == "iter_shells":
            bench_iter_shells()
        elif m == "pack_offsets":
            bench_pack_offsets()
        elif m == "iter_packed":
            bench_iter_packed()

    if dev.type != "cuda":
        try:
            rss = psutil.Process(os.getpid()).memory_info().rss
            print(f"\nProcess RSS:   {_fmt_bytes(rss)}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
