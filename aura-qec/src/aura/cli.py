"""
``aura`` command-line interface.

  aura info                                   codes, device profiles, versions
  aura compile  --code surface:5 --rounds 5   compile and inspect a QEC experiment
  aura calibrate [--config lab.toml] [--mode full|maintain|oracle]
  aura run       [--config lab.toml] --rounds 3 --shots 500 --decoder matching --mode stream
  aura console   [--config lab.toml] [--port 8765] [--token SECRET]
  aura link-loopback --shots 200              AURA-Link over real UDP sockets
  aura selftest                               fast end-to-end sanity check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _lab(args):
    from .lab import LabConfig, PatchConfig, VirtualLab
    if getattr(args, "config", None):
        return VirtualLab.from_file(args.config)
    patches = [PatchConfig(i, c) for i, c in enumerate(args.code or ["surface:3"])]
    return VirtualLab(LabConfig(seed=args.seed, profile=args.profile, patches=patches,
                                drift_speedup=args.drift_speedup))


def _add_lab_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="lab configuration file (.toml / .json)")
    p.add_argument("--code", action="append", help="code patch, e.g. surface:5, rep:7, bb:72 (repeatable)")
    p.add_argument("--profile", default="sota", help="device noise profile (sota, nisq, ideal)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--drift-speedup", type=float, default=1.0)


def cmd_info(args) -> int:
    from . import __version__
    from .codes import get_code
    from .link.crc import BACKEND
    from .qpu.device import PROFILES
    print(f"AURA-QEC {__version__}  (AURA-Link v1.1, CRC-32C backend: {BACKEND})")
    print("device profiles:", ", ".join(sorted(PROFILES)))
    print("codes:")
    for spec in ["rep:5", "surface:3", "surface:5", "surface:7"] + [f"bb:{k}" for k in ("72", "90", "108", "144", "288")]:
        c = get_code(spec)
        print(f"  {spec:10s} {c.label:14s} {c.name}  X-checks {c.num_x_checks}  Z-checks {c.num_z_checks}")
    return 0


def cmd_compile(args) -> int:
    from .codes import get_code
    from .host.calibration import CalibrationStore
    from .host.compiler import QecCompiler, memory_experiment, to_stim_text, uniform_snapshot
    from .qpu import build_device
    code = get_code(args.code)
    dev = build_device([(0, code)], args.profile, seed=args.seed)
    comp = QecCompiler(dev.spec, CalibrationStore(dev.spec))
    exp = memory_experiment(code, args.rounds, args.basis, allow_uncalibrated=True, schedule=args.schedule,
                            dynamical_decoupling=not args.no_dd)
    snap = uniform_snapshot(dev.spec, args.p)
    ce = comp.compile(exp, snapshot=snap)
    pm = ce.manifest.patches[0]
    print(f"{exp.name}: {dev.spec.num_qubits} qubits, {len(dev.spec.couplers)} couplers")
    print(f"  schedule={ce.stats['schedule']}  ops={ce.stats['ops']}  DD pulses={ce.stats['dd_pulses']}")
    print(f"  shot duration {ce.stats['shot_duration_ns']:.0f} ns, {pm.num_detectors} detectors, "
          f"{pm.dem.num_mechanisms} DEM mechanisms (uniform p={args.p})")
    print(f"  config epoch 0x{ce.config_epoch:08x}; compile {ce.stats['compile_s']} s (DEM {ce.stats['dem_s']} s)")
    if args.listing:
        for ch in ce.sequencers.channels[: args.listing]:
            print(f"--- channel {ch.channel} ({ch.kind} {ch.target})")
            print(ch.listing())
    if args.stim:
        Path(args.stim).write_text(to_stim_text(ce.qpu_program, pm.detectors, pm.observables, snap,
                                                det_coords=pm.det_coords))
        print(f"  wrote Stim circuit to {args.stim}")
    if args.manifest:
        Path(args.manifest).write_bytes(ce.manifest.to_blob())
        print(f"  wrote decoder manifest ({len(ce.manifest.to_blob())} bytes, zlib JSON) to {args.manifest}")
    return 0


def cmd_calibrate(args) -> int:
    lab = _lab(args)
    rep = lab.calibrate(args.mode)
    print(json.dumps(rep, indent=2, default=str))
    if args.save:
        lab.store.save(args.save, lab.sim.now)
        print(f"calibration saved to {args.save}")
    return 0


def cmd_run(args) -> int:
    lab = _lab(args)
    if args.load_calibration:
        lab.store.load(args.load_calibration)
    else:
        print("calibrating:", lab.calibrate(args.calibration)["actions"])
    exp = lab.experiment(rounds=args.rounds, basis=args.basis, branching=args.branching)
    res = lab.run(exp, shots=args.shots, decoder=args.decoder, mode=args.mode)
    print(res.summary())
    if res.branching:
        print("feed-forward:", json.dumps(res.branching, default=str))
    if args.out:
        Path(args.out).write_text(res.to_json())
        print(f"results written to {args.out}")
    return 0


def cmd_console(args) -> int:
    from .host.console import ConsoleServer
    lab = _lab(args)
    if args.calibration != "none":
        print("calibrating:", lab.calibrate(args.calibration)["actions"])
    srv = ConsoleServer(lab, args.host, args.port, args.token)
    host, port = srv.address[:2]
    print(f"AURA ops console on http://{host}:{port}/  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.stop()
    return 0


def cmd_link_loopback(args) -> int:
    from .codes import get_code
    from .host.calibration import CalibrationStore
    from .host.compiler import QecCompiler, memory_experiment
    from .qpu import build_device
    from .testbed.udp_loopback import run_udp_loopback
    code = get_code(args.code or "surface:3")
    dev = build_device([(0, code)], "sota", seed=1)
    ce = QecCompiler(dev.spec, CalibrationStore(dev.spec)).compile(
        memory_experiment(code, args.rounds, allow_uncalibrated=True, build_dem=False))
    rep = run_udp_loopback(ce.manifest, shots=args.shots, host=args.host)
    print(rep.summary())
    return 0 if rep.ok else 1


def cmd_selftest(args) -> int:
    from .lab import LabConfig, PatchConfig, VirtualLab
    t0 = time.perf_counter()
    lab = VirtualLab(LabConfig(seed=7, patches=[PatchConfig(0, "surface:3")]))
    lab.calibrate("oracle")
    r1 = lab.run(lab.experiment(rounds=3), shots=40, decoder="oracle", mode="stream")
    ok = r1.patches[0].logical_errors == 0 and r1.patches[0].missing_corrections == 0
    r2 = lab.run(lab.experiment(rounds=3), shots=2000, decoder="matching", mode="batch")
    ok &= r2.patches[0].ler < r2.patches[0].raw_logical_errors / r2.shots
    print(r1.summary())
    print(r2.summary())
    print(f"selftest {'PASS' if ok else 'FAIL'} in {time.perf_counter() - t0:.1f} s")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aura", description="AURA-QEC Layer 0 + AURA-Link toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="list codes, profiles and versions").set_defaults(fn=cmd_info)

    p = sub.add_parser("compile", help="compile a memory experiment and inspect it")
    p.add_argument("--code", default="surface:3")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--basis", default="Z", choices=["Z", "X"])
    p.add_argument("--schedule", default="auto", choices=["auto", "interleaved", "separated"])
    p.add_argument("--no-dd", action="store_true", help="disable dynamical decoupling")
    p.add_argument("--p", type=float, default=1e-3, help="uniform noise level for the DEM")
    p.add_argument("--profile", default="sota")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--listing", type=int, default=0, help="print N sequencer channel listings")
    p.add_argument("--stim", help="export the noisy circuit as a Stim file")
    p.add_argument("--manifest", help="write the decoder manifest blob")
    p.set_defaults(fn=cmd_compile)

    p = sub.add_parser("calibrate", help="run the calibration graph")
    _add_lab_args(p)
    p.add_argument("--mode", default="full", choices=["full", "maintain", "oracle"])
    p.add_argument("--save", help="write the calibration snapshot (JSON)")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("run", help="calibrate and run a QEC memory experiment")
    _add_lab_args(p)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--basis", default="Z", choices=["Z", "X"])
    p.add_argument("--shots", type=int, default=200)
    p.add_argument("--decoder", default="matching", choices=["matching", "oracle", "null"])
    p.add_argument("--mode", default="stream", choices=["stream", "batch"])
    p.add_argument("--branching", action="store_true", help="append the decoder-conditioned feed-forward test")
    p.add_argument("--calibration", default="full", choices=["full", "oracle"])
    p.add_argument("--load-calibration", help="use a saved calibration snapshot instead of calibrating")
    p.add_argument("--out", help="write the full result JSON")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("console", help="serve the operations console")
    _add_lab_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--token", help="require X-Aura-Token on every request")
    p.add_argument("--calibration", default="none", choices=["none", "full", "oracle"])
    p.set_defaults(fn=cmd_console)

    p = sub.add_parser("link-loopback", help="AURA-Link over real UDP sockets (soak test)")
    p.add_argument("--code", default="surface:3")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--shots", type=int, default=200)
    p.add_argument("--host", default="127.0.0.1")
    p.set_defaults(fn=cmd_link_loopback)

    sub.add_parser("selftest", help="fast end-to-end sanity check").set_defaults(fn=cmd_selftest)
    args = ap.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
