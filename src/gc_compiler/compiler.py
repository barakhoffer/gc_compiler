"""Top-level compiler: configuration in, GDS + netlist + metrics out."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import gdsfactory as gf

from gc_compiler.array.floorplan import gc_banked_macro, gc_macro, macro_geometry
from gc_compiler.array.tiler import array_geometry, gc_array
from gc_compiler.cells.tap_row import tap_row_geometry
from gc_compiler.config import ArrayConfig
from gc_compiler.netlist.spice import array_netlist, macro_netlist
from gc_compiler.periphery.logic import LevelShifterConfig, LogicConfig
from gc_compiler.tech.rules import Technology
from gc_compiler.verify.timing import Characterisation, Corner


@dataclass
class BuildResult:
    component: gf.Component
    gds: Path
    spice: Path
    metrics_file: Path
    metrics: dict
    decoded: bool = False
    banked: bool = False

    def __str__(self) -> str:
        m = self.metrics
        macro = "macro_width_um" in m
        width = m["macro_width_um"] if macro else m["width_um"]
        height = m["macro_height_um"] if macro else m["height_um"]
        per_bit = m["macro_area_per_bit_um2"] if macro else m["area_per_bit_um2"]
        banks = "2 banks of " if m.get("banked") else ""
        return (
            f"{m['name']}: {banks}{m['rows']}x{m['cols']} = {m['bits']} bits, "
            f"{width:.2f} x {height:.2f} um, {per_bit:.3f} um2/bit"
        )


class GainCellCompiler:
    """Builds a gain-cell memory from an :class:`ArrayConfig`."""

    def __init__(
        self,
        cfg: ArrayConfig | None = None,
        tech: Technology | None = None,
        logic: LogicConfig | None = None,
        ls: LevelShifterConfig | None = None,
    ) -> None:
        if tech is None:
            from gc_compiler.tech import default_technology

            tech = default_technology()
        self.tech = tech
        self.cfg = cfg or ArrayConfig.for_tech(tech)
        self.logic = logic or LogicConfig.for_tech(tech)
        self.ls = ls or LevelShifterConfig.for_tech(tech)


    def metrics(
        self,
        name: str | None = None,
        macro: bool = False,
        component: gf.Component | None = None,
    ) -> dict:
        """``component`` is the built macro; when it carries decoders, the macro dimensions
        come from it (``macro_geometry`` describes the array, drivers and columns only)."""
        cfg, tech = self.cfg, self.tech
        geo = array_geometry(tech, cfg)
        tap = tap_row_geometry(tech, geo.cell)
        data = {
            "name": name or (f"{cfg.name}_macro" if macro else cfg.name),
            "technology": tech.name,
            "rows": cfg.rows,
            "cols": cfg.cols,
            "bits": geo.bits,
            "transistors": 3 * cfg.rows * cfg.total_cols,
            "replica_column": cfg.replica_column,
            "negative_write_wl": cfg.negative_write_wl,
            "wordline_straps": cfg.wordline_straps,
            "cell_pitch_x_um": geo.cell.pitch_x,
            "cell_pitch_y_um": geo.cell.pitch_y,
            "cell_area_um2": geo.cell.area,
            "tap_rows": len(geo.tap_row_y),
            "tap_row_height_um": tap.height,
            "width_um": geo.width,
            "height_um": geo.height,
            "area_um2": geo.area,
            "area_per_bit_um2": geo.area / geo.bits,
            "array_efficiency": geo.cell.area * geo.bits / geo.area,
        }
        if macro:
            macro_geo = macro_geometry(tech, cfg, self.logic, self.ls)
            # per row: two inverters (4 devices), or with a negative write word line the read
            # inverter plus the write driver's inverter, level shifter and output stage (10)
            drivers = 10 if cfg.negative_write_wl else 4
            data.update(
                {
                    "transistors": 3 * cfg.rows * cfg.total_cols + drivers * cfg.rows + 5 * cfg.total_cols,
                    "macro_width_um": macro_geo.width,
                    "macro_height_um": macro_geo.height,
                    "macro_area_um2": macro_geo.area,
                    "macro_area_per_bit_um2": macro_geo.area / geo.bits,
                    "macro_efficiency": macro_geo.efficiency,
                    "driver_width_um": macro_geo.driver_width,
                    "column_height_um": macro_geo.column_height,
                    "decoded": False,
                }
            )
            if cfg.negative_write_wl:
                data["write_wl_low_v"] = cfg.write_wl_low
                data["write_oxide_stress_v"] = self.tech.electrical.vdd_nominal - cfg.write_wl_low
            if component is not None and component.info.get("decoded"):
                width, height = component.info["width"], component.info["height"]
                banks = 2 if component.info.get("banked") else 1
                bits = banks * geo.bits
                data.update(
                    {
                        "decoded": True,
                        "banked": banks == 2,
                        "bits": bits,
                        "transistors": banks * data["transistors"],  # plus the decoder, not counted
                        "macro_width_um": width,
                        "macro_height_um": height,
                        "macro_area_um2": width * height,
                        "macro_area_per_bit_um2": width * height / bits,
                        "macro_efficiency": geo.cell.area * bits / (width * height),
                    }
                )
        return data

    def build(
        self,
        outdir: str | Path = "build",
        name: str | None = None,
        macro: bool = True,
        decoded: bool | None = None,
        banked: bool | None = None,
    ) -> BuildResult:
        """By default the whole memory: a decoded, banked macro. ``macro=False`` builds the
        bitcell array alone. ``decoded`` (with ``macro``; the default) places and wires the row
        decoder, so the macro takes ``addr``/``we``/``re`` instead of raw word-line inputs.
        ``banked`` (with ``decoded``) builds two banks of the array around one central decoder
        (see :func:`~gc_compiler.array.floorplan.gc_banked_macro`) -- the default for a decoded
        macro (13 to 25% less area per bit); ``banked=False`` builds one."""
        if decoded is None:
            decoded = macro
        if banked is None:
            banked = decoded
        if decoded and not macro:
            raise ValueError("decoded=True needs macro=True")
        if banked and not decoded:
            raise ValueError("banked=True needs decoded=True (the decoder is what picks the bank)")
        if self.cfg.negative_write_wl and not macro:
            raise ValueError("negative_write_wl needs the periphery: use macro=True")
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        name = name or (f"{self.cfg.name}_macro" if macro else self.cfg.name)

        if banked:
            component = gc_banked_macro(self.tech, self.cfg, self.logic, name=name, ls=self.ls)
            netlist = macro_netlist(self.tech, self.cfg, self.logic, name=name, ls=self.ls, banked=True)
        elif macro:
            component = gc_macro(self.tech, self.cfg, self.logic, name=name, decoded=decoded, ls=self.ls)
            # the netlist must describe the same interface as the layout: raw per-row
            # wwl_in/rwl_in unless the decoders were placed too
            netlist = macro_netlist(
                self.tech, self.cfg, self.logic, name=name, decoded=decoded, ls=self.ls
            )
        else:
            component = gc_array(self.tech, self.cfg, name=name)
            netlist = array_netlist(self.tech, self.cfg, name=name)

        gds = outdir / f"{name}.gds"
        component.write_gds(gds)
        spice = outdir / f"{name}.spice"
        spice.write_text(netlist)

        metrics = self.metrics(name, macro=macro, component=component)
        metrics_file = outdir / f"{name}.json"
        metrics_file.write_text(json.dumps(metrics, indent=2) + "\n")

        return BuildResult(
            component=component,
            gds=gds,
            spice=spice,
            metrics_file=metrics_file,
            metrics=metrics,
            decoded=decoded,
            banked=banked,
        )

    def memory(
        self,
        outdir: str | Path = "build",
        name: str | None = None,
        banked: bool = True,
        clock_ns: float = 10.0,
        interval_us: float | None = None,
        characterisation=None,
        margin: float = 0.5,
        timing=None,
        handshake: bool = False,
    ):
        """The whole memory as one hard block, ``<name>_top``: the decoded macro with its refresh
        controller synthesized, placed and routed under it by LibreLane, DRC/LVS clean and timed
        at every corner, and checked by the generated testbench at gate level (see
        :func:`gc_compiler.flow.memory.build_memory`). ``characterisation`` (what
        :meth:`characterise` returned for the same configuration) sets the controller's cycle
        counts and the macro's timing library; without it both use conservative defaults, sized
        for small macros -- a large one's read can outlast the default 4 ns, so give it a
        ``characterisation``, or a ``timing`` (:class:`~gc_compiler.rtl.ControllerTiming`) with a
        read time that covers its slowest corner (holding ``re`` longer is harmless: the macro
        ends each read itself) -- or ``handshake``: the controller then ends each read on the
        macro's own ``rdy``, however long it takes, at the cost of a synchroniser's latency."""
        from gc_compiler.flow.memory import build_memory

        if timing is None:
            timing = self._controller_timing(clock_ns, interval_us, characterisation, margin, handshake)
        name = name or f"{self.cfg.name}{'_b' if banked else ''}_mem"
        return build_memory(
            self.tech, self.cfg, name, outdir, timing, banked=banked, logic=self.logic,
            ls=self.ls if self.cfg.negative_write_wl else None, characterisation=characterisation,
        )

    def _controller_timing(self, clock_ns, interval_us, characterisation, margin, handshake=False):
        from gc_compiler.rtl import ControllerTiming, default_refresh_interval_us

        if interval_us is None:
            interval_us = default_refresh_interval_us(self.cfg, self.tech)
        if characterisation is None:
            return ControllerTiming.default(
                self.tech, clock_ns=clock_ns, interval_us=interval_us, margin=margin, handshake=handshake
            )
        reports = characterisation if isinstance(characterisation, tuple) else (characterisation,)
        data = {"corners": [c for r in reports for c in r.to_dict()["corners"]]}
        return ControllerTiming.from_characterisation(data, clock_ns, interval_us, margin, handshake)

    def refresh_rtl(
        self,
        result: BuildResult,
        clock_ns: float = 10.0,
        interval_us: float | None = None,
        characterisation=None,
        margin: float = 0.5,
        timing=None,
        handshake: bool = False,
    ) -> list[Path]:
        """Write the refresh controller, a behavioural model of the macro, a top level and a
        testbench (see :mod:`gc_compiler.rtl`) next to the GDS; returns their paths.

        ``characterisation`` (what :meth:`characterise` returned: one report, or a pair for a
        banked macro) sets the read, write and precharge times, from the slowest corner;
        without it the controller uses :class:`~gc_compiler.rtl.ControllerTiming`'s
        conservative defaults. ``interval_us`` is the refresh interval to keep; by default the one
        measured for this cell (:func:`~gc_compiler.rtl.default_refresh_interval_us`), else measure
        it (:func:`~gc_compiler.verify.timing.macro_refresh_interval`, worst corner and mismatch).
        ``handshake`` ends each read on the replica's ``rdy`` instead of after a fixed read time
        (see :class:`~gc_compiler.rtl.ControllerTiming`).
        """
        if not result.decoded:
            raise ValueError("the refresh controller drives the address pins: build(macro=True, decoded=True)")
        if not self.cfg.self_timed_read:
            raise ValueError(
                "the controller holds re for a whole number of clock cycles, and no single read time "
                "is right at every corner (a slow corner needs longer than a fast corner's stored '0's "
                "last): build the macro with self_timed_read, which ends each read by itself"
            )
        from gc_compiler.rtl import refresh_rtl

        if timing is None:
            timing = self._controller_timing(clock_ns, interval_us, characterisation, margin, handshake)
        name = result.component.name
        paths = []
        for filename, text in refresh_rtl(name, self.cfg, timing, result.banked, self.tech).items():
            path = result.gds.with_name(filename)
            path.write_text(text)
            paths.append(path)
        return paths

    def characterise(
        self,
        result: BuildResult,
        workdir: str | Path | None = None,
        corners: tuple[Corner, ...] | None = None,
        **options,
    ) -> Characterisation | tuple[Characterisation, Characterisation]:
        """Post-layout timing of a built, decoded macro (see :mod:`gc_compiler.verify.timing`);
        one per bank, saved as ``<name>.bank<b>.timing.json``, for a banked macro.

        Extracts the macro's GDS with parasitics and simulates reads and writes with ngspice at
        each corner (the standard five unless ``corners`` says otherwise). The characterisation
        is saved next to the GDS as ``<name>.timing.json``.
        """
        if not result.decoded:
            raise ValueError("timing characterisation needs a decoded macro (build(macro=True, decoded=True))")
        from gc_compiler.verify.timing import characterise

        name = result.component.name
        workdir = Path(workdir) if workdir else result.gds.parent / "timing"
        if result.banked:
            # both banks, from one extraction: the two are not mirror images (the predecode
            # sits in the middle, nearer each bank's far face than its near one), so neither
            # stands in for the other
            from gc_compiler.verify.parasitics import extract_flat

            flat = extract_flat(result.gds, name, workdir / "extract", tech=self.tech)
            reports = tuple(
                characterise(
                    self.tech, self.cfg, result.gds, name, workdir=workdir / f"bank{bank}",
                    corners=corners, bank=bank, flat=flat, **options,
                )
                for bank in (0, 1)
            )
            for bank, report in enumerate(reports):
                report.save(result.gds.with_name(f"{name}.bank{bank}.timing.json"))
            return reports
        report = characterise(
            self.tech, self.cfg, result.gds, name, workdir=workdir, corners=corners, **options
        )
        report.save(result.gds.with_name(f"{name}.timing.json"))
        return report
