# gc_compiler

A generic gain-cell (GC-eDRAM) memory compiler built on
[gdsfactory](https://gdsfactory.github.io/gdsfactory/), demonstrated on the open-source
SkyWater **sky130** PDK, and ported to **IHP SG13G2** (layout, DRC, LVS, characterisation and
the hard block; see
[Porting to another process](#porting-to-another-process)).

The compiler generates a **3T gain-cell** bitcell (PMOS write port, NMOS read pair), tiles
it into a mirrored array with tap rows, and emits GDS, a hierarchical SPICE netlist and
area metrics. Process knowledge lives behind one technology adapter
(`gc_compiler.tech.Technology`), and that is checked, not just claimed: every reference structure
built for IHP SG13G2 -- a process with a different stack, contact and via sizes, spacings and implant scheme --
passes SG13G2's own rules and IHP's decks, and a guard test keeps process names out of the rest of
the code. A port is an adapter, within the structural assumptions listed under
[Porting to another process](#porting-to-another-process).

```
config.py        BitcellConfig / ArrayConfig (defaults: the technology's sizing)
tech/rules.py    LayerRole, DesignRules, Technology   <- process independent
tech/{stack,electrical,tools,policy,characterised}.py  the groups an adapter fills
tech/sky130.py   Sky130 adapter                       <- every sky130 value
tech/sg13g2.py   IHP SG13G2 adapter                   <- every SG13G2 value
primitives/      geometry, contacts, device stacks
cells/           3T bitcell, tap row
array/           mirrored tiling, tap insertion, macro floorplan
periphery/       logic slice, word-line drivers (plain and negative-rail), column circuits, decoder
compiler.py      GDS + SPICE + metrics
verify/          KLayout DRC, magic LVS, ngspice retention/read, post-layout timing,
                 rules_drc: a DRC generated from the technology's own DesignRules
measurements/    (repository root) the sweeps behind every measured number in tech/*.py and this
                 README: scripts/, and their results in data/ (bash measurements/scripts/run_all.sh)
```

## Getting started

The KLayout and gdsfactory wheels need a few shared libraries, and DRC/simulation need the
sky130 PDK, so everything runs inside the nix shell:

```bash
uv sync --extra dev
uv run gc-compiler --rows 32 --cols 32 --out build --drc --lvs
uv run gc-compiler --rows 16 --cols 20 --characterise --out build
uv run gc-compiler --rows 16 --cols 20 --one-bank --characterise --monte-carlo 50 --corners tt,ss
uv run gc-compiler --rows 16 --cols 20 --one-bank --plain-cell --neg-wl --characterise --drc --lvs
uv run gc-compiler --rows 16 --cols 128 --characterise --corners tt --drc --lvs
uv run gc-compiler --rows 16 --cols 20 --characterise --corners tt,ss,ff --refresh --clock-ns 5
uv run gc-compiler --rows 16 --cols 20 --controller --out build
uv run pytest
```

The notebook also runs on Google Colab: its setup cells (section 0) detect Colab and
install what the nix shell provides -- Nix, the tools from nixpkgs at the pinned revision, both PDKs
at the verified versions, and the compiler from GitHub; run locally, they only check the tools.

`PDK_ROOT` defaults to `~/.ciel` (a ciel/volare sky130A install). For IHP SG13G2 (`--tech ihp-sg13g2`),
`ciel enable --pdk-family ihp-sg13g2 5e6d592e4002946a4616f798c357f0f3c06cf3b6` installs the version it was
verified with into the same root.

### Defaults

Each choice below was measured on extracted layout (the sections further down have the numbers);
each has a way back.

| Default | Why | Opt out |
|---|---|---|
| The whole memory: array, drivers, column circuits and the row decoder | a macro that takes `addr`/`we`/`re`/`din` and gives `dout` | `--no-decoded` (raw per-row word-line inputs), `--no-macro` (the array alone) / `build(macro=..., decoded=...)` |
| Standard-Vt write device with a 1.0 um `mr` (`BitcellConfig()`) | a written '0' reads back at every corner with no extra supply (the plain cell's fails at sf; at sf only just, and not with the loop-back: [Self-timed reads](#self-timed-reads-the-default-with-the-replica---no-self-timed)), and reads and writes of other rows do not disturb it (the LVT write device's does: [The write device's leakage](#the-write-devices-leakage-and-the-default-cell)); 154 us refresh interval | `--plain-cell` / `BitcellConfig.plain()` |
| Replica column | it times the read, and self-timed reads need it; +0.7 to 1.3% area | `--no-replica` / `replica_column=False` |
| Self-timed reads (with the replica) | no fixed read time is right at every corner; the replica ends each read itself; +2 to 6% area | `--no-self-timed` / `self_timed_read=False` |
| Two banks around one decoder (decoded macros) | 13 to 24% less area per bit | `--one-bank` / `build(..., banked=False)` |
| Word-line straps and 4x drivers from 64 columns (`WIDE_ARRAY`) | 64 columns: 1.78 -> 1.09 ns; 128: 3.42 -> 1.25 ns; 2.6-4% area | `--no-wl-straps`, `--wl-driver-strength 1` / `wordline_straps=False, wordline_driver_strength=1` |

`--rows`/`--cols` give one bank's size, so a banked macro holds twice as many bits. The
measurement scripts (`measurements/scripts/`) pin the configuration they were measured with: the plain cell, one bank,
no self-timing, no straps, and 1x drivers.

## Two DRC decks

`verify/drc.run_drc` runs the PDK's KLayout deck **and** magic's full deck (`drc(full)`, on a
flattened copy, the way LibreLane runs it) and requires both clean; magic's rules are reported as
`magic:<rule>`. The KLayout `mr` deck is a subset: it does not check poly-contact or poly spacing to
diffusion (`licon.14`, `poly.4`, `licon.9`), the wells' enclosure of taps and p+ diffusion
(`diff/tap.10`, `.8`), the varactor rules, or latch-up tap distance (`LU.2`, `LU.3`) -- and the
layout broke all of them, 2,065 violations in a 6x5 banked macro (LibreLane's magic DRC found
them). Fixed:

* **Decoder gates** (most of the count): each standard cell contacted every gate beside both
  device stacks, one pad in the narrow gap between them. Now each gate is contacted once, where the
  logic slices contact theirs (`draw_gate_pads=False` in `primitives/devices.py`).
* **Well ends**: tap and logic strips end their wells flush with their end taps, relying on the
  next cell; at the end of a row there is none. `primitives/wells.close_well_ends` moves such a
  well edge out on the assembled array, macro and decoder.
* **Strap-column taps**: each row pair's n-well tap sat under the write word lines' poly -- a
  varactor (LVS had been stripping it as a "parasitic"; it no longer does). The tap now sits where
  at most one word line crosses it, and that one is cut either side of the tap, landed, and bridged
  in li; the column is 0.29 um wider (2.71 um, so a neighbouring driver's gate contacts keep their
  npc spacing too).
* **Latch-up**: every row's vss rail gets a p-tap in the strap column (the pairs' full-width
  n-well bands fence the p-strips between them off from the tap rows), and the strap columns sit
  half an interval in from the ends, so no column is more than 8 columns (10 um) from one; LU.2/3
  allow 15 um. The old placement (after every 16th column and at the end) left the first 16
  columns up to 20 um from a tap; the new one needs one more strap column where the array is a
  multiple of 16 wide (a 32-column array has three), so 32x32 costs about 3% more area per bit.

All the areas in this README are re-measured on the fixed layout (every macro has at least one
strap column, now 0.29 um wider, and the negative rail's strip is 0.18 um wider per side); most
moved by 0.5 to 1%, 32-column arrays by about 3%. Timing, energy and retention figures were
measured before these fixes, and stand: re-characterised afterwards (the notebook's three 16x20
macros at five corners), every access, cutoff and write time is within 0.01 ns of before (plain
cell 1.03 -> 1.02 ns at tt, 1.88 -> 1.87 at ss; LVT with both read gates 1.52 / 2.81 -> 1.52 / 2.82;
the self-timed cutoff 2.22 -> 2.23 at tt, 4.22 at ss unchanged).

## Porting to another process

**What is checked.** Everything the compiler knows about a process is on its `Technology`, and
three tests keep it that way:

* `tests/test_portability.py` builds twelve reference structures -- bitcells, a strapped array, plain,
  decoded, banked, wide, negative-rail and strong-driver macros, a hard macro with power stripes --
  with sky130 and with IHP SG13G2, and checks each against *its own technology's* rules with
  `verify/rules_drc.py`, a DRC generated from `DesignRules` (every rule field is checked or listed
  with the reason it is not). A number standing for a sky130 rule passes with sky130 and fails with
  SG13G2 (the check includes the grid). It needs only the adapters, not the PDKs.
* `tests/test_tech_neutral.py` counts, in the code outside `tech/`, strings naming process things
  (`sky130`, `met1`, `li1`, `VPWR`, `nfet`, `sg13`, `Metal1`, `TopMetal`, `Activ`, `pSD`, deck and
  setup file names) and bare decimal numbers; the counts may only go down. Sky130 names: 25 before, 0 now.
* `tests/test_layout_regression.py` pins the sky130 layout (per-layer geometry hashes of the twelve
  structures), so a refactor that must not move anything provably does not. The SG13G2 port left it
  unchanged: every term added for SG13G2 is a `max()` with the old value.

A technology with nothing measured (`characterised = None`, as SG13G2 so far) makes the compiler
refuse to guess: no default refresh interval (one measured on sky130 could lose data on a leakier
process) and no default controller timing (`NotCharacterised`). A configuration sized for another
technology is refused where its devices are drawn, not drawn illegal:
`BitcellConfig.for_tech(tech)`, `LogicConfig.for_tech(tech)`, `ArrayConfig.for_tech(tech)` size it
for the technology in use (the plain constructors keep the default technology's sizes).

**What a port supplies** (`tech/sky130.py` and `tech/sg13g2.py` are the models; `--tech ihp-sg13g2`,
or `--tech package.module:Class` for your own):

| group | what |
|---|---|
| layers | a GDS layer per `LayerRole`, label and pin layers, the implant/well/tap policy, marker decorations (sky130's npc) |
| `DesignRules` | widths, spacings, enclosures (with the relaxed short-side values the PDK allows), cut sizes and spacings, areas, device rules, latch-up distance |
| `MetalStack` | routing layers and their cuts, the contact and local-interconnect layer, LEF layer names, the power-pin layer |
| `Electrical` | nominal and rated supply, the corners (process, V, T), sheet resistances, gate capacitance |
| `ToolSetup` | the KLayout deck, its variables and any extra decks, the magic and netgen setups, whether magic's DRC is part of sign-off, the rules a leaf-cell check waives |
| `standard_cells()` | liberty per corner and its (V, T), Verilog models, placement site and row, supply pin names, power grid, RC corner, area per flop, SDC clock assumptions |
| `LayoutPolicy` | tap-row and strap-column intervals, the power stripes, overlap and margin |
| `Sizing` | the default cell, logic, predecode and level-shifter device sizes |
| `Characterised` | measured refresh intervals, strobe factor, controller timing (or `None`) |
| `spice_model`, `model_header`, `classify_model` | the device models, how a bench loads a corner, and how extraction's models are told apart |

### IHP SG13G2

`gc-compiler --tech ihp-sg13g2 --rows 8 --cols 8 --drc --lvs` builds a macro, runs IHP's KLayout
decks (the main rule table and the "extra" rules IHP's runner adds by default) and extracts it with
IHP's magic setup for netgen LVS; `--characterise`, `--controller` and `--neg-wl` work as on sky130.
The adapter (`tech/sg13g2.py`, IHP-Open-PDK commit pinned in `PDK_VERSION`) maps the compiler's layer
roles one metal up: SG13G2 has no local interconnect, so Metal1 plays sky130's li1 and Metal2..Metal5
its metal1..metal4; the macro's power pins are on TopMetal1, the vertical layer of IHP's
place-and-route grid, which a hard block's TopMetal2 straps land on.

| 8x8, decoded | sky130 | SG13G2 |
|---|---|---|
| bitcell (default) | 1.22 x 2.93 um (3.58 um^2) | 1.37 x 3.20 um (4.38 um^2) |
| one bank | 77.1 x 50.8 um, 61.2 um^2/bit | 78.8 x 51.0 um, 62.8 um^2/bit |
| two banks | 134.4 x 48.3 um, 50.7 um^2/bit | 138.2 x 47.3 um, 51.0 um^2/bit |
| refresh interval (worst of 20 dies, ss, 125 C) | 154 us | 100 us |
| read, 16x20 macro, ss | | 3.87 ns |

What the port measured (`measurements/scripts/sweep_retention_sg13g2.py`, `measurements/data/*sg13g2.json`):

* **The cell.** SG13G2 has no LVT device, so sky130's LVT write is not available. A plain cell's
  written '0' sits at ~0.4 V (the PMOS write stops ~|Vtp| above ground), but its read threshold is
  ~0.73-0.77 V, so it reads -- for 35 us at ss/125 C (26.9 us on the worst of 20 mismatch dies).
  Lengthening the storage node's read device (`mr`) stores more charge: 1.0 um gives 121 us nominal,
  100 us worst die, for 28% more cell area -- the default. A longer write device adds nothing.
* **The negative write rail** (`--neg-wl`, its isolated p-well IHP's nBuLay under an NWell ring) is
  at -0.12 V on SG13G2, so a write puts no more than the LV devices' rated 1.32 V across a gate
  oxide (-0.3 V would be 1.50 V). It lowers the written '0' (0.30 V at ss for the default cell) but
  leakage at 125 C still sets the interval: 120.9 us worst die (+21%), 37.8 us for the plain cell
  (+40%); not the default.
* **The controller's timing** (read 8 ns, write 10, precharge 2) comes from the macro's five-corner
  characterisation, rounded up as sky130's is; the write is the ~10 ns a '0' takes to settle through
  the PMOS write device (its crossing is 1.1 ns), which a refresh's write-back needs too. Its own share of a read (`sample_ns` 2.0) is from the
  routed blocks' static timing at ss: 1.53 ns past the macro's read for one bank, 1.79 banked
  (sky130's 1.5 would not do); the hold budget (0.5 ns) is under the fastest output, a 4x4 macro's
  0.57 ns at ff.
* **Simulation** needs IHP's PSP models compiled to OSDI: the nix shell provides OpenVAF (23.5.0,
  pinned), and the adapter compiles the models on first use into `build/osdi/`. IHP's mismatch
  models are switched on per instance (`mm_ok=1`); the adapter's mismatch library makes that the
  default, so Monte Carlo works as on sky130.
* **Sign-off is KLayout.** IHP's magic DRC deck (beta) applies the contact-bar rules CntB.b/h1 to
  square contacts, so magic is used for extraction only (`ToolSetup.magic_drc = False`, also in the
  hard block's LibreLane run); IHP's extra-rules deck is run on the macros and on the routed block.
* **The hard block** (`--controller`): LibreLane's grid stops its TopMetal1 straps at a macro, so a
  strap of each net is aligned onto the macro's own stripes (`PowerGrid.align_to_macro`), which carry
  the grid across it. 8x8 blocks, one bank and banked, are DRC and LVS clean, meet timing at
  tt/ss/ff and pass their gate-level testbenches.

**Structural assumptions** -- what an adapter cannot change:

* every contact lands on the local-interconnect role, and vias go up from it: sky130's li1, or on a
  process without one, its first metal (as SG13G2; the cost is one metal);
* an n-well process: PMOS in n-well, NMOS on the substrate;
* at least four routing layers above the local interconnect, word-line straps on the third; the hard
  macro's power pins on `MetalStack.power_pin_layer` (the fourth on sky130, TopMetal1 on SG13G2);
* an isolated p-well (a deep n-well, or SG13G2's nBuLay) for the negative-rail driver;
* minimum widths and device widths an even number of grid steps (drawn centred, they would
  otherwise land off the grid) -- device widths are checked and refused; the bitcell's column pitch
  is rounded up to even steps, since the column circuits centre themselves on it.

**What making it true turned up.** Most sky130 numbers in the code were coincidences a second
process exposes: an li strap exactly li.6's minimum area (0.33 x 0.17 = 0.0561), li exactly as wide
as a contact (both 0.17), a contact pitch and device widths that happen to be even grid steps, a
predecode channel gap and a tap-row placement with room to spare only at sky130's sizes. The
negative-rail driver (the worst: `y_inb0 = 2.6` um, hand-computed track pitches) is now laid out
from its rules. SG13G2 added what a same-stack test could not: a via larger than the contact below
it (Via1 0.19 over Cont 0.16: the via gets its own pad), metal spacing half again sky130's (via
stacks, the bitcell pitch and the decoder's pin offsets now keep it), minimum metal areas met by
growing each merged shape in place (`fix_min_area`) rather than per pad, implants that must enclose
a gate by more than the diffusion (pSD.i: the implants no longer overlap anywhere, checked), and
diffusion and taps on one layer. Two real bugs came out before: the negative-rail driver column's
n-implant ended flush with its end diffusion, and a strapped word-line driver's via2 pad could
leave a metal2 notch against its own output leg. The sky130 layout moved only where a rule-derived
value differs from the old number (smaller li pads, the negative-rail slice), and every sky130
DRC/LVS test still passes.

## Status

| Area | State |
|---|---|
| 3T bitcell layout | DRC clean, 1.22 x 2.46 µm (3.00 µm²) on sky130 (the plain cell; the default 1.22 x 2.93) |
| Array tiling, tap rows, shared bit-line contacts | done, DRC clean |
| Word-line drivers (landed from alternating sides) | done, DRC clean, LVS topology match |
| Column circuits (precharge, read buffer, write driver) | done, DRC clean, LVS topology match |
| Macro floorplan, SPICE netlist, metrics, CLI | done; macro DRC clean and LVS topology match, with and without decoders |
| Sign-off DRC: KLayout + magic `drc(full)` (`run_drc` requires both) | done: every macro, the decoder and the array clean under both |
| LVS (magic extract + netgen compare) | done |
| Vertical vss/n-well strap columns | done, DRC clean, LVS topology match |
| ngspice retention / read testbench | done |
| Static row decoder (netlist + pitch-matched routed layout) | DRC clean, LVS topology match, 2 to 64 rows |
| Decoder-to-macro physical integration (`decoded=True`) | done, DRC clean, LVS topology match |
| Replica bit-line column (default; `--no-replica`) | done, DRC clean, LVS topology match, +0.7 to 1.3% macro area |
| Post-layout timing characterisation (`--characterise`) | done: read/write path by stage, five PVT corners, `rdy` tracking |
| Mismatch Monte Carlo (`--monte-carlo N`) | done: seeded per-device mismatch dies at each corner, `rdy` margin at 3 sigma |
| Negative-rail write word-line driver (`--neg-wl`) | done: isolated-well driver + level shifter, DRC clean, LVS match; a written '0' reads back in every corner; +15 to 30% macro area |
| LVT write device + longer `mr` (`--write-flavor lvt`; the default before) | done: no extra supply, DRC clean, LVS match; a written '0' reads back in every corner; but its write device leaks 2.6 to 134 nA into the column's '0's whenever the write bit line is high, which reads and writes of other rows do |
| Default cell: standard-Vt write + 1.0 um `mr` | done: reads every corner on the extracted macro, idle and under read traffic (at sf with no margin: 1 of 20 mismatch dies fails, 18 of 20 with the loop-back refresh); 154 us refresh (worst of 20 dies, ss, 125 C) |
| Banked macro: two arrays, one central decoder (default; `--one-bank`) | done: DRC clean, LVS match with pin list; 13 to 24% less area per bit than a plain decoded macro; timing per bank (faster than plain at 32x32) |
| Word-line straps (`--wl-straps`, default from 64 columns) | done: metal3 over every word line, tied down in every strap column; DRC clean, LVS match with pin list (plain, negative rail, LVT, banked); no area; 16x128 access 3.42 -> 1.98 ns |
| Stronger word-line drivers (`--wl-driver-strength N`, default 4 from 64 columns) | done: scaled inverters and negative-rail output stages in wider driver columns; DRC clean, LVS match with pin list; with straps, 16x128 access 1.98 -> 1.25 ns at 4x for +2.6% area |
| Self-timed reads (default with the replica; `--no-self-timed`) | done: the replica's `rdy` gates the decoder's read enable, so every read ends itself and `dout` holds; reads correct at every corner however long `re` is held; DRC clean, LVS match with pin list (plain, banked, with straps, negative rail, 4x drivers); +2 to 6% area |
| Refresh (`--refresh`) | done: one access per row; by default the macro writes the row back itself (write driver `NAND(din, rbl)`; `--no-refresh-in-macro` loops it through the controller); restores decayed '1's and '0's on the extracted macro at every corner and under 20 mismatch dies at ss, up to 32 rows a bank on sky130 (taller banks write a decayed '1' back low: [the row cap](#refresh---refresh)); controller + behavioural model + testbench (iverilog) |
| Controller as hardware (`--controller`) | done: the macro and its refresh controller as one LibreLane-built hard block (GDS, LEF, timing libraries, netlists); DRC/LVS/antenna clean, timed at tt/ss/ff, the testbench passing on the routed netlist; the latch controller 2.1k (2 x 16x20) to 2.5k um2 (2 x 32x32) of cells; block 33.5 / 18.1 um2/bit against the macro's 19.3 / 12.5 |

### Address decoding

`netlist/decoder.py` generates the static row decoder, built so its row gates can sit at
the word-line pitch: every output is **one 2-input NAND** (plus an inverter for the write
rows). The address is split into a low and a high part, each predecoded to one-hot *bus
lines* that run past every row -- `L` lines from the low bits, and `H` lines from the high
bits already ANDed with the enable (one set with `we`, one with `re`). A bus line is one NOR
over the bits' *complement* literals (an AND is a NOR of inverted inputs), or for more than
three bits NANDs of groups feeding a NOR, so it comes straight out of a gate with no
inverter behind it. Row `r` is `NAND(L[low(r)], H[high(r)])`. The split is whichever needs
the fewest gates, and only the lines the requested outputs use are built.

The inverting word-line drivers require **active-high `wwl_in`** (AND of select and `we`)
and **active-low `rwl_in`** (NAND of select and `re`). This makes the selected PMOS write
word line low and the selected NMOS read word line high. Disabled enables and unused
addresses select no rows. `tests/test_decoder.py` checks schematic/layout gate-plan
equivalence through 64 rows, including split decoders and non-power-of-two sizes, and
checks the decoder together with the word-line drivers. Four-row ngspice operating-point
tests check all enable combinations; this is not transient timing or glitch verification.

`periphery/decoder.py::row_decoder()` builds the routed decoder from the fixed-height cells
below (used at their natural width, not rounded up to a placement site: a `nand2` is 2.29 um,
under the 2.46 um word-line pitch). Its floorplan, bottom to top:

* **Row strips.** One holds every row's NAND, the other the write rows' output inverters.
  Their cells are *placed at the driver inputs*: each output's pin lands on, or a fraction
  of a micron from, the driver input it feeds. Everything between a row cell and its driver,
  and between a write row's NAND and its inverter, is a straight or lightly jogged metal2
  wire -- no track, no router. The inverters are placed *flipped*, which puts their p-tap
  side toward the drivers (no n-well within 1.27 um of the driver column's) and lets their
  vdd rail land on the NAND strip's vdd rail: the two strips share one rail and overlap.
* **Bus channel.** Each predecode line is one long metal1 track that every row NAND taps
  with a metal2 stub from its input pin. Routing channels have no metal1 in them, so the
  tracks sit on it (via1 to the stubs): no metal3, and a tighter track pitch (0.405 um, with
  pads elongated along the track, vs 0.67 um on metal3).
* **Predecode strips** above it, one per logic level (a literal-inverter level, then one NOR
  level), gates scheduled as late as their consumers allow so a bus line is always produced
  in the strip just above the bus channel.

Pins never run through a cell: a `y` pin sits in a cell's lower half and exits down, `a`/`b`
exit up (a NAND also has an internal metal2 strap across its lower half). Where two stubs from
opposite sides of a channel could touch, the tracks are ordered from a vertical-constraint
graph and nets with disjoint extents share a row (`periphery/channel.py`); a net that cannot
be ordered is split into one track per side. Nets with several tracks are tied by a metal2
trunk beyond the strips; supplies reach a metal1 (vdd) and a metal2 (vss) trunk from each
strip's rail. Default cell names distinguish output subsets and logic sizing, so left/right
half-decoders cannot accidentally reuse the same cell. Built alone (no interface), the
outputs are spaced at the array's word-line pitch, in the order given.

A predecode line is one NOR driving a long metal1 bus, so it rises through up to three series
PMOS of the cell's device width; the schematic-level checks here (function, DRC, LVS) do not
cover that timing, and a slower line would want wider PMOS in those gates.

Physical regressions cover full/split decoders from 2 to 64 rows with Sky130 DRC and filtered
topology LVS (also checked by hand up to 128 rows and across non-power-of-two sizes). The
port-list caveat above still applies. Notebook section 14 generates and plots an 8-row
standalone decoder, writes its GDS and reference SPICE to `notebooks/build/decoder/` when run
from the notebook directory, and runs DRC/LVS on it.

### Fixed-height decoder cells

`periphery/standard_cells.py` is the foundation for replacing the decoder's irregular
gate column with standard-cell rows. `standard_cell(tech, "inv")` and
`standard_cell(tech, "nand2")` (and `nand3`, `nor2`, `nor3`) have a common height, horizontal M1 power rails, aligned
well/implant/tap bands, explicit placement boundaries, and M2 signal access. Widths
are multiples of a process-derived placement site. The NAND shares a PMOS diffusion
strip with node order `y-vdd-y`, rather than placing the parallel PMOS devices in
separate columns. A common `LogicConfig` sets transistor sizing for the entire family;
its legacy `pitch` override is not accepted by these cells.

```python
from gc_compiler.periphery import standard_cell, standard_cell_row
from gc_compiler.tech.sky130 import Sky130

tech = Sky130()
row = standard_cell_row(tech, ("inv", "nand2", "inv"))
row.write_gds("build/decoder_standard_cell_row.gds")
```

The row's supplies connect by abutment, without per-gate supply jumpers. Signal pins
remain independent (`a_0`, `y_0`, `a_1`, `b_1`, etc.). Tests cover individual gates at
two transistor sizings and identical/mixed row abutments, with DRC, filtered topology
LVS, and a geometric check that each supply rail spans the complete row. The LVS port
list caveat still applies. The NAND's `a` input drives the NMOS next to the output and
`b` the one next to vss, matching `nand2_subckt`; the inputs are logically but not
electrically interchangeable, and a lone-cell LVS cannot tell, since netgen may permute
the cell's ports. `row_decoder()` above routes these cells; mirrored row sharing is
future work.

### Existing decoder integration

`periphery/logic.py::nand2()` is the physical NAND2 gate this all runs on -- parallel PMOS
pull-up columns, one series-NMOS pull-down, DRC clean and LVS-verified. Building it
surfaced a real lesson: DRC was clean on the first working version, but LVS caught that
both inputs' jumpers ran through the same y-level and shorted them together (netgen:
"Merged 1 parallel devices"). The fix routes the farther input over a dedicated metal2
channel above the whole cell (`_connect_over()`) instead of a direct jog. See
`/memories/repo/conventions.md` for the general lesson (DRC cannot see connectivity bugs;
always LVS a new multi-device logic cell).

**Decoders in the macro.** `gc_macro(..., decoded=True)` (CLI: `--one-bank`, or
`GainCellCompiler.build(banked=False)`) places a half-decoder beyond each driver column and
wires it up; the macro then takes `addr_*`/`we`/`re` instead of the raw per-row
`wwl_in_r`/`rwl_in_r`. A decoded macro is the default (two banks of it, below);
`--no-decoded` / `decoded=False` keeps the raw inputs, and the netlist the compiler writes
always matches the layout it built. See `array/wordline_decoder.py`:

* The left decoder is rotated a quarter turn and the right one transposed, so both face
  their driver column and their strips are ordered the same way up the macro.
* A decoder routes its own outputs: each driver input is handed to it as a target
  (`DecoderInterface.targets`), its row cells are placed at those targets, and each output
  reaches its driver on a direct metal2 wire.
* `addr`/`we`/`re` are shared by the two halves. Each gets a metal2 trunk pinned to the
  same macro y in both decoders (`DecoderInterface.trunks`), above everything else in the
  macro, and one straight metal2 wire joins them. Supplies leave each decoder the same way
  and land on the driver columns' metal1 rails, which are extended up to meet them.

Because the row cells sit at the driver inputs, a decoder is only as wide (along the array)
as the array is tall and about 30 um deep for 32 rows: a 32x32 macro is 119 x 110 um (24%
array efficiency, against 51% without decoders and 7.5% for the first, unmatched decoder),
64x32 is 144 x 199 um, 128x64 200 x 378 um (33%). Most of a decoder's depth is its routing
channels (the predecode bus and the literal lines) rather than its gates.

`tests/test_wordline_decoder.py` checks that each of the `2 * rows` word-line inputs is fed by
exactly its own decoder output, that the shared trunks are one net across both decoders, and
that no signal touches a supply (`verify/connectivity.py`, a KLayout net tracer); DRC on the
assembly and on a decoded macro; device-level LVS of each half-decoder exactly as the macro
builds it; and LVS of both decoders together with the real driver slices against gates (the
two driver columns get a vdd each, since nothing joins them without the array).

**Driver-slice short (fixed).** Wiring the decoders up exposed a bug in `periphery/wl_driver.py`
that predates them: the input pad sat in the 0.22 um between the inverter's gate and its source
contact, where its metal1 overlapped the inverter's own vss strap and its metal2 the output
leg, so *both inputs of every slice were shorted to vss and to the output*. DRC cannot see it --
the overlapping shapes merge into one polygon -- but magic LVS and the net tracer both do. The
contact is now lifted on a li strip to just above the output leg, so the `*_in` ports sit at
y = 0.96 / 3.42 um of a slice rather than 0.145 / 2.605. `test_driver_slice_nets_are_all_separate`
and the driver-inclusive LVS above both fail on the old code.

**Macro-level LVS passes**, with or without decoders (layout vs. `macro_netlist(...)`, checked at
8x4 up to 32x32, including sizes past a strap column; `gc-compiler [--no-decoded] --lvs`).
Getting there took three more fixes, none of them visible to DRC:

* The column circuits had the same short as the drivers: the write driver's and read buffer's
  gate contacts overlapped the inverter's own vss stub, shorting `din` and `rbl` to vss. The
  contact is now made at the bit-line track, on a li strip that runs sideways from the pad
  (`periphery/column.py::_lift_input`).
* The column circuits were placed at a uniform pitch, but the array inserts a strap column every
  16 columns and shifts every later column by its width -- past 16 columns every bit line missed
  its column circuit. They now sit at the array's own column positions, with their horizontal
  rails bridged across the strap gaps (`array/macro_supply.py`).
* Nothing distributed vdd: each driver column and each column-circuit group had its own. Each
  side's driver rails are now extended down beside the column circuits and tied to the column
  rails -- vss on metal1, vdd on metal2 over the vss rail -- so vdd and vss are single nets
  across the macro (the array's own `vnw` stays separate, as in the netlist). The macro exposes
  `din_*`, `dout_*`, `pch`, `vdd` and `vss`, labelled for LVS.

Macro LVS checks the pin list too (the port-list fix above): every pin but the array's own `vnw`.

### Banked macro: one decoder for two arrays (the default; `--one-bank` for one)

A single array already pays for two predecodes: a row's two word lines are too close to land
from one side (`periphery/wl_driver.py`), so `side_outputs()` gives each half-decoder one of
them for *every* row, and each half decodes the whole address. `gc_banked_macro()` (the default
decoded build: `gc-compiler`, `GainCellCompiler.build()`) puts two banks of
`rows x cols` on either side of one **central decoder** and adds one address bit, the top one,
to pick the bank. The bank bit is simply the address bit above the row bits -- bank `b`'s row
`r` is decoded as logical row `b * 2**n + r` -- so it folds into the `H` lines with the enable
and the gate generator is unchanged. At the logic level the predecode per bank falls from 46 to
19 gates for 32 rows, 110 to 36 for 128; the row gates cannot be shared and stay the same.

```
[outer drv][ bank 0 ][inner drv][ face 0 | predecode | face 1 ][inner drv][ bank 1 ][outer drv]
    ^================= metal3 ====|                              |==== metal3 =================^
```

* Each bank still needs driver columns on both edges, so its *face* of row gates
  (`array/wordline_decoder.py::central_decoder`) holds two groups around one bus channel: the
  near group, whose outputs reach the inner column on metal2, as before, and the far group,
  mirrored beyond the channel (`[near inv][near NAND][bus channel][far NAND][far inv]`), whose
  outputs rise to metal3 on their pins, pass over the near group and cross the bank -- one
  track per row, on a layer the array does not use -- to the outer column. Merging the groups
  saves a channel and a gap per bank, but not tracks: every bus net must reach its trunk on the
  strips' left, so no two share a row, and the merged channel holds the union of both groups'
  nets (22 / 28 / 44 tracks at 16 / 32 / 64 rows).
* The faces and the predecode are parts of one decoder (`periphery/decoder.py`, `part=
  "rows2"`/`"predecode"`), and every part takes the *whole* decoder's low/high address split: a
  face's own cheapest split names its bus lines differently. (LVS caught it at 12 rows, where
  the splits first differ.)
* **The predecode is packed**: its logic levels fill as few strips as fit along the rows
  (`_pack_levels`), instead of one strip per level however short. A strip no longer than the
  rows keeps the supply trunks -- aligned across all parts -- from lifting the macro's top edge
  (at 16 rows the first level alone was longer than the rows). Packing is not free: a net whose
  producer and consumer share a strip needs a trunk, which costs height and loads the gate
  driving it. The central decoder is built both ways (`DecoderInterface.pack`) and keeps the
  smaller; its 4x predecode barely feels the extra wiring. A plain half-decoder's minimum-size
  predecode does -- packed, a plain 32x32 macro was 4% smaller but read 8% slower, and 64x32 9%
  smaller but 24% slower (1.95 -> 2.42 ns) -- so the plain macro stays unpacked.
* The bus lines join the parts on trunks pinned at the same column in all three. Placed as
  before, above the array, they cost 11 to 15 um of macro height and the entire saving (a
  32x32 bank came out at parity). They now sit on the strips' other side, below the rows, in the
  column-circuit band, which is empty beside the decoder (`DecoderInterface.trunks` left of the
  strips; `periphery/channel.py`'s `tied_left`). The band is 12.8 um; the trunks need 10.6 um at
  16 rows, 14.7 at 32 and 26.7 at 128, so from 32 rows up they extend the macro downward a
  little.
* `pch` joins the banks on metal3 under the decoder (at its height the driver columns' supply
  rails and the column circuits' metal2 leave no room on the lower layers); a negative-rail
  macro's `vneg` link crosses the decoder the same way.

Pins: `addr_0..addr_n` (`addr_n` picks the bank), `we`/`re`, `din_{bank}_{col}`,
`dout_{bank}_{col}`, `rdy_{bank}` with a replica, and one `pch`/`vdd`/`vss` (and `vneg`).
Each array keeps its own n-well bias, so the netlist has `vnw_0`/`vnw_1`.

| per bank | plain decoded (1 bank) | banked (2 banks) | area per bit | array efficiency |
|---|---|---|---|---|
| 16x20 | 93.5 x 64.7 um | 163.0 x 61.4 um | 18.88 -> 15.65 um2 (-17%) | 15.9 -> 19.2% |
| 32x32 | 118.9 x 109.6 um | 208.3 x 105.9 um | 12.73 -> 10.77 um2 (-15%) | 23.6 -> 27.9% |
| 64x32 | 143.8 x 199.0 um | 219.5 x 198.3 um | 13.98 -> 10.62 um2 (-24%) | 21.5 -> 28.3% |
| 128x64 | 200.0 x 378.0 um | 341.6 x 384.0 um | 9.23 -> 8.01 um2 (-13%) | 32.5 -> 37.5% |

Before the predecode was packed and the faces merged (and before the DRC fixes), the banked
column read 17.34 / 11.22 / 11.50 / 8.44 um2 (and plain 18.77 / 12.38 / 13.66 / 9.05): packing alone took 16x20 to 15.98,
32x32 to 10.75 and 64x32 to 10.61; merging the faces about 3% more.

**Timing: the shared predecode has to be driven harder.** Its lines drive the row gates of
both banks plus a trunk across the decoder, against one face in a plain macro. Built from
the plain decoder's cells, the banked 16x20 macro's decoder took 1.5 to 1.8 ns at tt against
0.5 to 0.65 -- the whole access doubled (2.1 / 2.4 ns per bank against 1.02). Since this
predecode is built once for two banks, bigger devices cost little area per bit;
`PREDECODE_LOGIC` (`periphery/logic.py`) sizes it at four times the logic cells' widths, and
`gc_banked_macro(..., predecode_logic=)` trades further (16x20 with replica, tt, the slower
bank's worst read, re to dout; measured before the predecode was packed and the faces merged,
so the areas are the old ones):

| predecode PMOS / NMOS width (um) | 0.64 / 0.42 | 1.28 / 0.84 | 1.92 / 1.26 | 2.56 / 1.68 (default) | 3.84 / 2.52 |
|---|---|---|---|---|---|
| decoder (ns) | 1.79 | 1.10 | 0.84 | 0.70 | 0.56 |
| access (ns) | 2.34 | 1.62 | 1.34 | 1.20 | 1.05 |
| area per bit (um2) | 16.93 | 17.15 | 17.37 | 17.59 | 18.03 |

Against a plain decoded macro, same bench (replica, tt, worst read of a fresh '1' over the
first and last rows and both end columns; write of a '1' into the last row's far cell):

| per bank | plain: access / write / area | banked, bank 0 / bank 1: access | write | area |
|---|---|---|---|---|
| 16x20 | 1.02 ns / 0.15 ns / 19.13 um2 | 1.14 / 1.14 ns | 0.16 ns | 15.88 um2 (-17%) |
| 32x32 | 1.66 ns / 0.18 ns / 12.86 um2 | 1.43 / 1.50 ns | 0.19 ns | 10.90 um2 (-15%) |

At 32x32 the banked macro is both smaller and faster: a 32-row face loads the plain macro's
minimum-size predecode enough that the bigger shared one wins despite driving both banks.
Merging the faces brought the banks together (they differed by 10 to 12% when bank 1's far
face sat beyond the predecode; now by up to 5%), and still both are characterised (`--characterise` on a banked macro extracts once and reports one
table per bank, saved as `<name>.bank<b>.timing.json`; `build_model(..., bank=)` drops the other
bank's cells and column circuits and keeps its row gates, which load the shared predecode).
`rdy` tracks as before (dout/rdy 1.02 at tt).

`tests/test_banked.py`: the standalone central decoder is DRC clean and LVS-matches the flat
banked schematic pin for pin (6 and 16 rows); the whole macro is DRC clean and matches
`macro_netlist(banked=True)` pin for pin but `vnw_0`/`vnw_1` (10x6, 14x5 with a replica, 6x7
with the negative rail); and it beats a plain decoded macro of the same bank size.
`tests/test_decoder.py` checks every address of both banks selects exactly one row of one bank.


## Word-line straps (`--wl-straps`; on by default from 64 columns)

A word line is poly (48 ohm per square) across the whole array, so its far end lags the end its
driver is at: at 128 columns a read word line is about 60 kohm end to end. `ArrayConfig(
wordline_straps=True)` / `--wl-straps` adds a metal3 line over every word line, fed by its driver
and tied down to the poly in every strap column, so no stretch of poly is longer than
`strap_column_interval` columns (16 by default). The array uses no metal3, so straps cost no area.

* **Tracks** (`periphery/wl_driver.py::strap_track_y`). Metal3 is 0.3 um wide with 0.3 um
  spacing, but a strap is 0.37 um -- as wide as a via2 landing, so the landings need no pads that
  would poke toward the next strap. A row pair has four word lines; the two write lines are 0.47
  um apart, and a read line is 0.44 um from the next pair's, so those straps are pushed apart,
  symmetrically about the pair's middle and edges (by at most 0.12 um from their word line).
* **Ties** (`array/strap_column.py`). Every word line gets a contacted poly landing in each strap
  column (before, only the lines crossing n-well had one), stacked to metal2, a short metal2 jog
  to the strap's track, and via2. The landings alternate between the column's two slots, and a
  read line's landing slides 0.015 um toward its pair's middle to keep poly spacing from the next
  pair's read line, which only the strap column brings that close.
* **Drivers** (`_route_to_wordline`, plain and negative-rail): the output's metal2 dog-leg also
  rises to the strap and runs it to the slice's array edge.
* **Banked macro.** Its far decoder outputs already crossed the bank on metal3, one line per row
  at its driver input's height. With straps they move to the two spare tracks between a pair's
  straps (`far_track_y`) and jog down to the driver input on metal2 just outside the outer column.
* **Timing model** (`FlatNetlist.add_wordline_rc(taps=...)`): each word line's poly ladder gets
  a parallel metal3 ladder from the driver end (0.047 ohm per square), joined to the poly at each
  strap column through the contact stack (licon, mcon, via1, via2: 169 ohm).

Extracted, tt, with replica; worst read of a fresh '1' over the end rows and end columns:

| macro | access (ns) | word-line driver stage (ns) | `rdy` (ns) | write, far / near cell (ns) | area per bit |
|---|---|---|---|---|---|
| 16x64 | 1.78 | 0.75 | 1.04 | 0.29 / 0.15 | 9.88 um2 |
| 16x64 strapped | **1.47** (-17%) | 0.52 | 1.43 | 0.23 / 0.23 | 9.88 um2 |
| 16x128 | 3.42 | 2.25 | 1.04 | 0.41 / 0.15 | 7.75 um2 |
| 16x128 strapped | **1.98** (-42%) | 0.93 | 1.94 | 0.30 / 0.30 | 7.75 um2 |

* **Unstrapped, self-timing breaks on wide arrays.** The replica column sits next to the driver
  of half the word lines, so at 128 columns `rdy` rises at 1.04 ns while the far column's data is
  valid only at 3.42 ns (3.3x): a strobe at 1.25x `rdy` would read the far column wrong. Strapped,
  every column sees nearly the same word line, and `rdy` tracks again (dout/rdy 1.02).
* **The cost** is on the near side: the minimum-size driver now also charges the strap, so a
  write near the driver takes 0.23 to 0.30 ns instead of 0.15. At 128 columns the driver stage is
  still half the access -- which the next option addresses.

**Stronger drivers** (`ArrayConfig(wordline_driver_strength=N)` / `--wl-driver-strength N`).
The word-line drivers' device widths are the logic cells' times N (`periphery/logic.py::
driver_logic`); the devices lie along the driver column's width, so the column gets wider, not
taller. The netlist gets its own `gc_inv_drv` subcircuit for them (the column circuits keep
`gc_inv`), and a negative-rail driver scales its output stage and its level shifter's input
inverter; at 4x its output NMOS moves 0.05 um right, so its now edge-to-edge contact li clears the
gate pad it shares with the latch. Strapped, tt, with replica:

| macro | strength | access (ns) | write, far / near (ns) | area per bit |
|---|---|---|---|---|
| 16x64 | 1 / 2 / 4 | 1.47 / 1.21 / **1.09** | 0.23 / 0.23 -> 0.18 / 0.17 -> 0.14 / 0.14 | 9.88 / 10.01 / 10.28 um2 |
| 16x128 | 1 / 2 / 4 | 1.98 / 1.50 / **1.25** | 0.30 / 0.30 -> 0.23 / 0.23 -> 0.17 / 0.17 | 7.75 / 7.81 / 7.95 um2 |
| 16x128 unstrapped | 1 / 4 | 3.42 / 3.18 | 0.41 / 0.15 -> 0.40 / 0.11 | 7.75 / 7.95 um2 |

A strong driver needs the strap: unstrapped, the far column waits on the poly whatever drives it
(4x buys 7%). Together they take the 16x128 macro from 3.42 to 1.25 ns (-63%) for 2.6% area, with
`rdy` tracking the data within 4% and writes next to the driver faster than before straps (0.17 ns
against 0.15 unstrapped, 0.30 strapped at 1x). The decoder gets about 45 ps slower (it drives the
bigger gates). `tests/test_driver_strength.py`: netlist, geometry, the negative driver's output
NMOS, and DRC and pin-level LVS of 4x plain, negative-rail (strapped, with replica) and banked
macros.

Both are on by default from 64 columns (`config.WIDE_ARRAY`): straps, and drivers at 4x. Below
that they are off; `--wl-straps`/`--no-wl-straps` and `--wl-driver-strength N` override either way. `tests/test_straps.py`: track spacing, one strap per word line, DRC and pin-level
LVS of strapped plain, negative-rail (with replica), LVT and banked macros, and a far-column read
that the model makes faster.

## Self-timed reads (the default with the replica; `--no-self-timed`)

A read has a window: the data is valid from about 1.1x the replica's `rdy`, and a stored '0' --
read in subthreshold -- starts to flip if the read word line stays up much longer. From the plain
cell's characterisation (earliest of 50 mismatch dies):

| corner | `rdy` | data valid | earliest '0' flip |
|---|---|---|---|
| tt | 0.96 ns | 1.07 ns | 2.80 ns |
| ss | 1.84 ns | 2.02 ns | 9.83 ns |
| ff | 0.58 ns | 0.68 ns | 1.24 ns |

No single read time fits every corner: ss needs 2 ns, and ff's '0's flip from 1.24 ns. A read
must end where *its own corner's* replica says -- which a clocked controller cannot do (its phases
are whole cycles, and synchronising `rdy` alone costs more than ff's margin).

`ArrayConfig(self_timed_read=True)` -- the default whenever there is a replica column, which is
itself the default; `self_timed_read=False` / `--no-self-timed` turns it off -- makes the macro end its
reads itself: the decoder's read enable becomes `re AND NOT rdy`, so the read word line turns off
one decoder delay after the replica fires; `pch` is still up, so the bit lines -- and `dout` --
hold. `re` can then be held as long as anyone likes.

* **In the decoder** (`netlist/decoder.py::_gates`): a read bus line is an AND of its high address
  bits and `re`, drawn as one NOR over their complements; NOT `rdy` is one more term, whose
  complement is `rdy` itself -- no gate at all while the line still fits a NOR3 (up to one high
  bit). Past that, `re` is gated once, `re_i_n = NAND(re, NOT rdy)`, and every read line takes
  it in place of `re_n`, so the lines stay NOR3. A banked decoder takes both banks' `rdy`
  (`NOT rdy` is their NOR: only the bank being read fires its replica). In a plain half-decoder,
  laid out one strip per logic level, the lone `rdy` inverter joins the next level's strip rather
  than costing a strip of its own.
* **`rdy` to the decoders**: a shared trunk like `re`'s. In a plain macro the trunk line crosses
  the top of the macro, and metal3 over the array is taken (word-line straps, the banked far
  lines), so `rdy` goes round: the line runs on past the right decoder, drops outside the macro's
  right edge on metal3, runs back under its bottom edge and comes up into the replica's track. In a
  banked macro each bank's `rdy` runs on metal3 from its trunk beside the central decoder, above the
  `pch` bridge and below the first row's straps, to its replica column and down through its column
  band.

Measured (extracted 8x20 with replica, `re` held for 10 ns, a written '0' and a '1' in the near
and far columns of the first and last rows): every read correct at tt, ff and ss -- where the same
macro without self-timing lets its '0's flip at 2.5-2.9 ns (tt) and 0.9-1.0 ns (ff). The word line
turns itself off at 1.41-1.53 ns (tt), 0.86-0.99 ns (ff) and 2.70-2.81 ns (ss), after the data
(0.77-0.87, 0.46-0.55, 1.49-1.60 ns). The decoder is a stage deeper on the read path (0.43 ns at tt
for the 8-row decoder, where `re` fed the row gates directly before).

With mismatch (`--characterise --monte-carlo 20`, extracted 16x20 with replica; each die reads a
'0' its own write left and a decayed '1', `re` held for the whole window):

| cell | tt | ss | ff | sf | fs |
|---|---|---|---|---|---|
| plain: '0' / '1' fails | 0 / 0 | 0 / 0 | 0 / 0 | **20** / 0 | 0 / 0 |
| LVT write, `mr_length=0.5`: '0' / '1' fails | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| LVT write, `mr_length=0.5`: access (ns) | 1.63 | 3.13 | 0.99 | 1.84 | 1.62 |
| default cell, in-macro refresh: '0' / '1' fails | 0 / 0 | 0 / 0 | 0 / 0 | **1** / 0 | 0 / 0 |
| default cell, in-macro refresh: access (ns) | 1.75 | 3.48 | 1.04 | 1.82 | 1.92 |
| default cell, loop-back (`--no-refresh-in-macro`): '0' / '1' fails | 0 / 0 | 0 / 0 | 0 / 0 | **18** / 0 | 0 / 0 |

No LVT die's '0' flips at all before its word line turns off. The plain cell's sf failures are its
own (its written '0', 0.88 V, is already above what sf reads as a '0'): at the 1.25x strobe half its
dies failed there, and a read held to the cutoff loses them all -- self-timing does not rescue a
cell whose '0' is out of spec, and the LVT cell (or the negative rail) is what sf needs. The
default cell (a standard-Vt write, a 1.0 um `mr`) is at the edge at sf: its written '0', 0.92 V, is
as high as a column with in-macro refresh still reads as a '0' there, and one die of 20 fails (none
of 20 in either bank of a banked 16x20). A loop-back column reads a '0' at sf only up to 0.86 V, so
with `--no-refresh-in-macro` (or without self-timed reads, which takes the loop-back) the default
cell's '0' fails at sf; the LVT cell, with its lower '0', does not. `gc-compiler` warns when it
builds the default sky130 cell that way (`characterised.loopback_marginal`); IHP SG13G2's default
cell writes its '0' at 0.40 V against a 0.54 V loop-back limit and needs no warning.

Area, with replica (um2/bit): plain 16x20 19.13 -> 20.11 (+5.1%), 32x32 12.86 -> 13.33 (+3.7%);
banked 16x20 15.88 -> 16.31 (+2.7%), 32x32 10.90 -> 11.14 (+2.2%); with the default cell, one bank
of 16x20 21.47 -> 22.25 (+3.6%; the LVT cell's 20.17 -> 21.36, +5.9%) -- mostly the metal3 route round
the macro's edge (0.5 um out: a via2 landing and metal3 spacing), and the banked decoder's extra
inputs. The refresh controller is generated
only for self-timed macros (`GainCellCompiler.refresh_rtl` refuses others): with the read ending
itself, its read phase only has to outlast the slowest corner's cutoff (`t_cutoff` in the
characterisation, plus 20%).

`tests/test_self_timed.py`: `rdy` reaches every read row's line and no write row's, DRC and
pin-level LVS of plain and banked self-timed macros (also with straps, the negative rail and 4x
drivers), and a long-`re` read at tt and ff that keeps its '0's.

## Refresh (`--refresh`)

A stored '0' leaks up, so every row has to be rewritten within the refresh interval. The macro
refreshes a row with **one ordinary access**; the logic that decides when, and which row, is a
generated synchronous controller in Verilog. By default the macro writes the row back itself (its
write driver is `NAND(din, rbl)`: below, after the loop-back); without self-timed reads, or with
`--no-refresh-in-macro`, the data loops through the controller, as described first.

**The access** (`verify/timing.py::simulate_refresh`). `re` and `pch` rise on the row. At the
read strobe (1.25x `rdy`, where a normal read samples) `re` falls but `pch` stays up: the read word
line is off and the precharge still off, so every read bit line -- and `dout` -- holds what it
read. (On a self-timed macro the word line turns itself off there, and `re` only has to outlast
that.) `we` then pulses with `din = NOT dout`, fed back by the controller; the column's write
driver inverts `din` again, so each cell gets its own value back. Last, `pch` falls. Nothing is
stored outside the row. Holding `re` up through the write instead is wrong: a '0' keeps pulling
its bit line down slowly, and once `dout` flips mid-write the loop-back writes a '1' -- a decayed
'0' at 0.85 V came back at 1.8 V that way. (A write-back mux inside the column was the first
plan; it needs five nets where the 1.22 um column has room for three metal2 tracks.)

On the extracted 8x20 macro with replica, both end columns, first and last rows, a '1' decayed to
0.7 vdd and a '0' decayed to 0.75 V (0.8 V for the LVT `mr_length` cell):

| cell, corner | '1' | '0' | read (to the strobe) | whole access |
|---|---|---|---|---|
| plain, tt | -> 1.80 V | -> 0.66 V | 0.74-0.83 ns | 2.7-2.8 ns |
| plain, ss | -> 1.62 V | -> 0.65-0.66 V | 1.26-1.34 ns | 3.3 ns |
| plain, ff | -> 1.98 V | -> 0.66 V | 0.52-0.61 ns | 2.5-2.6 ns |
| plain, fs | -> 1.80 V | -> 0.57-0.58 V | 0.79-0.88 ns | 2.8-2.9 ns |
| LVT `mr_length`, tt | -> 1.80 V | -> 0.47 V | 0.85-0.94 ns | 2.9 ns |
| LVT `mr_length`, ss | -> 1.62 V | -> 0.43 V | 1.51-1.58 ns | 3.5-3.6 ns |
| LVT `mr_length`, sf | -> 1.80 V | -> 0.60 V | 0.85-0.95 ns | 2.9-3.0 ns |

(The write-back pulse is 2 ns here; the whole access is that plus the read.) A refresh can only
restore what a read would read right: a '0' whose bit line is already past vdd/2 at the strobe
is frozen near the read buffer's threshold, and an in-between level goes back
(`RefreshTiming.read_ok` flags it). That is the same limit as for a read -- the refresh interval is
what keeps every '0' inside it. `access_energy` now measures `refresh_j` (one refresh of a row of
decayed '1's, the worst row), and `refresh_power` uses it.

**The controller** (`rtl.py`; `gc-compiler ... --refresh [--clock-ns T] [--refresh-interval US]`,
or `GainCellCompiler.refresh_rtl`) writes four files next to the GDS:

* `<name>_ctrl.v`: user side `clk rst_n req we addr din -> dout dout_valid busy` (a request is
  taken on a clock edge where `busy` is low, and its `we`, `addr` and `din` must hold until `busy`
  is low again -- the controller keeps no copy, they drive the macro directly; a read's data comes
  with `dout_valid` and holds until the next read); macro side
  `m_re m_we m_pch m_addr m_din m_dout` (per bank for a banked macro), and `m_rdy` with
  `--handshake` (below). One state machine runs user
  reads and writes and refreshes; a timer requests the next row every `REFRESH_CYCLES` so every
  row is refreshed within half the interval, and a pending refresh takes the next idle cycle. It
  hides the macro's inversion (`dout = NOT din`): user data reads back as written. Every phase is
  a whole number of cycles from the macro's timing -- with `--characterise`, the slowest corner's
  read cutoff, write and precharge; otherwise conservative defaults (stated in the file's header).
  It needs a self-timed macro (the default, see above): a clocked read of fixed length cannot
  be right at every corner. The interval defaults to the one measured for the cell, worst die at
  ss/125 C (`rtl.default_refresh_interval_us`): 154 us for the default cell (a standard-Vt write
  device, a 1.0 um `mr`), 614 us for the LVT write with a 0.5 um `mr` and 571 us with both read
  gates at 0.5 um (both measured idle: their '0's do not survive read or write traffic in their
  column, see [The write device's leakage](#the-write-devices-leakage-and-the-default-cell)), 88 us
  for the plain cell with a -0.3 V rail, and the shortest measured, 45 us, for any other cell.
* `<name>_model.v`: a behavioural model of the macro (its pins, no supplies) that keeps
  retention: a row not rewritten within `RETENTION_NS` reads `x`.
* `<name>_top.v`: the controller wired to the macro.
* `<name>_tb.v`: self-checking -- every row written, an interval of random reads and writes
  against a golden copy, then two intervals with no user access, then every row read back.
* `<name>_ctrl.sdc`: the controller's timing constraints (below).

`tests/test_refresh.py`: controller parameters, the Verilog against the macro's pins (plain and
banked), the testbench passing under iverilog -- and failing with the refresh timer disabled, so
the bench really tests refresh -- the handshake against a slow and a stuck replica, the SPICE
refresh access at tt and ss, and the CLI.

**The macro writes the refresh back (`refresh_in_macro`, the default with self-timed reads;
`--no-refresh-in-macro` / `ArrayConfig(refresh_in_macro=False)` keep the loop-back above).** Each column's write driver becomes `wbl = NAND(din, rbl)` instead of an inverter (a
custom slice in the column: rbl's PMOS 2x, both NMOS 2x). A refresh drives every `din` high, so
each column writes back the bit its own read bit line holds since the read; nothing loops through
the controller, so there is no `dout -> din` path to time, and `pch` falls a cycle after `we`.
Outside a user write `din` is high (a flop opens the user's data from the write's start to the end
of its precharge, steady as `we` falls), so the write bit lines idle low. They were held high
instead for a while -- `din` low, to save read energy: with `din` high, each column that reads a '1'
raises its write bit line, `NAND(1, rbl)`, and lowers it again with the precharge, which doubles a
read's energy -- but a write bit line left high leaks into every '0' of its column through the write
devices, and on the extracted macro that lost every '0' within 2 us of idling (see
[The write device's leakage](#the-write-devices-leakage-and-the-default-cell)). The read energy is
the price of in-macro refresh now (the loop-back's write bit lines do not move in a read). A
refresh raises `din` from its start: raised only once the read had ended, the write bit line's fall
coupled back through the NAND into the then floating read bit line and turned a marginal '0' (0.55 V
on SG13G2 at tt) into a '1'.
`measurements/scripts/compare_refresh_in_macro.py`, extracted 16x20 macro (energy at tt, every
column kept; refresh under 20 mismatch dies at ss):

| | sky130 loop-back -> in-macro | SG13G2 loop-back -> in-macro |
|---|---|---|
| macro area per bit, 16x20 / 8x8 banked | +5.4% / +8.9% | +4.4% / +7.4% |
| read access tt / ss | 1.43 -> 1.52 / 2.76 -> 2.96 ns | 2.26 -> 2.33 / 3.87 -> 3.98 ns |
| precharge tt | 0.37 -> 0.64 ns | 0.35 -> 0.40 ns |
| read energy (all '1's), tt | 1.90 -> 4.33 pJ | 1.26 -> 2.72 pJ (1.37 with `din` low) |
| refresh energy, tt | 4.11 -> 4.38 pJ | 3.10 -> 2.89 pJ |
| refresh at every corner; mismatch dies failing at ss | yes; 0 of 20 | yes; 0 of 20 |

It is the default: no data path leaves the macro in a refresh (nothing to time after routing,
nothing a slow corner can miss), for 4-9% of macro area and about twice a read's energy (with `din`
high outside writes, as the write bit lines have to idle, each read of a '1' swings its write bit
line). The sky130 column is the default cell's, with `din` driven as the controller drives it; it
precharges slower than the LVT cell's did with `din` low (0.37 -> 0.64 ns at tt, against 0.35 ->
0.45). Without
self-timed reads (`ArrayConfig.basic`) the loop-back remains, and so does every reference layout
the regression test pins. (The controller-area measurements below were made with the loop-back and
the LVT cell, the default then.)

**At most 32 rows a bank on sky130.** A read bit line runs beside its own write bit line, on
metal2, the whole height of the column: on 2 x 128x128 it is 43 fF of the read bit line's 84 fF.
In a refresh the write bit line follows what the read senses (`NAND(1, rbl)`), and its swing
couples the floating read bit line back toward the NAND's threshold. A tall column's decayed '1'
leaves its bit line only just past it when the replica ends the read (0.81 V at 128 rows), so the
NAND drives the write bit line only part way, and the '1' is written back below where it was. A '1'
decayed to 0.7 vdd, refreshed, on the extracted macros (bank 0, nominal devices):

| macro | tt (1.26 V ->) | ss, 125 C (1.13 V ->) |
|---|---|---|
| 2 x 32x32 | 1.80 V | 1.56 V (2 ns write), 1.60 V (6 ns) |
| 2 x 64x64 | 1.63 V (1.65 V however long the write) | 1.19 V (2 ns), 1.27 V (6 or 10 ns) |
| 2 x 128x128 | 1.06 V | -- (its benches outlast the 30-minute limit) |

(Restored is 0.95 vdd.) So `characterised.refresh_max_rows` is 32 on sky130: `gc-compiler` warns
above it, and `memory()` / `--controller`, whose controller would refresh the macro, refuse.
Timing the write-back differently only moves the loss: `din` held low through the read (every
write bit line high and still) restores the '1' at 128 rows (1.77 V at tt), but a '0' column's
write bit line then falls after the read and turns a marginal '0' into a '1', as above. A keeper on
the read bit line (an NMOS to vss gated by `dout`) restores the '1' too, and on a short SG13G2
column turns '0's at 0.55 V into '1's at four corners of five. Taller banks need the two bit lines
apart, or shielded, in the cell.

**At most 16 rows a bank on SG13G2**, for the same reason. A '1' decayed to 0.7 vdd, refreshed,
comes back to 1.06 V on a 2 x 16x20 macro at ss, 125 C (1.03 V needed), but only to 1.138 V on
2 x 32x32 already at tt (1.14 V needed), 1.11 V on 2 x 64x64 and 0.78 V on 2 x 128x128; at ss
the 128-row macro misreads it. The level is the one the access leaves the cell at, after the write
word line closes: it trails `we` through the decoder (1.4 ns at ss), so the node is still rising
when `we` falls.

## The controller as hardware: one hard block (`--controller`)

`gc-compiler ... --controller` (or `GainCellCompiler.memory()`, `flow/memory.py`) builds the whole
memory as one hard block, `<name>_top`: the decoded macro with its refresh controller synthesized,
placed and routed under it by [LibreLane](https://github.com/librelane/librelane) (in the nix shell,
from nixpkgs, with yosys, OpenROAD, magic, netgen and KLayout). What it takes:

* **The macro as a hard macro** (`flow/abstract.py`). Its supplies are rails on metal1/metal2 under
  three layers of wiring, so `array/power.py` adds metal4 stripes and ties each down to its own
  rails with via stacks where there is room -- computed from the layout, per stack shape: the base
  pad inside the rail, every pad and cut above clear of what is drawn there. Each bank's n-well
  gets its own stripe. Then a LEF abstract (each signal pin its whole metal polygon, the supplies
  the stripes, every other metal1-metal3 shape an obstruction), a timing library per corner (a read
  as an arc from `re` to every `dout`), and Verilog blackboxes (unpowered, and powered for LVS).
  The macro is moved to the origin, with a placement boundary and its pins declared in its own top
  cell, so magic extracts it with the right ports.
* **The flow** (`flow/librelane.py`): the macro fixed at the top of the die, a band of standard
  cells under it (where its data pins are) sized for the controller at 90% of what placement can use
  (below), the user's pins along the bottom edge, and the core's metal5 straps crossing the macro every 20 um onto its
  stripes. LibreLane synthesizes, places, builds the clock tree, routes -- into the macro for the
  address and enable pins -- and signs off: KLayout DRC and magic DRC (both, macro included),
  netgen LVS of the extracted block,
  and multi-corner static timing, slew and load with the SDC below.
* **The constraints** (`rtl.controller_sdc`, `rtl.top_sdc`) follow the access protocol: a read is
  sampled `READ_CYCLES` edges after `re` rises, and the macro's outputs move no sooner than
  `macro_min_ns` after the controller's (hold). The refresh loop-back (`--no-refresh-in-macro`) -- `dout` through the
  controller back to `din`, before `we` rises one edge later -- ends on pins with no timing check
  (a user write changes `din` as `we` rises, harmlessly), so no constraint can hold it; the flow
  measures it on the routed block instead (`flow/sta.loopback_arrival`, OpenSTA with the extracted
  parasitics): at 2 x 16x20 it has 13.8 / 12.1 / 14.4 ns (tt/ss/ff) to spare in its 19.75 ns. `READ_CYCLES` now budgets the controller's own sampling (`sample_ns`, 1.5 ns) on top of
  the macro's read; with only the read, a 5 ns clock missed setup at the slow corner. `rst_n` goes
  through a two-flop synchroniser, and `busy` covers the reset.
* **The check at gate level**: the generated testbench is run on the routed netlist (the macro as
  its behavioural model; `flow/gatesim.py`). `build_memory` raises unless the block is DRC, LVS and
  antenna clean, meets setup and hold at tt, ss and ff, and passes.

Results, 10 ns clock, the default configuration (two banks, the default cell, self-timed reads, in-macro
refresh, the latch controller below). Setup is met at every corner -- the worst slack STA shows is 0,
the `dout` latches borrowing time inside their window -- and slew and load are within limits:

| memory | bits | block | macro alone | controller (flops + latches) | hold slack tt / ss / ff | power* |
|---|---|---|---|---|---|---|
| 2 x 16x20 | 640 | 21,416 um2 (33.5 um2/bit) | 12,362 um2 (19.3) | 2,062 um2, 207 cells, 47 | 0.26 / 0.64 / 0.13 ns | 0.53 mW |
| 2 x 32x32 | 2048 | 36,969 um2 (18.1 um2/bit) | 25,641 um2 (12.5) | 2,525 um2, 259 cells, 59 | 0.25 / 0.62 / 0.13 ns | 0.66 mW |
| 2 x 128x128 (`--read-ns 12`; above the row cap, now refused) | 32768 | 303,979 um2 (9.28 um2/bit) | 226,477 um2 (6.91) | 7,913 um2, 1,000 cells, 156 | 0.25 / 0.63 / 0.13 ns | 3.3 mW |

**A smaller controller** (`measurements/scripts/sweep_controller_area.py`; before it, the 2 x 16x20
and 2 x 32x32 controllers were 3,774 and 5,286 um2, their blocks 34.6 and 19.1 um2/bit, with the LVT cell). The controller was mostly flops, two a column: `wdata`, a copy of the user's write data, and
`dout`, the read data, each with a hold multiplexer, besides the macro address `m_addr`. Now:

* **No copy of the user's request.** `we`, `addr` and `din` hold from the edge that takes a request
  until `busy` falls, and drive the macro directly (`m_addr = refreshing ? ref_row : addr`,
  `m_din = ~din`). A refresh first spends a cycle with its row on the address (`S_ARM`), so the
  address never moves as `re` rises, and the refresh row advances only once `re` and `we` are down.
* **`dout` is a latch**, open while `clk` is low in a user read's last cycle and closing on the edge
  that ends the read -- where the flop took the data, so the macro-side setup and hold are the
  same -- and holding until the next read. Its enable is `!clk && dout_open_q`, a flop set by the
  edge that starts the read's last cycle (with `--handshake`, from the synchroniser's view of `rdy`
  an edge ahead): it cannot glitch open, and its clock-gating check has half a cycle, which a
  decoded enable (state, count, the bank's `rdy`) missed at ss. The constraints take `dout` an edge after the latch closes (a 2-cycle multicycle from
  the latches), as the user does, with `dout_valid`; STA reports the latches borrowing time
  (slack 0) within their window.
* **The flow** (the library's `flow_config`): a 4x clock-tree root buffer instead of the default
  16x, which sat at every tree root of a 50-flop design, and on sky130 the drive-1 cells its PDK
  keeps from synthesis. Clock gating, area-optimising synthesis (AREA 1-3) and reset-less data
  flops were measured on the flop controller too; with no wide data register left, none applies.

What is left: 22-23 flops of control (refresh timer and row, state, phase, strobes, reset
synchroniser) and a latch a column. Routed, 10 ns clock, all DRC/LVS clean and timed at tt/ss/ff
(sky130 with the LVT cell, the default then: the dies carry its smaller array):

| block | sky130 controller | sky130 die | SG13G2 controller | SG13G2 die |
|---|---|---|---|---|
| 8x8 one bank | 2364 -> 1560 um2 (-34%) | 10545 -> 9074 um2 (-14%) | 4204 -> 2836 um2 (-33%) | 14134 -> 12066 um2 (-15%) |
| 8x8 banked | 2494 -> 1687 um2 (-32%) | 14999 -> 12972 um2 (-14%) | 4443 -> 2986 um2 (-33%) | 19370 -> 15952 um2 (-18%) |
| 8x64 banked | 6087* -> 4423 um2 (-27%) | 33621* -> 31035 um2 (-8%) | 12503* -> 7361 um2 (-41%) | 45650* -> 37014 um2 (-19%) |

(* against the flop controller with clock gating and the flow settings, the best it got.) The
flow settings alone are worth 7-21% of the latch controller (PDK defaults: 1971 / 2078 / 4736 and
3233 / 3456 / 8301 um2).

**The band** (`flow/librelane.memory_config`) is sized as the area placement can use -- between
the core's bottom margin and the macro's halo (`BOTTOM_MARGIN_ROWS`, `MACRO_HALO`, both set
explicitly) -- at 90% of it. Sized as the whole gap, the margin and halo had silently taken ~9 um:
harmless under an 8-column macro's tall band, half of a wide one's, which overfilled and was
rebuilt taller at random. A banked macro's banks sit either side of its decoder, ~190 um apart at
64 columns, and every column's `din` and `dout` span both: long wires across the band, whose routes
come out longer than placement guessed. So the band is made tall enough for those nets from the
start (`StdCellLibrary.crossing_nets_per_um`: 2.5 a um on sky130, 2.2 on SG13G2, measured on routed
8x64 blocks), nets over 300 um get repeaters, slew and load are repaired with a 50% margin and again
on the global route's parasitics, antennas with a 30% margin and six diode passes in the detailed
router, and the band is rebuilt taller (all of it, cells and crossing height) on a congested global
route, a router that leaves a violation, or a block that misses only slew, load or antenna limits;
each attempt keeps its run and why it was retried (`metrics["attempts"]`). Pins the block leaves
unconnected (the replica's `rdy` without the handshake) are obstructions in the macro's LEF: the
router ran a wire between the macro's pins and over an unconnected one. A block counts
as clean only within its slew and load limits too (`FlowRun.clean`): the earlier wide sky130 runs
met setup and hold with 63 load and 78 slew violations at ss. The user's pins sit on the bottom
edge, control first, then each column's `din` and `dout` under its column (`user_pin_order`; the
placer had put them on the side edges).

**A large macro needs a longer read.** Without `--characterise` the controller assumes a 4 ns read,
enough for the small macros above but not for 2 x 128x128, whose reads end at about 7 ns already
at tt (built before the 32-row cap above, which now refuses it). `--read-ns` (or `memory(timing=ControllerTiming(read_ns=...))`) sets a bound; a generous one is
safe, since the macro ends each read itself and holding `re` longer only adds latency -- the
128x128 block above uses 12 ns (the tt cutoff times the slow/typical ratio measured at 16x20). Its
band is set by the 256 `din`/`dout` nets that cross between the banks, not by its cells (hence a
larger die than the flop controller's 277,947 um2, whose band was then too short to route clean
by today's checks); the first height congested and the second, 1.25x, signed off clean (56 minutes
in all, `metrics["attempts"]`). Before, the controller was 16,445 um2 with 292 flops, 7.1 mW. And
the macro's LEF gives each signal pin its antenna area (a lower bound on the gate an input drives,
the diffusion that drives an output): without it the antenna check took the long `dout` wires to
the controller for unprotected (5 violations at 128x128), and the wires into the macro went
unchecked.

**Or let the replica end the read (`--handshake`).** The macro already knows when its read is
done -- the replica's `rdy` -- so with `--handshake` (`ControllerTiming(handshake=True)`, or
`memory(handshake=True)`) the controller watches it instead of counting to a fixed `read_ns`.
`rdy` is asynchronous to the clock, so each bank's goes through a two-flop synchroniser (and a
third flop to see it rise), and the data is taken two edges after the edge that catches it. The
read then lasts what that die, corner and bank need, however large the macro, and there is no
read time to get right; `read_ns` only feeds the macro's timing library. What it still needs:
`dout` must trail `rdy` by less than two cycles less the controller's own sampling
(`rdy_lag_ns`, 1 ns by default, from the characterisation with `--characterise`: the replica
tracks the data within a few percent once the word lines are strapped), which the parameters
check. A read whose `rdy` never comes ends after `timeout_ns` (50 ns) and sets a sticky
`read_timeout` output, so a broken replica cannot stop refresh. Since `rdy` only falls when `pch`
does, a stale `rdy` from the last read never ends the next one: the controller waits for a rise.
The price is latency: a read takes at least three cycles (catch, synchronise, take), where a
fixed read of a small macro at 10 ns takes one. In the static timing the macro's `rdy` is a false
path into the synchroniser, `m_dout` is timed from the edge that catches `rdy`, and the refresh
loop-back gets the three edges from that edge to `we`.
At 2 x 16x20, 10 ns clock, both blocks DRC, LVS and antenna clean and passing at gate level
(measured with the earlier controller -- flop data registers, the loop-back refresh):

| read | block | controller | setup slack tt / ss / ff | loop-back slack tt / ss / ff | power* |
|---|---|---|---|---|---|
| fixed (4 ns) | 22,128 um2 | 3,761 um2, 275 cells, 71 flops | 4.21 / 2.70 / 4.74 ns | 13.8 / 12.1 / 14.4 ns | 1.2 mW |
| `--handshake` | 22,619 um2 (+2.2%) | 4,493 um2, 321 cells, 82 flops | 5.74 / 3.94 / 6.41 ns | 26.8 / 25.0 / 27.4 ns | 1.5 mW |

The synchronisers and the wider timeout counter cost 11 flops and 730 um2 of cells; the setup
slack grows, since no path now has to fit the macro's read into a fixed count of cycles.
The testbench proves the point: its macro model raises `rdy` 25 ns after `re` (2.5 cycles, far
beyond any `read_ns`) and the handshake controller keeps every row, while the fixed-time
controller against the same model reads wrong; with a replica that never fires, the reads time
out and the bench sees `read_timeout`.

With `--characterise` (`gc-compiler --rows 16 --cols 20 --controller --characterise --corners
tt,ss,ff`), both banks are characterised first and the measurements set the block's timing instead
of the conservative defaults: the controller holds a read for the slowest corner's self-timed cutoff
plus 20% (6.18 ns: ss cuts off at 5.15 ns), writes for three times the slowest write (1.46 ns) and
precharges for 1.26 ns, and the macro's timing library carries each corner's own read (2.66 / 5.15 /
1.59 ns at tt / ss / ff, against 4 ns everywhere by default). The block is the same size, DRC and
LVS clean and passes its testbench; its setup slack grows from 0 (the `dout` latches borrowing) to
1.29 ns at tt and 2.63 ns at ff, and stays 0 at ss.

\* LibreLane's estimate with its default switching activity; most of it is the clock: the
controller clocks its ~50-60 flops and latches every cycle to refresh one row every few microseconds.

What this says:

* **The controller is a fifth of a small macro.** With no copy of the user's request and `dout` in
  latches, it is two dozen control flops and a latch a column (it was two flops a column, as
  large as a small macro). It grows with the columns, not the rows, so its share falls as the
  macro grows.
* **The clock is the power.** A divided or gated refresh clock -- the controller only has to act
  once per row interval -- is the next thing to do if this block's power matters.
* **5 ns does not close at the slow corner** (the controller alone: -0.05 ns plain, -0.40 ns
  banked, at ss 1.6 V/100 C); 10 ns closes with 2.7 ns to spare. `--clock-ns` sets it, and the
  flow says where it stops.

`tests/test_memory_block.py`: the views list exactly the netlist's pins, the power stripes keep
the macro DRC clean and on their own nets (LVS), the constraints, and -- with LibreLane -- a
one-bank and a banked block built, clean and passing at gate level, and a banked one with the
handshake.

## Replica bit-line and timing characterisation

**The replica column.** `ArrayConfig(replica_column=True)` (the default; `--no-replica` drops it) adds one more bit-line
column after the last data column. It is not counted in `cols`/`bits`, but its cells sit on
every row's word lines, so a read through it sees the same decoder, driver and word-line delay
as any other cell. What differs is the cell (`gain_cell_3t(replica=True)`): its storage node is
tied to the write bit line (a metal2 stub over the node's li strip), and the column's write
driver has its input tied to vss, so `wbl_rep` -- and with it every replica storage node -- is
held at vdd. The replica read bit line therefore discharges on *every* read, through the same
read pair, precharge device and read buffer as a data column; the buffer's output is the macro's
new **`rdy`** pin, which rises when the replica bit line has discharged. The netlist has a
matching `gc3t_rep` subcircuit (`mw` shorted, `mr` gated by `wbl`), and macro LVS covers it.
Area cost: 1.3% for 16x20, 1.0% for 32x32, 0.7% for 64x64.

**Timing characterisation.** `gc-compiler --characterise` (or
`GainCellCompiler.characterise(result)`) runs `verify/timing.py`:

1. **Extract**: magic extracts the macro GDS with parasitic capacitance into one flat SPICE
   netlist, its nets named after the layout's labels (`verify/parasitics.py`).
2. **Reduce** (`verify/flatnet.py`): the decoders, drivers and column circuits are kept, and so are
   the cells of both end columns and the replica; every other column becomes a multiplied gate
   load on the word lines (all it does to an access elsewhere). A word line is poly all the way
   across the array, so each becomes a distributed RC ladder along the columns -- extraction
   gives its capacitance, the resistance is analytic (`Technology.poly_sheet_resistance`) -- driven
   from the end its driver is at.
3. **Simulate** with ngspice at tt, ss (0.9 vdd, 125 C), ff (1.1 vdd, -40 C), sf and fs: a write
   of a '1' and a '0' (the levels they settle at), then reads of the far and near cell of the
   first and last rows (drivers on opposite sides), from a fresh '1', a decayed '1' (70% of
   the level) and a '0'. Times are measured from the `re`/`we` edge to the decoder output, the
   word line at the cell, the bit line crossing vdd/2, `dout` and `rdy`.

> **Numbers corrected (Sep 2026).** Earlier versions of this section were measured on benches
> that started ngspice with `uic`, which silently dropped the storage-node initial conditions.
> Every number below is from the
> corrected benches, as regenerated by `measurements/scripts/` into `measurements/data/`.

Result for a 16x20 decoded macro with replica (`measurements/data/characterise.json`; the read tolerance
is the highest '0' that still reads, bisected on the extracted macro independently of the write;
refresh from `measurements/data/retention.json`):

| corner | '0' written (V) | read tolerance (V) | margin (V) | '0' reads | access (ns) | write (ns) | refresh (us) |
|---|---|---|---|---|---|---|---|
| tt_1.8V_25C | 0.69 | 1.01 | +0.32 | ok | 1.03 | 0.14 | 443 |
| ss_1.62V_125C | 0.63 | 1.00 | +0.37 | ok | 1.88 | 0.24 | 58 |
| ff_1.98V_-40C | 0.70 | 0.99 | +0.29 | ok | 0.66 | 0.10 | 317 |
| sf_1.8V_25C | 0.88 | 0.86 | -0.02 | **FAIL** | 1.15 | 0.19 | 0 |
| fs_1.8V_25C | 0.48 | 1.12 | +0.65 | ok | 1.02 | 0.11 | 837 |

What this says:

* **`rdy` tracks the data.** The data of the slowest legitimate read of a '1' is valid at 1.08 to
  1.16x the replica's `rdy` time; `Characterisation.recommended_margin()` gives x1.15 here.
  The replica shares the rows' word lines, so unlike a fixed delay chain it sees the decoder,
  driver and word-line RC of the actual array.
* **A written '0' is marginal, not broken.** A PMOS write device only pulls the storage node
  down to about |Vtp|, so a '0' settles at 0.5 to 0.9 V. The read device tolerates about 1.0 V
  in four corners, so the '0' reads there with 0.3 to 0.65 V to spare -- but at sf (slow PMOS
  writes the highest '0', fast NMOS tolerates the lowest) it misses by 20 mV, and half the
  mismatch dies misread (below). The margin that is left elsewhere is what retention spends:
  the plain cell's refresh interval is the shortest of any option (58 us at ss). Two fixes
  follow: the negative-rail driver and the LVT write device.
* The reduction is accurate to a few percent and on the pessimistic side (`measurements/scripts/sweep_reduction.py`
  re-measures it).

Running ngspice on the whole `sky130.lib.spice` costs ~40 s per process to parse, mostly the
resistor/capacitor corners and the several-thousand-bin special-cell library that a bench like
this never uses. `Technology.model_header()` loads only the MOSFET/varactor corner file plus the
typical resistor parameters it refers to, which gives bit-identical waveforms in 0.2 s; five
corners of a 16x20 macro take about 3 minutes on 16 cores.

### Mismatch and Monte Carlo

The corner sweep says how slow the slowest die is; it says nothing about how far two
neighbouring transistors of one die differ, and that difference is what limits a self-timed
read: `rdy` comes from the replica's read pair and the data from a real cell's, and if one is a
few millivolts weaker than the other the strobe is early or late. `--monte-carlo N` (with
`--characterise`; `GainCellCompiler.characterise(..., monte_carlo=N)`; `verify/montecarlo.py`)
simulates **N dies** at each corner. A die is a seed: `.option seed` plus the PDK's
`mc_mm_switch` give every transistor instance in the reduced macro -- decoder gates, drivers,
column circuits, the kept cells, the replica -- its own independent Gaussian offsets in `vth0`,
`voff` and `toxe`, with the Pelgrom `1/sqrt(W*L)` sigma (about 13 mV of `vth0` for a minimum
nfet). The same seed is the same silicon in every simulation and at every corner, and the same
seed twice is bit-identical.

Each die is two ngspice runs of row 1: one access reads a decayed '1' (70% of vdd) at column 0,
the far end of the word line, and a '0' at the last column, starting from the level *that die's own
write* left; one write stores a '1' and a '0'. It records the tracking ratio `t_dout / t_rdy`,
whether the '0' survives until the strobe (1.25x `rdy`), the level a written '0' settles at, and the
write time. 50 dies at each of the five corners of the 16x20 macro with replica:

| corner | access (ns) | dout/rdy | '0' written (V) | '1' / '0' fails |
|---|---|---|---|---|
| tt_1.8V_25C | 1.07 ± 0.02 | 1.11 ± 0.02 | 0.68 ± 0.03 | 0/50, 0/50 |
| ss_1.62V_125C | 2.02 ± 0.05 | 1.10 ± 0.02 | 0.62 ± 0.03 | 0/50, 0/50 |
| ff_1.98V_-40C | 0.68 ± 0.01 | 1.16 ± 0.01 | 0.70 ± 0.03 | 0/50, 0/50 |
| sf_1.8V_25C | 1.16 ± 0.03 | 1.08 ± 0.01 | 0.88 ± 0.03 | 0/50, 25/50 |
| fs_1.8V_25C | 1.11 ± 0.02 | 1.15 ± 0.02 | 0.47 ± 0.03 | 0/50, 0/50 |

* **Mismatch barely moves the read**: `access` and `dout/rdy` vary by 1 to 3% from die to die,
  because they are the same path built of the same devices. The 3σ strobe is x1.22
  (`Characterisation.statistical_margin()`).
* **A '0' is where mismatch bites.** The written level spreads by about 33 mV (1σ); at sf, where
  the nominal '0' already sits 20 mV past the tolerance, 25 of 50 dies misread.

The limits of this: mismatch is **local** only -- global process variation is what the corner
sweep is for (the library's `mc_pr_switch` is not referenced by any model in this PDK release, so
there is nothing to switch on); only devices that are real in the reduced netlist are varied
(the gate loads standing in for the other columns are not, and only load the word line); 0 failures
in 50 dies only bounds the failure rate below about 7% (`verify.montecarlo.upper_bound`), and the
sigma figures are a Gaussian extrapolation from a 50-die histogram (each sigma good to about 10%),
so the `3σ` columns are estimates, not measurements. Cost: ~3 s per die on 8 parallel ngspice
runs, most of it setting up ~700 hierarchical device instances rather than simulating, and
adding more parallel runs does not help on this machine -- 50 dies at five corners took 14 minutes.

## Negative-rail write driver (`--neg-wl`)

The finding above -- a PMOS write with its word line at vss leaves a '0' at 0.5 to 0.9 V, which
misreads at sf and leaves little retention elsewhere -- has one fix that keeps the cell: drive
the write word line *below* vss.
`ArrayConfig(negative_write_wl=True)` / `--neg-wl` replaces each `wwl` driver with one that swings
between `vdd` and a new **`vneg`** supply (`ArrayConfig.write_wl_low`, default -0.3 V), and the macro
gains a `vneg` pin. The read path and the rest of the macro are untouched.

**The driver** (`netlist/spice.py::neg_driver_subckt`, layout in `periphery/wl_driver_neg.py`) is eight
devices where the plain driver had two. A plain inverter with its NMOS source at `vneg` does not work:
its input low is 0 V, so that NMOS sees `Vgs = |vneg|` and is never off. So a level shifter moves
the input into the `vdd`/`vneg` domain first, then an inverter in that domain drives the word line:

```
inb = not in              pa: PMOS  inb -> x        nx: NMOS  x gated by z   } latch, source vneg,
                          pb: PMOS  in  -> z        nz: NMOS  z gated by x   } body vneg (isolated)
x follows in, z opposes   pd/nd: inverter, input x, output wwl (between vdd and vneg)
```

The first level shifter I tried had the *input* devices as NMOS at `vneg` and burned 2 to 30 uA per
row at rest (an input at 0 V is not off against a source at -0.7 V). Putting the inputs on PMOS at
`vdd` (fully off at either rail) and the latch on NMOS fixes that -- the whole 16x20 macro now idles at
under 3 uA on `vneg` -- but makes it ratioed: the PMOS have to overpower a latch with 2.5 V of gate
drive, so the latch is drawn long (0.42 x 0.8 um) and the inputs wide (1.5 um). That pair is the smallest
that flips in every corner and at 125 C (worst 260 ps for `in` to the word line), checked from `vneg` =
-0.2 to -1.0 V.

**The isolated well.** The NMOS that pull to `vneg` need a body at `vneg`: with a source at -0.8 V and a body on the 0 V
substrate the source-body junction would be forward biased. sky130 provides an isolated p-well only inside
a **deep n-well**: an n-well ring 1.43 um wide (1.03 um inside the deep n-well's edge, 0.4 outside)
around the p-type interior, whose substrate tap is the `vneg` rail. Magic extracts that node as the NMOS
body, so LVS compares it (`test_slice_lvs_matches_and_sees_the_isolated_body` shows an output NMOS with
its body on vss fails LVS). Two rules matter that the KLayout deck we sign off with does not have,
both found by running magic's own deck (now a required second sign-off: see "Two DRC decks" below):

* `dnwell.5`: a deep n-well edge may not cut a P+ diffusion. The deep n-well's inner margin puts its edge
  inside the level shifter's PMOS well, so the PMOS end left of it and the ring's wall is wider.
* `nwell.7`: an n-well not overlapping the deep n-well stays 4.5 um from it. The array's n-wells would
  have been 1.8 um away, so each strip grew 3.4 um (the word-line landing is moved to the array side so
  the poly stubs stay short).

Both are now checked geometrically in `tests/test_negative_wl.py`, along with ring closure and
`nwell.6`. The wells are continuous down the whole column, bridged across the tap-row gaps (which
would otherwise split the deep n-well into islands closer than its 6.3 um spacing), and closed by
n-well caps at both ends; the two strips' `vneg` rails are joined on metal3 over the top of the array,
where the `vneg` pin is. The slices keep the plain driver's contract (poly stubs on the right,
`<line>_in` on metal2 on the left), so the decoders needed no change beyond finding the inputs where
they now are.

> **Numbers corrected (Sep 2026).** Earlier versions of this section were measured on benches
> that started ngspice with `uic`, which silently dropped the storage-node initial conditions.
> Every number below is from the
> corrected benches, as regenerated by `measurements/scripts/` into `measurements/data/`.

**Result** (extracted, 16x20 decoded macro with replica, `vneg` = -0.8 V, the former default; DRC clean, LVS match):

| corner | '0' written (V) | read tolerance (V) | margin (V) | '0' reads | access (ns) | write (ns) | refresh (us) |
|---|---|---|---|---|---|---|---|
| tt_1.8V_25C | 0.01 | 1.00 | +0.98 | ok | 1.06 | 0.12 | 869 |
| ss_1.62V_125C | 0.00 | 0.99 | +0.98 | ok | 1.92 | 0.17 | 122 |
| ff_1.98V_-40C | 0.02 | 0.99 | +0.97 | ok | 0.68 | 0.09 | 639 |
| sf_1.8V_25C | 0.20 | 0.86 | +0.66 | ok | 1.18 | 0.16 | 609 |
| fs_1.8V_25C | 0.00 | 1.12 | +1.12 | ok | 1.04 | 0.09 | 1007 |

With mismatch (50 dies per corner):

| corner | access (ns) | dout/rdy | '0' written (V) | '1' / '0' fails |
|---|---|---|---|---|
| tt_1.8V_25C | 1.10 ± 0.02 | 1.12 ± 0.02 | 0.02 ± 0.02 | 0/50, 0/50 |
| ss_1.62V_125C | 2.07 ± 0.04 | 1.10 ± 0.02 | 0.01 ± 0.01 | 0/50, 0/50 |
| ff_1.98V_-40C | 0.69 ± 0.01 | 1.17 ± 0.02 | 0.02 ± 0.02 | 0/50, 0/50 |
| sf_1.8V_25C | 1.19 ± 0.03 | 1.08 ± 0.01 | 0.21 ± 0.03 | 0/50, 0/50 |
| fs_1.8V_25C | 1.14 ± 0.02 | 1.16 ± 0.02 | 0.00 ± 0.00 | 0/50, 0/50 |

Every die reads its own '0' back in every corner. Reads are about 3% slower than the plain
macro's (the strip is wider, so every read word line's route is longer). **Choosing `vneg`:** the
former -0.8 V default was picked on the uncorrected benches, where the plain '0' failed in four corners.
On the corrected ones the plain '0' misses only at sf and only by 20 mV, and
`measurements/scripts/sweep_vneg.py` (`measurements/data/vneg.json`) shows how little is needed:

| vneg | sf '0' (V) | sf margin (V) | sf MC fails | worst refresh (us, ss) | write oxide stress (V) |
|---|---|---|---|---|---|
| -0.2 V | 0.71 | +0.14 | 0/20 | 77 | 2.18 |
| -0.3 V | 0.63 | +0.23 | 0/20 | 88 | 2.28 |
| -0.4 V | 0.54 | +0.31 | 0/20 | 97 | 2.38 |
| -0.5 V | 0.46 | +0.40 | 0/20 | 107 | 2.48 |
| -0.6 V | 0.37 | +0.48 | 0/20 | 115 | 2.58 |
| -0.8 V (former default) | 0.20 | +0.66 | 0/50 | 122 | 2.78 |

The read is safe from -0.2 V; retention grows only 60% from -0.2 to -0.8 V, because it is set by
the plain cell's write device's leakage and its drain junction, not by the rail. So the default is now
-0.3 V; `write_wl_low` / `--vneg` sets another level (the layout does not depend on it).

**What it costs, and what it does not solve.**

* **Area.** The strip goes from 5.4 to 19.5 um per side, a fixed 28.1 um of macro width: +30% for 16x20
  (94.7 to 122.8 um, 16% to 12% array efficiency), +23% for 32x32, +15% for 64x64.
* **Oxide stress.** With the word line at `vneg`, a write puts `vdd - vneg` across the write
  device's and the output stage's gate oxide: 2.1 V at the -0.3 V default (2.28 V at the ff supply),
  2.6 V at -0.8 V (2.78 V), above what the 1.8 V devices are rated for. A write pulse is short, so
  the time-dependent stress is small, but it is a reliability question this flow does not answer;
  `Characterisation.write_stress()` reports the number. The read allows |`vneg`| down to 0.2 V.
* **The supply.** `vneg` is a pin: something has to generate `vneg` (a charge pump, or an external
  supply), and it sees a peak of 0.25 to 0.6 mA in a write and under 3 uA at rest. That generator is
  not part of this compiler.
* **Ring taps.** The deep n-well is biased through the level shifter's n-well tap; the ring's right
  wall has no tap of its own.
* **Retention.** In the table above: 122 us at ss, the worst corner. That is *shorter* than the
  LVT options' despite a much larger margin in volts -- this option keeps the default 0.18 um
  write device, and the device that holds the storage node leaks more than the 0.35 um LVT one.
  Margin in volts does not translate directly into refresh time.

## A supply-free alternative: LVT write device + a longer read gate

The negative rail fixes a written '0' but adds a pin something else has to supply, 2.1 to 2.6 V of gate-oxide
stress in a write, and +15 to 30% area. The finding under "Replica bit-line and timing
characterisation" above has another lever: the '0' a PMOS write leaves is about `|Vtp|` above its
gate, and the read device's own threshold sets how high a '0' it still calls a '0'. Both are ordinary
`BitcellConfig` fields, not a new option -- `write_flavor`
picks the write device's implant (`lvt` lowers `|Vtp|`, so the write pulls lower) and `read_length` is
the read gate's length (a longer NMOS has a higher threshold, so it tolerates a higher '0'). Neither
needs a new driver, a new supply, or a pin: the LVT cell is a bitcell like any other and drops into
the existing drivers, decoders and column circuits unchanged. Its `mr_length=0.5` variant (below) was
the compiler's default cell until its write device's leakage turned up (next section), and is now
`BitcellConfig(write_flavor="lvt", write_length=0.35, mr_length=0.5)` (CLI `--write-flavor lvt
--write-length 0.35 --mr-length 0.5`); the "both read gates" variant is
`BitcellConfig.plain(write_flavor="lvt", write_length=0.35, read_length=0.5)` (CLI `--plain-cell
--write-flavor lvt --write-length 0.35 --read-length 0.5`). (Everything in this section was measured
with the write bit lines idling low, which hides that leakage.)

`write_length` has to grow from 0.18 to 0.35 um because sky130's LVT PMOS model is only characterised
at `L >= 0.35`; `read_length` from 0.15 to 0.5 um is chosen the same way `vneg` was for the negative
rail -- the smallest that passes every corner with margin. The cell itself grows a little in `x`: an
LVT/HVT marker has its own enclosure of the diffusion it covers, and where that is wider than the poly
endcap the marker, not the gate, sets how close the diffusion can sit to the cell edge (`diff_margin`
in `cells/bitcell_3t.py`), so it never pokes out of the n-well at the array's boundary.

**Result** (extracted, 16x20 decoded macro with replica; DRC clean, LVS match), every variant:

| option | area (um2/bit) | access @ tt (ns) | sf '0' / tolerance (V) | sf MC '0' fails | worst refresh (us) |
|---|---|---|---|---|---|
| plain cell | 19.13 | 1.03 | 0.88 / 0.86 | 25/50 | 0 |
| `write_flavor=lvt` only | 20.16 | 1.05 | 0.47 / 0.86 | 0/50 | 382 |
| + `mr_length=0.5` | 20.16 | 1.14 | 0.50 / 1.03 | 0/50 | 703 |
| + `read_length=0.5` (both read gates) | 21.83 | 1.52 | 0.50 / 0.93 | 0/50 | 617 |
| + `read_length=1.0` | 26.81 | 2.33 | 0.53 / 0.93 | 0/50 | 994 |

Per corner, the recommended variant -- the default cell before (`write_flavor=lvt`, `write_length=0.35`, `mr_length=0.5`):

| corner | '0' written (V) | read tolerance (V) | margin (V) | '0' reads | access (ns) | write (ns) | refresh (us) |
|---|---|---|---|---|---|---|---|
| tt_1.8V_25C | 0.37 | 1.08 | +0.71 | ok | 1.14 | 0.21 | 1670 |
| ss_1.62V_125C | 0.28 | 1.03 | +0.75 | ok | 2.15 | 0.35 | 703 |
| ff_1.98V_-40C | 0.41 | 1.13 | +0.72 | ok | 0.72 | 0.14 | 1589 |
| sf_1.8V_25C | 0.50 | 1.03 | +0.53 | ok | 1.26 | 0.29 | 1449 |
| fs_1.8V_25C | 0.24 | 1.17 | +0.93 | ok | 1.16 | 0.14 | 1926 |

With mismatch (50 dies per corner):

| corner | access (ns) | dout/rdy | '0' written (V) | '1' / '0' fails |
|---|---|---|---|---|
| tt_1.8V_25C | 1.24 ± 0.02 | 1.15 ± 0.02 | 0.37 ± 0.05 | 0/50, 0/50 |
| ss_1.62V_125C | 2.39 ± 0.05 | 1.13 ± 0.03 | 0.28 ± 0.05 | 0/50, 0/50 |
| ff_1.98V_-40C | 0.76 ± 0.01 | 1.18 ± 0.02 | 0.42 ± 0.05 | 0/50, 0/50 |
| sf_1.8V_25C | 1.31 ± 0.03 | 1.11 ± 0.02 | 0.50 ± 0.05 | 0/50, 0/50 |
| fs_1.8V_25C | 1.31 ± 0.02 | 1.19 ± 0.03 | 0.24 ± 0.05 | 1/50, 0/50 |

What this says:

* **The LVT write device does most of the work on the level.** It lowers the sf '0' from 0.88 to
  0.47 V for +5% area and no access-time cost.
* **`mr_length` is the right read-side knob.** Only `mr`'s threshold sets the tolerance; `ms`
  in series only adds cell height and `rwl` load. `mr_length=0.5` raises the sf tolerance from
  0.86 to 1.03 V at no further area in this layout, and nearly doubles the worst refresh interval
  (382 to 703 us). Lengthening both read gates (`read_length`) buys less tolerance for more area
  and much slower access (1.52 ns at 0.5 um, 2.33 ns at 1.0 um, against 1.14).
* **Every variant reads every die's '0' in every corner** (0/50 failures).
* **Retention** (`measurements/data/retention.json`, measured on the extracted macro): the worst corner is
  ss for every variant. The drop at 125 C is the storage node's junction: the write device's p+
  drain is reverse-biased against its n-well while a '0' is stored. With the models' junction
  saturation currents zeroed the 125 C interval goes from 74 to 590 us (plain) and 819 to 1852 us
  (this variant); zeroing GIDL changes nothing (`measurements/scripts/check_leakage.py`). 703 us for the recommended one; 994 us at `read_length=1.0`, at +40% area
  and twice the access time.

* **No negative supply, no gate-oxide stress, no extra pin.** Unlike the negative rail, nothing outside
  the macro has to change, which is why it was the default cell (the negative rail stays available for the plain
  cell, `--plain-cell --neg-wl`).

## The write device's leakage, and the default cell

An off write device -- gate at vdd -- still leaks from its write bit line into the storage node. While
the write bit line is high, a stored '0' leaks up, and the device's gate-source voltage stays at zero
as it does (its source is the bit line, at vdd), so it keeps leaking all the way; a '1' beside a low
write bit line leaks down only until its device's gate-source voltage turns it further off (1.80 to
1.61 V in 10 us at tt). How much it leaks depends on the device, and on sky130 the LVT one -- the
default cell's until now -- leaks three to five orders of magnitude more than the standard one
(`measurements/scripts/compare_write_device.py leak`, storage node at 0.3 V, write bit line and gate
at vdd; `measurements/data/write_device.json`):

| corner | LVT write device (0.42/0.35 um) | standard-Vt write device (0.42/0.18 um) |
|---|---|---|
| tt, 25 C | 2.6 nA | 2.9 pA |
| ss, 125 C | 22.6 nA | 2.8 pA |
| ff, -40 C | 1.1 nA | 4.9 pA |
| sf, 25 C | 0.03 nA | 1.5 pA |
| fs, 25 C | 134 nA | 0.43 nA |

The refresh intervals of the LVT cells (above) were measured with the write bit lines low. A macro
does not keep them there:

* **Idle.** With in-macro refresh, `din` was held low outside a write (to save read energy, see
  [Refresh](#refresh---refresh)), so `NAND(0, rbl)` held every write bit line high: on the extracted
  macro (16x8), 2 us of idling lost every '0' at tt, ss, ff and fs. The controller now holds `din`
  high outside a write; the write bit lines idle low -- and at fs (134 nA) the LVT cell still loses
  them within the 2 us (`compare_write_device.py macro`, below).
* **Reads and writes.** Every read of a '1' raises its column's write bit line (`NAND(1, rbl)`) until
  the precharge, and every write of a '1' holds it high through the write: 300 reads of another row
  of '1's (2 us of traffic) still lost every '0' of the column at tt, ss, ff and fs. No idle level of
  the write bit line fixes that.

So the default cell is now a **standard-Vt write device (0.18 um) with a 1.0 um `mr`**
(`BitcellConfig()`). Its written '0' is higher than the LVT device's (about 1.0 V at sf on the
macro, the slow PMOS, and the release of the word line and the bit line couple it up a little), which a
0.5 um `mr` misreads at sf; a 1.0 um one does not. On the extracted macro, idle 2 us and hammered by
300 reads (`compare_write_device.py macro`):

| cell | tt | ss | ff | sf | fs |
|---|---|---|---|---|---|
| LVT write, 0.5 um `mr` (the default before), `din` high idle | idle ok, hammered **lost** | ok, **lost** | ok, **lost** | ok, ok | **lost**, **lost** |
| standard-Vt write, 0.5 um `mr` | ok, ok | ok, ok | ok, ok | **misread**, **misread** | ok, ok |
| standard-Vt write, 1.0 um `mr` (the default) | ok, ok | ok, ok | ok, ok | ok, ok | ok, ok |

(These are nominal devices. Under mismatch the default cell's '0' at sf has no margin left: one of
20 dies misreads it on a 16x20 macro, and with the loop-back refresh 18 of 20 do --
[Self-timed reads](#self-timed-reads-the-default-with-the-replica---no-self-timed).)

A longer LVT write device lowers that '0' without the LVT cell's disturb, but trades sf for fs. With
a 0.5 um `mr`, a 0.75 um or 1.0 um LVT write device writes the sf '0' at 0.70-0.73 V (0.21 V of
margin with in-macro refresh, 0.12 V with the loop-back, no failing die of 20), and an idle '0' holds at
every corner. Hammered, though, both lose every '0' of the column at fs: their off current there is
1.8 nA and 0.73 nA against the standard device's 0.43 nA. They also cost 8% and 17% of cell area,
35-70x the standard device's leakage at ss, and a layout fix: their write word line's poly covers the
strap column's n-well tap (615 DRC violations and an LVS short on an 8x8 macro at 0.75 um). The
default stays; sf is its marginal corner.

Its refresh interval, measured as before (`sweep_retention.py`, `std_mr1`; write bit lines low, as
the controller now holds them, and no traffic needed to break it): 2331 / 193 / 2381 / 1223 / 2986 us
at tt / ss / ff / sf / fs, **154 us** for the worst of 20 mismatch dies at ss (724 us at sf) -- the
controller's default for the cell. The cost: a 7% larger cell than the LVT one (1.22 x 2.93 um,
3.58 um^2, against 3.34), and slower reads -- `rdy` at about 2.4 ns at tt and 4.3 ns at ss on the
16x8 macro, against 1.7 and 3.0. IHP SG13G2 is not affected: it has no LVT device, and its
standard write device leaks 3 to 80 pA.

(Most numbers elsewhere in this README -- macro areas, access times, the hard blocks -- were
measured with the LVT cell. Where they depend on the cell they move by about its 7% of array area
and its slower read.)

## License

Apache License 2.0 (`LICENSE`).
