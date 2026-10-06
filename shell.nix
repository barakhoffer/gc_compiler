# Dev shell for NixOS: the KLayout/gdsfactory wheels are prebuilt binaries that need a
# few shared libraries which NixOS does not expose globally.
{pkgs ? import <nixpkgs> {}}:
let
  # OpenVAF compiles IHP SG13G2's Verilog-A transistor models (PSP 103) into the OSDI libraries
  # ngspice loads. Not in nixpkgs: the official 23.5.0 Linux binary, pinned, patched for NixOS.
  openvaf = pkgs.stdenv.mkDerivation {
    pname = "openvaf";
    version = "23.5.0";
    src = pkgs.fetchurl {
      url = "https://openva.fra1.cdn.digitaloceanspaces.com/openvaf_23_5_0_linux_amd64.tar.gz";
      sha256 = "1w748jgii026qsrqs0x4cfdxkfv1na4bx1ywc3sak9s8v65f1h3r";
    };
    sourceRoot = ".";
    nativeBuildInputs = [pkgs.autoPatchelfHook];
    buildInputs = [pkgs.stdenv.cc.cc.lib];
    installPhase = "install -Dm755 openvaf $out/bin/openvaf";
  };
in
pkgs.mkShell {
  packages = with pkgs; [
    openvaf # IHP SG13G2's device models, for ngspice
    python312
    uv
    klayout
    ngspice
    magic-vlsi
    iverilog
    # the refresh controller as hardware: LibreLane's RTL-to-GDS flow and the tools it drives.
    # Only its executable: as a package it would put its own (3.13) Python modules on
    # PYTHONPATH, where the project's 3.12 venv would import them.
    (writeShellScriptBin "librelane" ''exec ${librelane}/bin/librelane "$@"'')
    yosys
    openroad
    netgen-vlsi # LVS (nixpkgs' plain "netgen" is a mesher)
    verilator # LibreLane's linter
  ];

  # Shared libraries required by the manylinux wheels (klayout, matplotlib, ...).
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
    expat
    zlib
    curl
    libpng
    stdenv.cc.cc.lib
  ]);

  # sky130A, installed with ciel/volare.
  PDK_ROOT = builtins.getEnv "HOME" + "/.ciel";
  PDK_VARIANT = "sky130A";

  shellHook = ''
    export UV_PYTHON=${pkgs.python312}/bin/python3.12
  '';
}
