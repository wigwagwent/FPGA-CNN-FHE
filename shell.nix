{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    rustc
    cargo
    rustfmt
    rust-analyzer
    clippy
    pkg-config
    openssl
    libiconv
  ];

  RUST_BACKTRACE = 1;

  shellHook = ''
    echo "Rust development environment"
    echo "Rust version: $(rustc --version)"
    echo "Cargo version: $(cargo --version)"
  '';
}
