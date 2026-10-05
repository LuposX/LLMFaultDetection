{
  # Dev environment for bit-flip fault injection studies on Qwen3-0.6B.
  #
  # Fully declarative: all Python dependencies come from nixpkgs stable.
  # Stable (nixos-25.11) is deliberate — torchWithRocm 2.9.1 is in the
  # binary cache there, while on unstable it would compile the whole ROCm
  # closure from source (see DESIGN.md). ROCm 6.4.3 on this branch is the
  # first series with RDNA4 (gfx1201, RX 9070 XT) support.

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      pythonEnv = pkgs.python3.withPackages (ps: with ps; [
        torchWithRocm # PyTorch with AMD ROCm backend
        transformers
        accelerate
        lm-eval
        matplotlib
      ]);
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pythonEnv
          pkgs.git
        ];
      };
    };
}
