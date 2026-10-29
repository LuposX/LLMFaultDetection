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
      # Force every Python package (accelerate, lm-eval, ...) onto the ROCm
      # torch build; otherwise the default torch is pulled in too and the
      # env fails with duplicate site-packages paths. This mirrors nixpkgs'
      # torchWithRocm but on the pre-override torch to avoid recursion.
      pythonWithRocm = pkgs.python3.override {
        packageOverrides = pyself: pyprev: {
          torch = pyprev.torch.override {
            triton = pyprev.triton-no-cuda;
            rocmSupport = true;
            cudaSupport = false;
          };
          # not in nixpkgs; see pkgs/laya.nix. Depends on the ROCm torch
          # above, which is why it lives in this override set.
          laya = pyself.callPackage ./pkgs/laya.nix {
            torch = pyself.torch;
          };
        };
      };
      pythonEnv = pythonWithRocm.withPackages (ps: with ps; [
        torchWithRocm # PyTorch with AMD ROCm backend
        transformers
        accelerate
        lm-eval
        matplotlib
        sentence-transformers # embedding-based fault-effect metrics (sembench.py)
        laya # decision-model judge for the semantic-difference metric
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
