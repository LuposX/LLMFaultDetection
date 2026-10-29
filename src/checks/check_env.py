"""Environment sanity check: versions, GPU visibility, optional model download."""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-test", action="store_true",
                        help="run a bf16 matmul on the GPU")
    parser.add_argument("--download", action="store_true",
                        help="download the model + tokenizer")
    args = parser.parse_args()

    print(f"torch            {torch.__version__}")
    print(f"hip (rocm)       {getattr(torch.version, 'hip', None)}")
    print(f"cuda available   {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"device           {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"capability       {props.major}.{props.minor} "
              f"({props.total_memory / 1e9:.1f} GB)")
        if args.gpu_test:
            a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
            b = a @ a
            torch.cuda.synchronize()
            print(f"bf16 matmul ok   ({b.shape}, sum={b.sum().item():.3e})")

    if args.download:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = "Qwen/Qwen3-0.6B"
        AutoTokenizer.from_pretrained(model_id)
        AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        print(f"downloaded       {model_id}")


if __name__ == "__main__":
    main()
