# Laya (Convai Innovations): non-autoregressive "System 1" decision model
# (ModernBERT-large + decision head). Used here as a semantic judge: given a
# state holding the healthy reference output and the faulty model's output,
# a noul question returns a calibrated P(same meaning).
# Not packaged in nixpkgs; built from the PyPI sdist. Core dependencies are
# exactly the ones already in the dev environment (torch, transformers,
# safetensors, huggingface-hub, numpy); serve/fast/mcp/langchain extras are
# not needed and not built.
{
  lib,
  buildPythonPackage,
  fetchPypi,
  setuptools,
  torch,
  transformers,
  safetensors,
  huggingface-hub,
  numpy,
}:

buildPythonPackage rec {
  pname = "laya";
  version = "0.4.7";
  pyproject = true;

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-w/Vn8hzphg6wY8Z7NTFtVJiIHmhW9tWPKwXOQxHossk=";
  };

  build-system = [ setuptools ];
  dependencies = [
    torch
    transformers
    safetensors
    huggingface-hub
    numpy
  ];

  # tests download model weights from Hugging Face
  doCheck = false;
  pythonImportsCheck = [ "laya" ];

  meta = {
    description = "Fast, non-autoregressive System 1 decision engine with calibrated probabilities";
    homepage = "https://pypi.org/project/laya/";
    license = lib.licenses.asl20;
  };
}
