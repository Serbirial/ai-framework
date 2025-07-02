import os
import argparse
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from onnxruntime_tools import optimizer
from onnxruntime_tools.transformers.onnx_model_t5 import T5OptimizationOptions
from onnxruntime.quantization import quantize_dynamic, QuantType


def export_to_onnx(model_name, export_dir):
    print(f"[+] Exporting {model_name} to ONNX...")
    model = ORTModelForSeq2SeqLM.from_pretrained(model_name, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)
    return os.path.join(export_dir, "model.onnx")


def optimize_onnx(model_path, model_name):
    print(f"[+] Optimizing ONNX graph...")
    options = T5OptimizationOptions(model_name)
    options.enable_gelu = True
    options.enable_layer_norm = True

    optimized_model = optimizer.optimize_model(
        model_path,
        model_type="t5",
        num_heads=12,
        hidden_size=768,
        optimization_options=options,
    )
    opt_path = model_path.replace(".onnx", "_opt.onnx")
    optimized_model.save_model_to_file(opt_path)
    return opt_path


def quantize_model(model_path):
    print(f"[+] Quantizing model to INT8...")
    quant_path = model_path.replace(".onnx", "_int8.onnx")
    quantize_dynamic(
        model_input=model_path,
        model_output=quant_path,
        weight_type=QuantType.QInt8,
        optimize_model=True
    )
    return quant_path


def run_all(model_name, output_dir="quant_output"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[+] Processing model: {model_name}")
    model_path = export_to_onnx(model_name, output_dir)
    optimized_path = optimize_onnx(model_path, model_name)
    quantized_path = quantize_model(optimized_path)

    print(f"\nDone! Quantized model saved at:\n{quantized_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize + Quantize T5-like models for Pi Zero 2W")
    parser.add_argument("model_name", help="HuggingFace model name (e.g., google/flan-t5-small)")
    parser.add_argument("--output_dir", default="quant_output", help="Where to save the ONNX + quantized models")
    args = parser.parse_args()

    run_all(args.model_name, args.output_dir)
