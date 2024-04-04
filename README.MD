# Mixtral Expert Trimmer

This program is designed to trim the experts in Mixtral models. It works by keeping only the selected experts at every layer of the model.

## Usage

The program takes three arguments:

- `--model_id`: The path to the pretrained Mixtral model.
- `--target_dir`: The directory where the trimmed model and its tokenizer will be saved.
- `--kept_experts`: The indices of the experts to keep. This should be a space-separated list of integers.

Example usage:

```bash
python main.py --model_id /workspace/models/mistralai_Mixtral-8x7B-Instruct-v0.1 --target_dir /src/models/mistralai_Mixtral-6x7B-Instruct-v0.1 --kept_experts 0 2 4 5 6 7
```

## License

This project is licensed under the terms of the Apache License 2.0.
```
