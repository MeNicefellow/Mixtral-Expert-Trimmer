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

Sure, you can add a disclaimer section to your `README.MD` file. Here's an example of how you might word it:

## Example Model

An example of a product model created with this tool can be found at [DrNicefellow/Mixtral-6x7B-Instruct-v0.1-bfloat16-Trimmed024567](https://huggingface.co/DrNicefellow/Mixtral-6x7B-Instruct-v0.1-bfloat16-Trimmed024567) on Hugging Face Model Hub.


## Disclaimer

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

Users are responsible for checking and validating the correctness of their configuration files, safetensor files, and binary files generated using the software. The developers assume no responsibility for any errors, omissions, or other issues coming in these files, or any consequences resulting from the use of these files.


## License

This project is licensed under the terms of the Apache License 2.0.

## Discord Server

Join our Discord server [here](https://discord.gg/xhcBDEM3).

## Feeling Generous? 😊

Eager to buy me a cup of 2$ coffe or iced tea?🍵☕ Sure, here is the link: [https://ko-fi.com/drnicefellow](https://ko-fi.com/drnicefellow). Please add a note on which one you want me to drink?