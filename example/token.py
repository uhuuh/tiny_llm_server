from transformers import AutoTokenizer

if __name__ == '__main__':
    model_path = "/mnt/c/Users/uh/code/ckpt/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(tokenizer.encode("Despite the stormy weather and the constant noise outside, "
                           "she continued reading her favorite mystery novel by the fireplace, "
                           "sipping hot chocolate slowly, "
                           "occasionally glancing at the rain-soaked window, "
                           "lost in thought, completely absorbed in the world of suspense and imagination."))

