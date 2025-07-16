import torch
import hydra
from hydra.utils import instantiate
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def translate_text(text, model, tokenizer, device, max_length=512):
    """
    Переводит текст с английского на русский используя Hugging Face модель
    """
    model.eval()
    
    with torch.no_grad():
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

       
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        
        
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translation


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Простой скрипт для перевода текста с английского на русский
    """
    
    torch.manual_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    print(f"Using device: {device}")

   
    model_name = "Helsinki-NLP/opus-mt-en-ru"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    print(f"Loaded Hugging Face model: {model_name}")

    
    text_to_translate = config.inferencer.text

    if not text_to_translate:
        print("No text provided in config")
        return

    print(f"Translating: {text_to_translate}")

   
    translation = translate_text(text_to_translate, model, tokenizer, device)
    
  
    output_file = config.inferencer.output_file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"English: {text_to_translate}\n")
        f.write(f"Russian: {translation}\n")
    
    print(f"Translation saved to {output_file}")
    print(f"English: {text_to_translate}")
    print(f"Russian: {translation}")


if __name__ == "__main__":
    main()