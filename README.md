# Llama-1b-Lora-Medical-qa
### THE HF_TOKEN is the access token for the Llama models . We have to get access token from out hugging face account and store at the secrets of kaggle addons.
### We have to ask for access of the Llama models and can use them only after the access is granted. The model i did PEFT is of 1 billion parameters.
### The above model is rank 16 LoRa with 4 bit Quantization . Lora done on q_proj and v_proj matrices . 
```sh
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

model_id = "mokshaik/llama-3-2-1b-medical-qa-lora-finetuned"
print(f"Loading model and tokenizer from Hugging Face Hub: {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, 
    load_in_4bit=True,         
    device_map="auto"          
)

model.eval()
print("Model loaded successfully for local inference!")

generation_config = GenerationConfig(
    max_new_tokens=512,       
    do_sample=True,
    temperature=0.7,          
    top_p=0.9,                 
    top_k=50,                 
    pad_token_id=tokenizer.pad_token_id, 
    eos_token_id=tokenizer.eos_token_id, 
)

prompt = """### Question:
Given the symptoms of sudden weakness in the left arm and leg, recent long-distance travel, and the presence of swollen and tender right lower leg, what specific cardiac abnormality is most likely to be found upon further evaluation that could explain these findings?

### Answer: """


print("Performing local inference...")
try:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad(): 
        generated_output = model.generate(
            **inputs,
            generation_config=generation_config,
        )

    response_text = tokenizer.decode(
        generated_output[0, inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    ).strip()

    if "### Question:" in response_text:
        response_text = response_text.split("### Question:")[0].strip()

    print("\nMODEL RESPONSE IS:")
    print(response_text)

except Exception as e:
    print(f"An error occurred during inference: {e}")
