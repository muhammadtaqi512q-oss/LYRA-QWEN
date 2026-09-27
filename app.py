from flask import Flask, render_template, request, jsonify
import torch
from transformers import pipeline

app = Flask(__name__)

print("Loading Hermes-3-Llama-3.2-3B Model...")
pipe = pipeline(
    "text-generation",
    model="muhammad-taqi512/LYRA-M-LLAMA",
    torch_dtype=torch.float32,
    device_map="auto"
)
print("Model Loaded Successfully!")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    user_prompt = data.get("prompt", "")

    if not user_prompt:
        return jsonify({"response": "Please enter a message."}), 400

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant, YOUR NAME IS LYRA-M-LLAMA AND YOU ARE FEMALE AI, YOUR OWNER AND CREATOR AND FOUNDER IS MUHAMMAD TAQI."},
        {"role": "user", "content": user_prompt}
    ]

    # Apply chat template for Hermes-3
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    outputs = pipe(
        prompt, 
        max_new_tokens=256, 
        do_sample=True, 
        temperature=0.7, 
        top_k=50, 
        top_p=0.95
    )

    generated_text = outputs[0]["generated_text"]

    # Parsing output for Hermes-3 ChatML format
    if "<|im_start|>assistant" in generated_text:
        response = generated_text.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
    else:
        # Fallback if the template structure varies
        response = generated_text[len(prompt):].strip()

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
