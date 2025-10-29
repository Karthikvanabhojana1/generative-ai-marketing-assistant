from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
# Load a text generation pipeline using GPT-2 model
generator = pipeline("text-generation", model="gpt2")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    outputs = generator(prompt, max_length=100, num_return_sequences=1)
    return jsonify({"generated_text": outputs[0]['generated_text']})

if __name__ == '__main__':
    app.run(debug=True)
