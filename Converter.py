from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

app = Flask(__name__)

# Load tokenizer and model
model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = TFAutoModelForSeq2SeqLM.from_pretrained("model/")  # Replace with correct path


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    input_text = request.form["input_text"]

    # Tokenize input text
    tokenized = tokenizer([input_text], return_tensors='np')

    # Generate output
    out = model.generate(**tokenized, max_length=128)

    # Decode output
    with tokenizer.as_target_tokenizer():
        decoded_output = tokenizer.decode(out[0], skip_special_tokens=True)

    return jsonify({"output": decoded_output})


if __name__ == "__main__":
    app.run(debug=True)
