# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)


model = GPT2LMHeadModel.from_pretrained("gpt2-kazakh")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-kazakh")


def generate_response(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


@app.route('/')
def index():
    return render_template('indexconcl.html')


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data['question']
    response = generate_response(question)
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)
