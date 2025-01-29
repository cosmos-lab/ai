
from transformers import pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)

def read_text_from_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read() 
    return text

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")

def answer_questions_from_txt(file_path, question):
    file_text = read_text_from_file(file_path)
    # answer = qa_pipeline(question=question, context=file_text)

    answer = qa_pipeline({
    'context': file_text,
    'question': question
})
    
    return answer['answer']

@app.route('/', methods=['GET'])
def answer_question():
    question = request.args.get('q', '')
    answer = answer_questions_from_txt("content.txt", question)
    return jsonify({'answer': answer})

if __name__ == "__main__":
     app.run(host='0.0.0.0', port=5000, debug=True)

