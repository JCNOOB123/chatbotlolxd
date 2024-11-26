from flask import Flask, request, jsonify
from model.inference import ChatBot
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize chatbot with the trained model
chatbot = ChatBot('chatbot_model.pth')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    response = chatbot.generate_response(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 