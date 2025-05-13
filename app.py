from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '').lower()
    
    if user_input == 'hi':
        return jsonify({'reply': 'Hello!'})
    else:
        return jsonify({'reply': f'You said: {user_input}'})

if __name__ == '__main__':
    app.run(debug=True)
