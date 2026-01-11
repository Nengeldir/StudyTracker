from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)
NOTES_FILE = "MindmapNotes.md"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/notes', methods=['GET'])
def get_notes():
    if not os.path.exists(NOTES_FILE):
        return jsonify({"content": ""}), 200
    try:
        with open(NOTES_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({"content": content}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/notes', methods=['POST'])
def save_notes():
    try:
        data = request.json
        content = data.get('content')
        if content is None:
            return jsonify({"error": "No content provided"}), 400
        
        with open(NOTES_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
