from flask import Flask, jsonify, request
from snailguard import protect_flask

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Welcome to SnailGuard AI Protected API"})

@app.route('/api/public')
def public():
    return jsonify({"data": "This is a public endpoint"})

@app.route('/api/protected')
@protect_flask()
def protected():
    return jsonify({
        "data": "This endpoint is protected by SnailGuard AI",
        "protection": {
            "zero_fp_guarantee": True,
            "economic_warfare": True,
            "cascade_phases": 4
        }
    })

@app.route('/api/sensitive', methods=['POST'])
@protect_flask(enable_economic_warfare=True)
def sensitive():
    data = request.get_json()
    return jsonify({
        "status": "processed",
        "user_data": "protected by economic warfare",
        "received": data
    })

if __name__ == '__main__':
    print("🚀 Starting Flask app with SnailGuard AI protection...")
    print("📧 Public endpoint: http://localhost:5000/api/public")
    print("🛡️ Protected endpoint: http://localhost:5000/api/protected") 
    print("💸 Economic warfare endpoint: POST http://localhost:5000/api/sensitive")
    app.run(debug=True, port=5000)