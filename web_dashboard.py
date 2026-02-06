"""
실시간 웹 대시보드 - Flask 서버
"""
from flask import Flask, render_template, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    """메인 대시보드 페이지"""
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    """실시간 데이터 API"""
    from ml_grid_bot import bot_instance
    
    if bot_instance is None:
        return jsonify({'error': 'Bot not running'}), 503
    
    try:
        data = bot_instance.get_dashboard_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def start_web_server():
    """웹 서버 시작"""
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    start_web_server()
