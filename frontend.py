import os
from flask import Flask, render_template, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
	__name__,
	template_folder=os.path.join(BASE_DIR, 'templates'),
	static_folder=os.path.join(BASE_DIR, 'static')
)


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
	return send_from_directory(os.path.join(app.static_folder, 'img'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=3000, debug=True)

