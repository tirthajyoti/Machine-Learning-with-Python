from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "<h1>Not Much Going On Here</h1><br> \
	But at least <b>you could fire up a web page</b> and put some <i>HTML code</i> in \
	using <a href=\"http://flask.pocoo.org/\">Flask</a>."
app.run(host='0.0.0.0', port=50000)