from flask import Flask,render_template

app = Flask(__name__)

@app.route('/')
def index():
    s='<h1>Hello</h1><p>This is a simple <b>Flask App</b></p>'
    return s

@app.route('/home')
def home():
    return '<h2>Welcome to Home Page!</h2>'

@app.route('/home/<place>')
def place(place):
    return '<h2>Welcome to the <i>'+ place + '</i> Page!</h2>'
 
@app.route('/html1')
def html1():
    return render_template('example.html',myvar='A grand variable')

@app.route('/html2')
def html2():
    return render_template('example.html')
    
@app.route('/links')
def links():
    return render_template('example.html',links=["https://google.com","https://microsoft.com","https://github.com"])
        
if __name__=='__main__':
    app.run()