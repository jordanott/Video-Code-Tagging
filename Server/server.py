from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/link/<int:link>')
def get_link(link):
    print link
    # show the post with the given id, the id is an integer
    return 'Link %d' % link

if __name__ == "__main__":
    # connect to ip adress
    app.run(host='0.0.0.0')
