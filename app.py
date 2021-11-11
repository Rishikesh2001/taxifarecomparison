from flask import Flask, jsonify
from flask.globals import request
from functions import *

app = Flask(__name__)
@app.route('/uber')

def priceUber():
   distance = request.args['distance']
   cab_type = request.args['cab_type']
   response = funUber(distance, cab_type)

   return jsonify(response)


@app.route('/ola')
def priceOla():
   distance = request.args['distance']
   cab_type = request.args['cab_type']
   response = funOla(distance, cab_type)

   return jsonify(response)

if __name__ == '__main__':
   app.run(debug=True)