from flask import Flask, jsonify, request
from utils import face
from werkzeug.utils import secure_filename
import os
app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def valid_file_request(request):
  if 'file' not in request.files:
    return False

  file = request.files['file']

  if file.filename == '':
    return False

  if file and allowed_file(file.filename):
    return True
  else:
    return False

@app.route("/recognize", methods=['POST'])
def recognize_faces():
  if valid_file_request(request):
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(f'data/test_face/{filename}')
  else:
    return f"Invalid file. Only allowed files {ALLOWED_EXTENSIONS}", 400

  result = face.recognize_face(f'data/test_face/{filename}')
  return jsonify({
      'result': result,
    }), 200

@app.route("/train_faces", methods=['POST'])
def train_faces():
 if valid_file_request(request):
  file = request.files['file']
  filename = secure_filename(file.filename)
  file_contents = file.read()

  if os.path.exists(f'data/train_faces/{filename.split(".")[0]}'):
    return f"Username already exists", 409
  else:
    os.mkdir(f'data/train_faces/{filename.split(".")[0]}')

  for i in range(30):
    with open(f'data/train_faces/{filename.split(".")[0]}/face_{i}.jpg', 'wb') as f:
      f.write(file_contents)

  face.train_faces(f'data/train_faces/{filename.split(".")[0]}')
  return "Training Completed", 200
 else:
  return f"Invalid file. Only allowed files {ALLOWED_EXTENSIONS}", 400


if __name__ == '__main__':
  app.run()