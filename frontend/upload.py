import os,sys
import subprocess
from flask import Flask, flash, request, redirect, render_template, send_from_directory, send_file
from flask import Markup
from werkzeug.utils import secure_filename

app=Flask(__name__)
app.secret_key = "secret_" # for encrypting the session
#It will allow below 16MB contents only, you can change it
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')
# Make directory if "uploads" folder not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return True

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/uploads/<filename>', methods=['GET', 'POST'])
def download(filename):
    uploads = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
    file = os.path.join(uploads, filename)
    print(file)
    return send_file(file, as_attachment=True)

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('files[]')

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # change it here!
        batcmd="cd ~/src/zk-ml/uchikoma && python3 main.py /data/ethsf/uploads/deploy_graph.txt /data/ethsf/uploads/deploy_param.params -o /data/ethsf/uploads/model -in %4 -on %158"
        result = subprocess.check_output(batcmd, shell=True)
        with open("uploads/model.txt", "w") as f:
            f.write(result.decode())
        flash(Markup('Model successfully converted, download here: <a href="/uploads/model.circom">model.circom</a> <a href="/uploads/model.json">model.json</a> <a href="/uploads/model.txt">trace</a>'))
        return redirect('/')

if __name__ == "__main__":
    app.run(host='127.0.0.1',port=6000,debug=False,threaded=True)
