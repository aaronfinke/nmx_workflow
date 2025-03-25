from flask import Flask, render_template, Response, jsonify, send_file
import random

import plotly.graph_objs as go
import plotly.io as pio
import os
import time
import json

from nmx_workflow.config import INTERIM_DATA_DIR

app = Flask(__name__)

# Simulated data
data = []

BINNING_LOG_PATH = INTERIM_DATA_DIR/'binning.log'
INSTRUMENTVIEW_PATH = INTERIM_DATA_DIR/'view.html'
HISTOGRAMS_JSON_PATH = INTERIM_DATA_DIR/'spotfinding.json'
INTEGRATION_JSON_PATH = INTERIM_DATA_DIR/'integration.json'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream_binning_log')
def stream_binning_log():
    def generate():
        # Wait for the file to be created
        while not os.path.exists(BINNING_LOG_PATH):
            # yield "data: Waiting for binning log to be created...\n\n"
            time.sleep(1)

        # print("Binning log found. Streaming data...")  # Debug
        with open(BINNING_LOG_PATH, 'r') as f:
            # Read the entire file from the beginning
            for line in f:
                yield f"data: {line.strip()}\n\n"
                if "Binning complete" in line:  # Check for completion keyword
                    # print("Binning process complete. Stopping stream.")  # Debug
                    # yield "data: Binning process complete. Stopping stream.\n\n"
                    return  # Stop streaming

        with open(BINNING_LOG_PATH, 'r') as f:
            # Seek to the end of the file
            f.seek(0, 2)
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line.strip()}\n\n"
                    if "Binning complete" in line:  # Check for completion keyword
                        return  # Stop streaming

                else:
                    time.sleep(1)  # Wait for new lines to be written

    return Response(generate(), content_type='text/event-stream')

@app.route('/check_instrumentview')
def check_instrumentview():
    if os.path.exists(INSTRUMENTVIEW_PATH):
        return jsonify({'ready': True})
    else:
        return jsonify({'ready': False})
        
@app.route('/get_instrumentview')
def get_instrumentview():
    if os.path.exists(INSTRUMENTVIEW_PATH):
        with open(INSTRUMENTVIEW_PATH, 'r') as f:
            return f.read()
    else:
        return "Instrument View is not ready yet.", 404
    
@app.route('/check_histograms')
def check_histograms():
    if os.path.exists(HISTOGRAMS_JSON_PATH):
        with open(HISTOGRAMS_JSON_PATH, 'r') as f:
            histograms = json.load(f)

        # Deserialize binary data if necessary
        for uid, histogram in histograms.items():
            for trace in histogram['data']:
                if 'x' in trace and isinstance(trace['x'], dict) and 'bdata' in trace['x']:
                    trace['x'] = deserialize_binary_data(trace['x'])
                if 'y' in trace and isinstance(trace['y'], dict) and 'bdata' in trace['y']:
                    trace['y'] = deserialize_binary_data(trace['y'])

        return jsonify({'ready': True, 'histograms': histograms})
    else:
        return jsonify({'ready': False})

@app.route('/check_integration')
def check_integration():
    if os.path.exists(INTEGRATION_JSON_PATH):
        with open(INTEGRATION_JSON_PATH, 'r') as f:
            integrations = json.load(f)

        # Deserialize binary data if necessary
        for uid, integration in integrations.items():
            for name in ['fig_res','fig_img']:
                for trace in integration[name]['data']:
                    if 'x' in trace and isinstance(trace['x'], dict) and 'bdata' in trace['x']:
                        trace['x'] = deserialize_binary_data(trace['x'])
                    if 'y' in trace and isinstance(trace['y'], dict) and 'bdata' in trace['y']:
                        trace['y'] = deserialize_binary_data(trace['y'])

        return jsonify({'ready': True, 'integration': integrations})
    else:
        return jsonify({'ready': False})


@app.route('/get_indexing_results')
def get_indexing_results():
    INDEXING_JSON_PATH = INTERIM_DATA_DIR / 'indexing.json'
    if os.path.exists(INDEXING_JSON_PATH):
        with open(INDEXING_JSON_PATH, 'r') as f:
            indexing_results = json.load(f)
        return jsonify({'ready': True, 'indexing_results': indexing_results})
    else:
        return jsonify({'ready': False})  

@app.route('/get_refine_results')
def get_refine_results():
    INDEXING_JSON_PATH = INTERIM_DATA_DIR / 'refine.json'
    if os.path.exists(INDEXING_JSON_PATH):
        with open(INDEXING_JSON_PATH, 'r') as f:
            refine_results = json.load(f)
        return jsonify({'ready': True, 'refine_results': refine_results})
    else:
        return jsonify({'ready': False})  

def deserialize_binary_data(data):
    """Deserialize binary data from the JSON."""
    import base64
    import numpy as np

    bdata = base64.b64decode(data['bdata'])
    dtype = np.dtype(data['dtype'])
    return np.frombuffer(bdata, dtype=dtype).tolist()

def wait_for_file(filepath):
    """Wait for a file to be created, with a timeout."""
    while not os.path.exists(filepath):
        time.sleep(1)  # Check every second

@app.route('/check_binning')
def check_binning():
    # Simulate readiness check for binning
    if os.path.exists(BINNING_LOG_PATH):
        return jsonify({'ready': True})
    return jsonify({'ready': False})

@app.route('/check_indexing')
def check_indexing():
    # Simulate readiness check for indexing
    INDEXING_JSON_PATH = INTERIM_DATA_DIR / 'indexing.json'
    if os.path.exists(INDEXING_JSON_PATH):
        return jsonify({'ready': True})
    return jsonify({'ready': False})

@app.route('/check_refinement')
def check_refinement():
    # Simulate readiness check for refinement
    REFINEMENT_JSON_PATH = INTERIM_DATA_DIR / 'refinement.json'
    if os.path.exists(REFINEMENT_JSON_PATH):
        return jsonify({'ready': True})
    return jsonify({'ready': False})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)