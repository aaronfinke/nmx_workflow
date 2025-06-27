from flask import Flask, render_template, Response, jsonify, stream_with_context, send_from_directory
from enum import Enum
import os
import time
import json

from nmx_workflow.config import INTERIM_DATA_DIR

app = Flask(__name__, static_folder="static")

# Simulated data
data = []

BINNING_LOG_PATH = INTERIM_DATA_DIR / "binning.log"
INSTRUMENTVIEW_PATH = INTERIM_DATA_DIR / "view.html"
INDEXING_JSON_PATH = INTERIM_DATA_DIR / "indexing.json"
HISTOGRAMS_JSON_PATH = INTERIM_DATA_DIR / "spotfinding.json"
INTEGRATION_JSON_PATH = INTERIM_DATA_DIR / "integration.json"
REFINE_JSON_PATH = INTERIM_DATA_DIR / "refine.json"

class Status(Enum):
    READY = "ready"
    ERROR = "error"
    PENDING = "pending"
    UNKNOWN = "unknown"

PROCESS_STATUS = {
    "binning": Status.UNKNOWN.value,
    "spotfinding": Status.UNKNOWN.value,
    "instrumentview": Status.UNKNOWN.value,
    "indexing": Status.UNKNOWN.value,
    "refinement": Status.UNKNOWN.value,
    "integration": Status.UNKNOWN.value,
    "scaling": Status.ERROR.value,
}

def update_process_status(process,status):
    PROCESS_STATUS[process] = status

@app.route('/stream_status')
def stream_status():
    def generate():
        while True:
            _ = indexing_status()
            _ = spotfinding_status()
            _ = instrumentview_status()
            _ = refine_status()
            _ = integration_status()
            _ = binning_status()
            yield f"data: {json.dumps(PROCESS_STATUS)}\n\n"
            time.sleep(3)  # Send updates every 5 seconds

    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route('/get_processing_status')
def get_processing_status():
    _ = indexing_status()
    _ = spotfinding_status()
    _ = instrumentview_status()
    _ = refine_status()
    _ = integration_status()
    _ = binning_status()
    return PROCESS_STATUS

@app.route("/")
def index():
    return render_template(
        "index.html"
    )


@app.route("/stream_binning_log")
def stream_binning_log():
    def generate():
        # Wait for the file to be created
        while not os.path.exists(BINNING_LOG_PATH):
            # yield "data: Waiting for binning log to be created...\n\n"
            time.sleep(1)

        # print("Binning log found. Streaming data...")  # Debug
        with open(BINNING_LOG_PATH, "r") as f:
            update_process_status("binning",Status.PENDING.value)
            # Read the entire file from the beginning
            for line in f:
                yield f"data: {line.strip()}\n\n"
                if "Binning complete" in line:  # Check for completion keyword
                    update_process_status("binning",Status.READY.value)
                    # print("Binning process complete. Stopping stream.")  # Debug
                    # yield "data: Binning process complete. Stopping stream.\n\n"
                    return  # Stop streaming

        with open(BINNING_LOG_PATH, "r") as f:
            # Seek to the end of the file
            f.seek(0, 2)
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line.strip()}\n\n"
                    if "Binning complete" in line:  # Check for completion keyword
                        update_process_status("binning",Status.READY.value)
                        return  # Stop streaming

                else:
                    time.sleep(1)  # Wait for new lines to be written

    return Response(generate(), content_type="text/event-stream")

def binning_status():
    if os.path.exists(BINNING_LOG_PATH):
        update_process_status("binning",Status.PENDING.value)
        try:
            with open(BINNING_LOG_PATH, "r") as f:
                for line in f:
                    if "Binning complete" in line:  # Check for completion keyword\
                        update_process_status("binning",Status.READY.value)
        except Exception as e:
            update_process_status("binning",Status.ERROR.value)
    else:
        update_process_status("binning",Status.PENDING.value)




@app.route("/instrumentview_status")
def instrumentview_status():
    if os.path.exists(INSTRUMENTVIEW_PATH):
        update_process_status('instrumentview',Status.READY.value)
        return jsonify({"status": Status.READY.value})
    else:
        update_process_status('instrumentview',Status.PENDING.value)        
        return jsonify({"status": Status.PENDING.value})


@app.route("/get_instrumentview")
def get_instrumentview():
    if os.path.exists(INSTRUMENTVIEW_PATH):
        with open(INSTRUMENTVIEW_PATH, "r") as f:
            return f.read()
    else:
        return "Instrument View is not ready yet.", 404


@app.route("/spotfinding_status")
def spotfinding_status():
    if os.path.exists(HISTOGRAMS_JSON_PATH):
        update_process_status('spotfinding',Status.READY.value) 
        return jsonify({"status": Status.READY.value})
    else:
        update_process_status('spotfinding',Status.PENDING.value) 
        return jsonify({"ready": Status.PENDING.value})


@app.route("/get_spotfinding")
def get_spotfinding():
    if os.path.exists(HISTOGRAMS_JSON_PATH):
        try:
            with open(HISTOGRAMS_JSON_PATH, "r") as f:
                histograms = json.load(f)

            # Deserialize binary data if necessary
            for uid, histogram in histograms.items():
                for trace in histogram["data"]:
                    if (
                        "x" in trace
                        and isinstance(trace["x"], dict)
                        and "bdata" in trace["x"]
                    ):
                        trace["x"] = deserialize_binary_data(trace["x"])
                    if (
                        "y" in trace
                        and isinstance(trace["y"], dict)
                        and "bdata" in trace["y"]
                    ):
                        trace["y"] = deserialize_binary_data(trace["y"])

            return jsonify({"histograms": histograms})
        except:
            return "No spotfinding results found", 404

@app.route("/data/interim/<path:filename>")
def serve_interim_data(filename):
    return send_from_directory(
        "/Users/aaronfinke/nmx_workflow/nmx_workflow/data/interim", filename
    )

@app.route("/integration_status")
def integration_status():
    if os.path.exists(INTEGRATION_JSON_PATH):
        update_process_status('integration',Status.READY.value) 
        return jsonify({"status": Status.READY.value})
    else:
        update_process_status('integration',Status.PENDING.value) 
        return jsonify({"status": Status.PENDING.value})


@app.route("/get_integration")
def get_integration():
    if os.path.exists(INTEGRATION_JSON_PATH):
        try:
            with open(INTEGRATION_JSON_PATH, "r") as f:
                integrations = json.load(f)

            # Deserialize binary data if necessary
            for uid, integration in integrations.items():
                for name in ["fig_res", "fig_img"]:
                    for trace in integration[name]["data"]:
                        if (
                            "x" in trace
                            and isinstance(trace["x"], dict)
                            and "bdata" in trace["x"]
                        ):
                            trace["x"] = deserialize_binary_data(trace["x"])
                        if (
                            "y" in trace
                            and isinstance(trace["y"], dict)
                            and "bdata" in trace["y"]
                        ):
                            trace["y"] = deserialize_binary_data(trace["y"])

            return jsonify({"integration": integrations})
        except:
            return "No integration data found", 404


@app.route("/indexing_status")
def indexing_status():
    if os.path.exists(INDEXING_JSON_PATH):
        update_process_status('indexing',Status.READY.value) 
        return jsonify({"status": Status.READY.value})
    else:
        update_process_status('indexing',Status.PENDING.value) 
        return jsonify({"status": Status.PENDING.value})


@app.route("/get_indexing")
def get_indexing():
    INDEXING_JSON_PATH = INTERIM_DATA_DIR / "indexing.json"
    if os.path.exists(INDEXING_JSON_PATH):
        try:
            with open(INDEXING_JSON_PATH, "r") as f:
                indexing_results = json.load(f)
            return jsonify({"indexing_results": indexing_results})
        except:
            return "No indexing results found", 404


@app.route("/refine_status")
def refine_status():
    if os.path.exists(REFINE_JSON_PATH):
        update_process_status('refinement',Status.READY.value) 
        return jsonify({"status": Status.READY.value})
    else:
        update_process_status('refinement',Status.PENDING.value) 
        return jsonify({"status": Status.PENDING.value})


@app.route("/get_refine")
def get_refine():
    if os.path.exists(REFINE_JSON_PATH):
        try:
            with open(REFINE_JSON_PATH, "r") as f:
                refine_results = json.load(f)
            return jsonify({"refine_results": refine_results})
        except:
            return "No refine results found", 404


def deserialize_binary_data(data):
    """Deserialize binary data from the JSON."""
    import base64
    import numpy as np

    bdata = base64.b64decode(data["bdata"])
    dtype = np.dtype(data["dtype"])
    return np.frombuffer(bdata, dtype=dtype).tolist()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
