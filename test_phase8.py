import requests
import base64
import time
import os

SERVER_URL = "http://localhost:5000"

BASE_DIR = r"C:\Medha\TumorGuard AI\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_005"
FLAIR = os.path.join(BASE_DIR, "BraTS20_Training_005_flair.nii")
T1CE = os.path.join(BASE_DIR, "BraTS20_Training_005_t1ce.nii")
T2 = os.path.join(BASE_DIR, "BraTS20_Training_005_t2.nii")

def log(msg):
    print(f"[*] {msg}")

def test_health():
    log("TEST: /health")
    resp = requests.get(f"{SERVER_URL}/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data['status'] == 'ok'
    assert data['model_loaded'] is True
    assert data['training_stats_loaded'] is True
    log("Health check passed.")

def test_missing_modality():
    log("TEST: Missing Modality (T2)")
    files = {
        'flair': open(FLAIR, 'rb'),
        't1ce': open(T1CE, 'rb'),
    }
    resp = requests.post(f"{SERVER_URL}/predict", files=files)
    assert resp.status_code == 400
    assert "Missing required modalities" in resp.json()['error']
    log("Missing modality check passed.")

def test_wrong_extension():
    log("TEST: Wrong extension")
    files = {
        'flair': ('test.jpg', b'fake data'),
        't1ce': open(T1CE, 'rb'),
        't2': open(T2, 'rb')
    }
    resp = requests.post(f"{SERVER_URL}/predict", files=files)
    assert resp.status_code == 400
    assert "Invalid file type" in resp.json()['error']
    log("Wrong extension check passed.")

def test_empty_request():
    log("TEST: Empty request")
    resp = requests.post(f"{SERVER_URL}/predict")
    assert resp.status_code == 400
    log("Empty request check passed.")

def test_valid_predict():
    log("TEST: Valid /predict")
    files = {
        'flair': open(FLAIR, 'rb'),
        't1ce': open(T1CE, 'rb'),
        't2': open(T2, 'rb')
    }
    start = time.time()
    resp = requests.post(f"{SERVER_URL}/predict", files=files)
    duration = time.time() - start
    
    assert resp.status_code == 200
    data = resp.json()
    
    log(f"Inference time: {duration:.2f} seconds")
    log(f"Response size: {len(resp.content) / 1024:.2f} KB")
    
    assert "slice_index" in data
    assert "familiarity_score" in data
    assert "predictive_entropy" in data
    assert "uncertainty_score" in data
    assert "reliability_score" in data
    assert "verdict" in data
    assert "images" in data
    assert "disclaimer" in data
    
    # Base64 validation
    for key, b64_str in data['images'].items():
        assert b64_str.startswith("data:image/png;base64,")
        raw_b64 = b64_str.split(",")[1]
        decoded = base64.b64decode(raw_b64)
        # PNG signature
        assert decoded.startswith(b'\x89PNG\r\n\x1a\n')
        log(f"Image '{key}' validated (PNG signature ok, size: {len(decoded)/1024:.2f} KB)")

    log("Valid /predict check passed.")

if __name__ == "__main__":
    test_health()
    test_missing_modality()
    test_wrong_extension()
    test_empty_request()
    test_valid_predict()
    print("\n[+] All backend tests passed successfully.")
