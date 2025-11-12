from flask import Flask, request, jsonify
import numpy as np
import base64
import cv2
import io
from PIL import Image
import insightface
import os

# Initialize Flask
app = Flask(__name__)

# Load the InsightFace ArcFace model
print("🔄 Loading ArcFace model (InsightFace)...")
model = insightface.app.FaceAnalysis(name='buffalo_l')  # buffalo_l gives best accuracy
model.prepare(ctx_id=0, det_size=(640, 640))
print("✅ ArcFace model loaded successfully")

# Utility: Convert base64 -> image (numpy)
def base64_to_image(base64_string):
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Compare faces using ArcFace embeddings
def compare_faces(img1, img2, threshold=0.35):
    faces1 = model.get(img1)
    faces2 = model.get(img2)

    if len(faces1) == 0 or len(faces2) == 0:
        return {"verified": False, "reason": "No face detected"}

    emb1 = faces1[0].normed_embedding
    emb2 = faces2[0].normed_embedding

    dist = np.linalg.norm(emb1 - emb2)
    print(f"🔍 Distance: {dist:.4f}")

    return {"verified": dist < threshold, "distance": float(dist)}

# Main verification API
@app.route("/verify", methods=["POST"])
def verify():
    try:
        data = request.get_json()
        probe_b64 = data.get("probeImage")
        ref_b64 = data.get("referenceImage")

        if not probe_b64 or not ref_b64:
            return jsonify({"verified": False, "error": "Missing images"}), 400

        img1 = base64_to_image(probe_b64)
        img2 = base64_to_image(ref_b64)

        faces1 = model.get(img1)
        faces2 = model.get(img2)

        if len(faces1) == 0 or len(faces2) == 0:
            return jsonify({"verified": False, "error": "No face detected"}), 400

        emb1 = faces1[0].normed_embedding
        emb2 = faces2[0].normed_embedding

        dist = float(np.linalg.norm(emb1 - emb2))
        verified = bool(dist < 0.8)  # threshold tuned for office lighting

        print(f"🔍 Distance: {dist:.4f}, Verified: {verified}")

        return jsonify({
            "verified": verified,
            "distance": dist
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"verified": False, "error": str(e)}), 500


@app.route("/")
def home():
    return jsonify({"message": "✅ ArcFace face verification service is running!"})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 6000))
    app.run(host="0.0.0.0", port=port)

