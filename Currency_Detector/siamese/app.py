import streamlit as st
import torch
from torchvision import transforms
import os
from PIL import Image
from siamese_model import Net  # Pulls your trained network structural layout

st.set_page_config(
    page_title="Currency Verification Engine",
    page_icon="🛡️",
    layout="centered"
)

# 🧠 HELPER FUNCTION: PREPROCESS IMAGE BYTES FROM THE WEB UPLOADER
def process_uploaded_image(uploaded_file, transform):
    """Converts a web-uploaded file stream into a standardized PyTorch matrix tensor."""
    image = Image.open(uploaded_file).convert('RGB')
    tensor = transform(image).unsqueeze(0)  # Dimensions: [1, 3, 224, 448]
    return image, tensor

# 🚀 CACHE THE MODEL WEIGHTS TO PREVENT RELOADING ON EVERY SCREEN CLICK
@st.cache_resource
def load_siamese_model(model_path="siamese_currency_model.pth"):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 🟢 Securely bind that path to your weight file name
    model_path = os.path.join(BASE_DIR, "siamese_currency_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Looking for weights at: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

# --- WEB APP FRONTEND VIEW LAYOUT ---
st.title("🛡️ Siamese Currency Authentication Engine")
st.markdown("Upload a **Trusted Reference Note** and an **Unknown Test Target Note** to scan visual textures for structural validation.")
st.write("---")

# 1. Initialize the background framework systems
transform = transforms.Compose([
    transforms.Resize((224, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    model, device = load_siamese_model()
    st.sidebar.success("✅ Neural Weights Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"❌ Weight Initialization Failure: {e}")
    st.stop()

# Adjustable Security Threshold slider on the left panel
threshold = st.sidebar.slider("Security Distance Threshold", min_value=0.5, max_value=5.0, value=2.5, step=0.1)
st.sidebar.info("💡 Distances below this wall are verified as GENUINE. Distances above are flagged as MISMATCH.")

# 2. FILE UPLOADER WIDGET INTERFACES
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Reference Note Profile")
    uploaded_ref = st.file_uploader("Upload authentic note", type=["jpg", "jpeg", "png"], key="ref")

with col2:
    st.subheader("2. Target Test Note")
    uploaded_test = st.file_uploader("Upload note to inspect", type=["jpg", "jpeg", "png"], key="test")

st.write("---")

# 3. MATCH SCAN EXECUTION STEP
if uploaded_ref is not None and uploaded_test is not None:
    # Transform web uploads into display images and vector tensors
    ref_image, ref_tensor = process_uploaded_image(uploaded_ref, transform)
    test_image, test_tensor = process_uploaded_image(uploaded_test, transform)

    # Render image layout snapshots side-by-side on the page
    display_col1, display_col2 = st.columns(2)
    with display_col1:
        st.image(ref_image, caption="Trusted Benchmark Reference Profile", use_container_width=True)
    with display_col2:
        st.image(test_image, caption="Uploaded Target Specimen", use_container_width=True)

    # SCAN ALGORITHM EXECUTION TRIGGER BUTTON
    if st.button("🚀 EXECUTE GEOMETRIC MATCH SCAN", type="primary"):
        with st.spinner("Processing deep metric vector coordinates..."):

            # Send tensor parameters to processing hardware unit
            ref_tensor = ref_tensor.to(device)
            test_tensor = test_tensor.to(device)

            # Pass tensors through the model
            with torch.no_grad():
                ref_embedding, test_embedding = model(ref_tensor, test_tensor)
                # Compute vector structural line metric distance
                distance = torch.nn.functional.pairwise_distance(ref_embedding, test_embedding).item()

        # RENDER METRIC RESULTS DATA CARD
        st.write("### 📊 Scanning Verification Report")

        metrics_col1, metrics_col2 = st.columns(2)
        metrics_col1.metric(label="Calculated Spatial Distance", value=f"{distance:.4f}")
        metrics_col2.metric(label="System Target Limit Check", value=f"{threshold:.4f}")

        # APPLICATION MATCH SYSTEM OUTPUT LOGIC
        if distance <= threshold:
            st.success("🟢 VERIFICATION SUCCESS: GENUINE PROFILE DETECTED")
            st.balloons()
            st.markdown(
                f"**System Status Analysis:** The target note shares a near-identical vector fingerprint coordinate "
                f"with the baseline note profile (Calculated distance **{distance:.4f}** falls safely under the "
                f"**{threshold:.4f}** threshold limits)."
            )
        else:
            st.error("🚨 VERIFICATION FAILURE: ANOMALY DISTANCE MISMATCH")
            st.markdown(
                f"**System Status Analysis:** **WARNING.** The network identified massive structural, layout, or alignment shifts "
                f"between sample instances. Distance **{distance:.4f}** breaches the safety threshold boundary."
            )
else:
    st.info("ℹ️ Standing by. Please upload both input image profiles above to initialize the authentication scanner.")