"""Interactive Streamlit dashboard for VerityAgent AI Defect Detector."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from verityagent_defect_detector.anomaly_model import load_model
from verityagent_defect_detector.comparison import ComparisonResult, compare_images
from verityagent_defect_detector.config import ComparisonConfig
from verityagent_defect_detector.image_io import decode_image_bytes
from verityagent_defect_detector.inference import AnomalyResult, predict_anomalies

st.set_page_config(page_title="VerityAgent | AI Defect Detector", page_icon="🔍", layout="wide")
st.markdown("""
<style>
:root{color-scheme:dark}.stApp{background:linear-gradient(145deg,#07111b 0%,#0b1724 55%,#101c2c 100%);color:#edf4fb}.stHeader{background:rgba(4,12,20,.75)}
[data-testid="stMetric"]{background:#101f2d;border:1px solid #164e70;padding:13px;border-radius:12px}
div[data-testid="stButton"]>button,div[data-testid="stDownloadButton"]>button{border:1px solid #1aa9ed;border-radius:8px}.hero{border:1px solid #147cb1;border-radius:18px;padding:24px;background:linear-gradient(115deg,rgba(17,43,66,.85),rgba(5,14,24,.88))}.muted{color:#a9bfd0}
</style>
""", unsafe_allow_html=True)
st.markdown("""<div class="hero"><div class="muted">VERITYAGENT · PYTHON AI MINI PROJECTS · PROJECT #09</div><h1>AI Defect Detector</h1><p>Compare a known-good reference, highlight candidate defects, and inspect the evidence.</p><p class="muted">OpenCV baseline · PyTorch anomaly model · Exportable inspection results</p></div>""", unsafe_allow_html=True)

mode = st.radio("Inspection method", ["Reference comparison (OpenCV)", "Trained anomaly model (PyTorch)"], horizontal=True)
result_key = "inspection_result"
if result_key not in st.session_state: st.session_state[result_key] = None


def _image_display(image: np.ndarray, *, grayscale: bool = False) -> Image.Image:
    if grayscale or image.ndim == 2: return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _show_result(result: ComparisonResult | AnomalyResult, method: str) -> None:
    report = result.to_report(); st.session_state[result_key] = {"result": result, "report": report, "method": method}
    st.success("Inspection finished. Review candidate regions before making a quality decision.")
    a,b,c,d=st.columns(4); a.metric("Candidate regions",len(result.regions)); b.metric("Affected pixels",f"{result.affected_pixel_ratio*100:.3f}%"); c.metric("Anomaly score",f"{result.anomaly_score:.5f}"); d.metric("Processing time",f"{result.processing_time_ms:.1f} ms")
    st.caption("Scores are algorithm outputs, not calibrated defect probabilities or certified quality measurements.")
    if isinstance(result, ComparisonResult):
        panels=[("Reference",result.reference_image,False),("Inspection",result.inspection_image,False),("Difference map",result.difference_map,True),("Anomaly heatmap",result.heatmap,False),("Detected regions",result.annotated_image,False),("Overlay",result.overlay,False)]
    else:
        panels=[("Inspection",result.inspection_image,False),("Anomaly map",np.clip(result.anomaly_map*255,0,255).astype(np.uint8),True),("Anomaly heatmap",result.heatmap,False),("Detected regions",result.annotated_image,False),("Overlay",result.overlay,False)]
    for start in range(0,len(panels),3):
        cols=st.columns(3)
        for column,(label,image,grayscale) in zip(cols,panels[start:start+3],strict=True):
            with column:
                st.markdown(f"**{label}**"); st.image(_image_display(image,grayscale=grayscale),use_container_width=True)
                ok,encoded=cv2.imencode(".png",image)
                if ok: st.download_button(f"Download {label.lower()}",encoded.tobytes(),file_name=f"{label.lower().replace(' ','_')}.png",mime="image/png",key=f"download_{method}_{label}")
    with st.expander("Detailed inspection report"): st.json(report)
    st.download_button("Download inspection report (JSON)",json.dumps(report,indent=2,allow_nan=False).encode("utf-8"),file_name="verityagent_defect_report.json",mime="application/json",key=f"download_report_{method}")


if mode.startswith("Reference"):
    left,right=st.columns(2)
    with left: reference_upload=st.file_uploader("Known-good reference image",type=["png","jpg","jpeg","bmp","tif","tiff","webp"],key="reference_upload")
    with right: inspection_upload=st.file_uploader("Inspection image",type=["png","jpg","jpeg","bmp","tif","tiff","webp"],key="inspection_upload")
    c1,c2,c3,c4=st.columns(4); threshold=c1.slider("Difference threshold",0,255,25); blur_kernel=c2.select_slider("Blur kernel",options=[1,3,5,7,9,11],value=3); min_area=c3.number_input("Minimum region area (px)",1,100000,100,10); color_mode=c4.selectbox("Comparison mode",["grayscale","color"])
    do_align=st.checkbox("Attempt affine image alignment (ECC)",False); normalize_illumination=st.checkbox("Normalize global illumination shift",True); run=st.button("Run reference comparison",type="primary",use_container_width=True)
    if run:
        if reference_upload is None or inspection_upload is None: st.error("Upload both a known-good reference and an inspection image.")
        else:
            try:
                result=compare_images(decode_image_bytes(reference_upload,name="reference image"),decode_image_bytes(inspection_upload,name="inspection image"),config=ComparisonConfig(threshold=threshold,blur_kernel=blur_kernel,min_area=int(min_area),align=do_align,mode=color_mode,normalize_illumination=normalize_illumination)); _show_result(result,"opencv")
            except (OSError,ValueError,TypeError,cv2.error) as exc: st.error(f"Inspection could not be completed: {exc}")
else:
    inspection_upload=st.file_uploader("Inspection image",type=["png","jpg","jpeg","bmp","tif","tiff","webp"],key="model_inspection_upload"); checkpoint_upload=st.file_uploader("Trained model checkpoint (.pt)",type=["pt","pth"],key="model_checkpoint_upload")
    min_area=st.number_input("Minimum anomaly region area (px)",1,100000,100,10,key="model_min_area"); run=st.button("Run trained-model inspection",type="primary",use_container_width=True)
    if checkpoint_upload is None: st.info("Trained-model inference is unavailable until you train a model and upload its checkpoint. OpenCV comparison works without a checkpoint.")
    if run:
        if inspection_upload is None or checkpoint_upload is None: st.error("Upload both an inspection image and a trained model checkpoint.")
        elif len(checkpoint_upload.getvalue())>100*1024*1024: st.error("Checkpoint exceeds the dashboard's 100 MB upload limit.")
        else:
            try:
                inspection=decode_image_bytes(inspection_upload,name="inspection image")
                with tempfile.TemporaryDirectory(prefix="verityagent-model-") as temporary_dir:
                    checkpoint_path=Path(temporary_dir)/"uploaded_checkpoint.pt"; checkpoint_path.write_bytes(checkpoint_upload.getvalue()); model,checkpoint_metadata=load_model(checkpoint_path,device="cpu"); safe_metadata={key:value for key,value in checkpoint_metadata.items() if key!="checkpoint_path"}; result=predict_anomalies(model,inspection,threshold=float(checkpoint_metadata["threshold"]),min_area=int(min_area),metadata=safe_metadata)
                _show_result(result,"pytorch")
            except (OSError,ValueError,TypeError,RuntimeError,cv2.error) as exc: st.error(f"Model inspection could not be completed: {exc}")

st.markdown("---"); st.caption("VerityAgent · Python AI Mini Projects · Project #09. Use representative, labeled data and human review before operational decisions.")
