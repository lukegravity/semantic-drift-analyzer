import streamlit as st
import pandas as pd
from core.data_loader import load_screaming_frog, load_gsc, merge_data
from core.processing import compute_centroid
from core.projection import reduce_umap, centre_on_centroid
from ui.visuals import plot_radial_topical_map
from utils.logger import log
import time

st.set_page_config(page_title="Semantic Drift Analyser", layout="wide")
st.title("🧭 Semantic Drift Analyser")
st.markdown("*Visualise how your content drifts from your site's core topics*")

# --- File upload in collapsible section ---
with st.expander("📁 Upload Data", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        sf_file = st.file_uploader("Screaming Frog export (with embeddings)", type=["csv"])
    with col2:
        gsc_file = st.file_uploader("Search Console data", type=["csv"])

if sf_file and gsc_file:
    # Create a placeholder for status messages
    status_container = st.empty()
    
    # Process data with status updates
    with status_container.container():
        with st.status("Processing...", expanded=True) as status:
            st.write("Loading files...")
            sf_df = load_screaming_frog(sf_file)
            gsc_df = load_gsc(gsc_file)
            df = merge_data(sf_df, gsc_df)
            
            st.write("Computing drift metrics...")
            centroid, df = compute_centroid(df)
            
            st.write("Mapping semantic space...")
            df, reducer = reduce_umap(df)
            df, centroid_coords = centre_on_centroid(df, centroid, reducer)
            
            status.update(label="Complete!", state="complete")
    
    # Clear the status after a short delay
    time.sleep(1)
    status_container.empty()

    # --- Plot ---
    plot_radial_topical_map(df)

    # --- Zone distribution metrics ---
    st.subheader("📊 Distribution")
    
    # Calculate zones
    max_dist = df["distance_from_centre"].max()
    df["normalized_distance"] = df["distance_from_centre"] / max_dist
    
    def assign_zone(norm_dist):
        if norm_dist <= 0.25:
            return "Core"
        elif norm_dist <= 0.5:
            return "Focus"
        elif norm_dist <= 0.75:
            return "Expansion"
        else:
            return "Peripheral"
    
    df["zone"] = df["normalized_distance"].apply(assign_zone)
    
    # Calculate distribution
    zone_distribution = df["zone"].value_counts()
    zone_percentages = (zone_distribution / len(df) * 100).round(1)
    
    # Display as metrics columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Core", f"{zone_percentages.get('Core', 0)}%")
    with col2:
        st.metric("Focus", f"{zone_percentages.get('Focus', 0)}%")
    with col3:
        st.metric("Expansion", f"{zone_percentages.get('Expansion', 0)}%")
    with col4:
        st.metric("Peripheral", f"{zone_percentages.get('Peripheral', 0)}%")

    # --- Top drift pages ---
    with st.expander("⚠️ High Drift Pages", expanded=False):
        top_drift = df.nlargest(10, "SDI")[["Address", "zone", "Clicks", "SDI"]]
        top_drift.columns = ["URL", "Zone", "Clicks", "Drift"]
        top_drift["Drift"] = top_drift["Drift"].round(2)
        st.dataframe(top_drift, use_container_width=True, hide_index=True)

    # --- Export ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            "📥 Export Analysis",
            df.to_csv(index=False).encode("utf-8"),
            "semantic_drift_analysis.csv",
            "text/csv",
            use_container_width=True
        )