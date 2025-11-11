import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

def plot_radial_topical_map(df):
    st.subheader("🌐 Semantic Drift Visualisation")

    # --- Controls ---
    chart_size = st.sidebar.slider("Chart Size", 400, 1000, 700, 50)
    radius_max = 1.0
    size_scale = st.sidebar.slider("Max Bubble Size (Clicks)", 100, 5000, 400, 50)
    opacity_min = st.sidebar.slider("Minimum Bubble Opacity", 0.1, 0.8, 0.2, 0.05)
    show_labels = st.sidebar.checkbox("Show Zone Labels", value=True)

    # --- Normalise inputs ---
    df["r_norm"] = (df["distance_from_centre"] / df["distance_from_centre"].max()) * radius_max
    df["theta"] = np.linspace(0, 2 * np.pi, len(df), endpoint=False)

    # Convert polar → cartesian for plotting
    df["x_radial"] = df["r_norm"] * np.cos(df["theta"])
    df["y_radial"] = df["r_norm"] * np.sin(df["theta"])

    # Visual encodings - MUCH BIGGER DOTS
    clicks_min = df["Clicks"].fillna(0).min()
    clicks_max = df["Clicks"].fillna(0).max()
    
    if clicks_max > clicks_min:
        # Normalize clicks to 0-1 range
        df["clicks_normalized"] = (df["Clicks"].fillna(clicks_min) - clicks_min) / (clicks_max - clicks_min)
        # Create dramatic size differences - minimum 50, max controlled by slider
        df["size_scaled"] = 50 + (df["clicks_normalized"] ** 0.7) * size_scale
    else:
        df["size_scaled"] = 100
    
    # Log-scale to dampen extreme hubs and improve comparability
    df["inlink_log"] = np.log1p(df["Inlinks"].fillna(0))
    df["clicks_log"] = np.log1p(df["Clicks"].fillna(0))

    # Use 95th percentile cap instead of raw max to avoid skew
    inlink_cap = np.percentile(df["inlink_log"], 95)
    clicks_cap = np.percentile(df["clicks_log"], 95)

    inlink_norm = (df["inlink_log"] / inlink_cap).clip(0, 1)
    clicks_norm = (df["clicks_log"] / clicks_cap).clip(0, 1)

    # Weighted blend: 80% structure (inlinks), 20% engagement (clicks)
    opacity_strength = st.sidebar.slider(
        "Opacity Strength (contrast between weak/strong links)",
        1.0, 5.0, 3.0, 0.1
    )

    # Stronger nonlinear emphasis on link prominence
    link_weight = np.power(inlink_norm, 1 / opacity_strength)
    click_weight = np.power(clicks_norm, 1 / (opacity_strength * 1.5))

    # Push low-link pages further down, exaggerate top performers
    contrast_boost = np.clip(
        np.power(link_weight * 0.8 + click_weight * 0.2, 2.2),
        0, 1
    )
    # Map into 0–1 range with floor from opacity_min
    df["opacity_scaled"] = (contrast_boost * (1 - opacity_min)) + opacity_min


    # Tooltip fields
    tooltip = [
        alt.Tooltip("Address", title="URL"),
        alt.Tooltip("Clicks", title="Clicks", format=","),
        alt.Tooltip("Inlinks", title="Inlinks"),
        alt.Tooltip("SDI", title="Semantic Drift Index", format=".3f"),
        alt.Tooltip("distance_from_centre", title="Topical Distance", format=".3f"),
    ]
    
    # --- Create circle outlines ONLY (no shading) ---
    radii = [0.25, 0.5, 0.75, 1.0]
    circle_layers = []
    
    for radius in radii:
        n_points = 60
        theta_vals = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        
        circle_points = pd.DataFrame({
            'x': radius * np.cos(theta_vals),
            'y': radius * np.sin(theta_vals)
        })
        
        circle_points = pd.concat([
            circle_points, 
            circle_points.iloc[[0]]
        ], ignore_index=True)
        
        circle_layer = (
            alt.Chart(circle_points)
            .mark_line(
                strokeWidth=2,
                stroke='#95a5a6',
                opacity=0.6,
                interpolate='cardinal-closed'
            )
            .encode(
                x=alt.X('x:Q', scale=alt.Scale(domain=(-1.2, 1.2))),
                y=alt.Y('y:Q', scale=alt.Scale(domain=(-1.2, 1.2))),
                order=alt.Order('index:O')
            )
            .transform_window(
                index='row_number()'
            )
        )
        circle_layers.append(circle_layer)

    # --- Color scheme selection ---
    palette_choice = st.sidebar.selectbox(
        "Color Palette for SDI (Drift)",
        options=[
            "Viridis (uniform)",
            "Blue-Green-Yellow (semantic flow)",
            "Red-Blue Divergent (legacy)"
        ],
        index=0
    )

    if palette_choice == "Viridis (uniform)":
        color_scale = alt.Scale(scheme="viridis", domain=[0, df["SDI"].max()])
    elif palette_choice == "Blue-Green-Yellow (semantic flow)":
        color_scale = alt.Scale(
            domain=[0, df["SDI"].max()],
            range=["#007AFF", "#00C853", "#FFEA00"]
        )
    else:  # Red-Blue Divergent (legacy)
        color_scale = alt.Scale(scheme="redblue", reverse=True, domain=[0, df["SDI"].max()])

    chart = (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X("x_radial", 
                   axis=alt.Axis(title="", labels=False, ticks=False, grid=False),
                   scale=alt.Scale(domain=(-1.2, 1.2))),
            y=alt.Y("y_radial", 
                   axis=alt.Axis(title="", labels=False, ticks=False, grid=False),
                   scale=alt.Scale(domain=(-1.2, 1.2))),
            size=alt.Size("size_scaled:Q", 
                         scale=alt.Scale(range=[50, size_scale]),
                         legend=None),
            color=alt.Color("SDI:Q", scale=color_scale, title="SDI (Drift)"),
            opacity=alt.Opacity("opacity_scaled", legend=None),
            tooltip=tooltip,
        )
        .interactive()
    )

        # --- Labels (conditional based on toggle) ---
    layers_to_combine = [*circle_layers, chart]
    
    if show_labels:
        orbit_labels = pd.DataFrame({
            "label": ["Core", "Focus", "Expansion", "Peripheral"],
            "text_x": [-0.125, -0.25, -0.375, -0.5],
            "text_y": [-0.125, -0.25, -0.375, -0.5],
            "box_x1": [-0.22, -0.39, -0.56, -0.73],
            "box_y1": [-0.15, -0.27, -0.40, -0.52],
            "box_x2": [0.03, -0.11, -0.28, -0.45],
            "box_y2": [-0.10, -0.22, -0.35, -0.47],
        })

        label_backgrounds = (
            alt.Chart(orbit_labels)
            .mark_rect(fill="white", opacity=0.8, cornerRadius=3)
            .encode(
                x=alt.X("box_x1:Q", scale=alt.Scale(domain=(-1.2, 1.2))),
                y=alt.Y("box_y1:Q", scale=alt.Scale(domain=(-1.2, 1.2))),
                x2="box_x2:Q",
                y2="box_y2:Q",
            )
        )

        labels_layer = (
            alt.Chart(orbit_labels)
            .mark_text(
                align="center",
                baseline="middle",
                fontSize=11,
                fontWeight="normal",
                color="#2c3e50",
            )
            .encode(
                x=alt.X("text_x:Q", scale=alt.Scale(domain=(-1.2, 1.2))),
                y=alt.Y("text_y:Q", scale=alt.Scale(domain=(-1.2, 1.2))),
                text="label:N",
            )
        )
        
        layers_to_combine.extend([label_backgrounds, labels_layer])

    # --- Combine all layers into one chart ---
    final_chart = (
        alt.layer(*layers_to_combine)
        .properties(width=chart_size, height=chart_size)
        .configure_view(stroke=None)
        .configure_axis(grid=False, labels=False, ticks=False)
    )

    # --- Display chart, centered ---
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.altair_chart(final_chart, use_container_width=False)

    # --- Fullscreen fix (CSS only) ---
    st.markdown("""
    <style>
    [data-testid="stFullscreenFrame"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        background: white !important;
        padding: 0 !important;
    }
    [data-testid="stFullscreenFrame"] .vega-embed,
    [data-testid="stFullscreenFrame"] .vega-embed > div {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        height: 100% !important;
    }
    [data-testid="stFullscreenFrame"] .vega-embed canvas,
    [data-testid="stFullscreenFrame"] .vega-embed svg {
        display: block !important;
        margin: auto !important;
        width: min(90vh, 90vw) !important;
        height: min(90vh, 90vw) !important;
        aspect-ratio: 1 / 1 !important;
        max-width: 100% !important;
        max-height: 100% !important;
    }
    [data-testid="stFullscreenFrame"] > div:first-child {
        padding: 0 !important;
        margin: 0 auto !important;
        width: 100% !important;
        height: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)



