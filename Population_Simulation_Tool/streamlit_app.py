# === streamlit_app.py ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import plotly.graph_objs as go

from simulation_core import run_simulation, Event

st.set_page_config(page_title="Population Simulation", layout="wide")
st.title("Population & Demographics Simulator")

st.sidebar.header("Simulation Settings")
init_pop = st.sidebar.slider("Initial Population", 100, 500000, 10000, step=100)
n_years = st.sidebar.slider("Simulation Years", 10, 30, 20)

base_child_support = st.sidebar.slider("Child Support", 0.0, 0.5, 0.0, step=0.05)
base_healthcare_quality = st.sidebar.slider("Healthcare Quality", 0.0, 1.0, 0.8, step=0.05)
urban_ratio = st.sidebar.slider("Urban Population Ratio", 0.0, 1.0, 0.6, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Education Impact Curve Control")
st.sidebar.markdown("Adjust the sliders below to change the shape of the education impact curve.")

n_anchors = 8
anchor_years = np.linspace(0, n_years - 1, n_anchors).astype(int)

if "edu_impact_values" not in st.session_state or len(st.session_state.edu_impact_values) != n_anchors:
    st.session_state.edu_impact_values = [0.5] * n_anchors

updated_vals = []
for i, year in enumerate(anchor_years):
    val = st.sidebar.slider(
        f"Impact in Year {year}", 0.0, 1.0,
        value=st.session_state.edu_impact_values[i],
        step=0.05,
        key=f"edu_slider_{year}"
    )
    updated_vals.append(val)

st.session_state.edu_impact_values = updated_vals

interpolator = interp1d(anchor_years, updated_vals, kind="linear", bounds_error=False, fill_value="extrapolate")
edu_impact_series = interpolator(np.arange(n_years)).tolist()

plot_curve = go.Figure(data=go.Scatter(x=np.arange(n_years), y=edu_impact_series, mode="lines"))
plot_curve.update_layout(
    width=350,
    height=250,
    margin=dict(l=10, r=10, t=30, b=10),
    title="Education Impact Preview",
    xaxis_title="Year", yaxis_title="Impact",
    yaxis=dict(range=[0, 1]),
    dragmode=False,
    hovermode=False,
    showlegend=False,
    modebar=dict(remove=["zoom", "pan", "select", "lasso", "zoomIn", "zoomOut", "autoScale", "resetScale"])
)
st.sidebar.plotly_chart(plot_curve, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Timed Events")
events = []

if st.sidebar.checkbox("Enable Baby Boom"):
    bb_start = st.sidebar.number_input("Baby Boom Start Year", 0, n_years - 1, value=min(20, n_years - 1))
    bb_end = st.sidebar.number_input("Baby Boom End Year", bb_start, n_years, value=min(40, n_years))
    bb_support = st.sidebar.slider("Baby Boom Child Support", 0.0, 0.5, 0.2, step=0.05)
    events.append(Event("Baby Boom", bb_start, bb_end, {"child_support": bb_support}))

if st.sidebar.checkbox("Enable Immigration Wave"):
    im_start = st.sidebar.number_input("Immigration Start Year", 0, n_years - 1, value=min(60, n_years - 1))
    im_end = st.sidebar.number_input("Immigration End Year", im_start, n_years, value=min(80, n_years))
    im_inflow = st.sidebar.slider("Annual Immigration Count", 0, 1000, 200, step=50)
    events.append(Event("Immigration Wave", im_start, im_end, {"immigration_inflow": im_inflow}))

if st.sidebar.button("Run Simulation"):
    st.subheader("Running Simulation...")
    with st.spinner("Please wait..."):
        results = run_simulation(
            init_pop_count=init_pop,
            n_years=n_years,
            urban_ratio=urban_ratio,
            base_child_support=base_child_support,
            base_education_impact=edu_impact_series,
            base_healthcare_quality=base_healthcare_quality,
            events=events
        )

        st.success("Simulation completed!")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Population Size Over Time")
            st.line_chart(results["pop_sizes"])
            st.markdown("### Dependency Ratio Over Time")
            st.line_chart(results["dependency_ratios"])
            st.markdown("### Average Education Level Over Time")
            st.line_chart(results["avg_education"])

        with col2:
            st.markdown("### Urban vs Rural Population")
            urban_rural_df = pd.DataFrame({
                "Urban": results["urban_population"],
                "Rural": results["rural_population"]
            })
            st.line_chart(urban_rural_df)

            st.markdown("### Childbearing Age Distribution")
            fig, ax = plt.subplots()
            ax.hist(results["child_bearing_ages"], bins=20, color='skyblue')
            ax.set_xlabel("Age")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

            st.markdown("### Final Age Distribution")
            fig, ax = plt.subplots()
            pd.Series(results["final_ages"]).hist(bins=30, ax=ax, color='salmon')
            ax.set_xlabel("Age")
            ax.set_ylabel("Count")
            st.pyplot(fig)

else:
    st.info("Adjust parameters in the sidebar and click 'Run Simulation' to begin.")
