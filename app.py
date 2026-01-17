import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import re

st.set_page_config(page_title="SensorFlow Pro", layout="wide")
st.title("SensorFlow Pro – Smart Sensor Data Analyzer")

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Cleaning", "Stats", "Box Plot", "Time Plot", "Histogram", "Compare", "ΔR/R", "Export"
])

# -------------------------------------------------
# GLOBAL: Load & Clean (Shared)
# -------------------------------------------------
@st.cache_data
def load_and_clean(uploaded_file):
    if not uploaded_file:
        return None, None, None, None, False

    try:
        df_raw = pd.read_csv(uploaded_file, dtype=str) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, dtype=str)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None, None, None, None, False

    # Check for time column
    time_candidates = [col for col in df_raw.columns if re.search(r'time|timestamp|date|secs', col, re.I)]
    has_time_col = bool(time_candidates)
    time_col = time_candidates[0] if time_candidates else None
    sensor_cols = [c for c in df_raw.columns if c != time_col] if time_col else df_raw.columns.tolist()

    cleaned_rows = []
    skipped_rows = []

    for idx, row in df_raw.iterrows():
        # Validate time column if present
        if has_time_col:
            t_raw = str(row[time_col]).strip()
            if t_raw in ['', 'nan', 'NaN', 'NA', '--NA--']:
                skipped_rows.append(f"Row {idx}: Empty time")
                continue
            try:
                float(t_raw)
            except:
                skipped_rows.append(f"Row {idx}: Non-numeric time '{t_raw}'")
                continue

        valid_row = True
        for col in sensor_cols:
            v = str(row[col]).strip()
            if v in ['', 'nan', 'NaN', 'NA', '--NA--']:
                continue
            try:
                float(v)
            except:
                skipped_rows.append(f"Row {idx}: Non-numeric in {col}")
                valid_row = False
                break
        if valid_row:
            cleaned_rows.append(row)

    if not cleaned_rows:
        st.error("No valid data.")
        return None, None, None, None, False

    df_clean = pd.DataFrame(cleaned_rows)
    for col in sensor_cols:
        df_clean[col] = pd.to_numeric(df_clean[col].replace(['--NA--', 'NA'], np.nan), errors='coerce')
    df_clean.dropna(axis=1, how='all', inplace=True)
    sensor_cols = [c for c in df_clean.columns if c != time_col] if time_col else df_clean.columns.tolist()

    if has_time_col:
        df_clean[time_col] = pd.to_numeric(df_clean[time_col], errors='coerce')
        df_clean = df_clean.dropna(subset=[time_col])
        df_clean = df_clean.sort_values(time_col)
        df_clean = df_clean.set_index(time_col, drop=True)
    else:
        # Use default integer index
        df_clean.index.name = 'Index'

    return df_clean, sensor_cols, time_col, skipped_rows, has_time_col

# -------------------------------------------------
# MAIN UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"], key="main")
df_clean, sensor_cols, time_col, skipped_rows, has_time_col = load_and_clean(uploaded_file)

if df_clean is None:
    st.stop()

# Drop constant columns
zero_var = df_clean.columns[df_clean.nunique() <= 1]
df_work = df_clean.copy()
if len(zero_var) > 0 and st.checkbox("Drop constant columns", value=True, key="drop_const"):
    df_work = df_work.loc[:, df_work.nunique() > 1]
    sensor_cols = [c for c in df_work.columns]

# -------------------------------------------------
# TAB 1: Cleaning
# -------------------------------------------------
with tab1:
    st.success(f"Loaded {len(df_clean)} rows | {len(sensor_cols)} columns")
    if not has_time_col:
        st.info("No time column detected. Using row index as x-axis.")
    if skipped_rows:
        with st.expander(f"Skipped {len(skipped_rows)} invalid rows"):
            for msg in skipped_rows[:20]:
                st.write(msg)
    if len(zero_var) > 0:
        st.warning(f"Constant columns removed: {', '.join(zero_var)}")

# -------------------------------------------------
# TAB 2: Stats
# -------------------------------------------------
with tab2:
    st.subheader("Statistical Summary")
    if sensor_cols:
        stats_df = pd.DataFrame({
            'Min': df_work.min(),
            'Max': df_work.max(),
            'Mean': df_work.mean(),
            'Std': df_work.std(),
            'Var': df_work.var(),
            'Skew': df_work.apply(lambda x: stats.skew(x.dropna()) if len(x.dropna()) > 2 else np.nan),
            'Kurt': df_work.apply(lambda x: stats.kurtosis(x.dropna()) if len(x.dropna()) > 2 else np.nan)
        }).round(4)

        def highlight_max(s):
            is_max = s == s.max()
            return ['font-weight: bold; color: #d32f2f' if v else '' for v in is_max]
        st.dataframe(stats_df.style.apply(highlight_max, axis=1), use_container_width=True)
        st.download_button("Download Stats", stats_df.to_csv().encode(), "stats.csv", "text/csv")
    else:
        st.info("No sensor data.")

# -------------------------------------------------
# TAB 3: Box Plot
# -------------------------------------------------
with tab3:
    st.subheader("Box Plot")
    if sensor_cols:
        col = st.selectbox("Select column", sensor_cols, key="box")
        fig = px.box(df_work, y=col, points="outliers", title=f"Box Plot: {col}")
        fig.update_layout(yaxis=dict(tickformat=".1f", separatethousands=True))
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# TAB 4: Time Plot
# -------------------------------------------------
with tab4:
    st.subheader("Time Series + Rolling Mean")
    if sensor_cols:
        cols = st.multiselect("Select columns", sensor_cols, default=sensor_cols[:2])
        win = st.slider("Rolling window (points)", 1, 100, 10, key="time_roll")
        roll = df_work[cols].rolling(win).mean()

        fig = go.Figure()
        for c in cols:
            fig.add_trace(go.Scatter(x=df_work.index, y=df_work[c], name=c))
            fig.add_trace(go.Scatter(x=roll.index, y=roll[c], name=f"{c} (roll)", line=dict(dash='dot')))
        fig.update_layout(
            xaxis_title="Time (s)" if has_time_col else "Index",
            yaxis_title="Resistance (Ω)",
            yaxis=dict(tickformat=".1f", separatethousands=True),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# TAB 5: Histogram
# -------------------------------------------------
with tab5:
    st.subheader("Distribution (Histogram + KDE)")

    if sensor_cols:
        col = st.selectbox("Select column", sensor_cols, key="hist")
        fig = px.histogram(
            df_work, x=col, nbins=50, marginal="violin",
            histnorm='probability density',
            title=f"Distribution of {col}"
        )
        fig.update_layout(yaxis=dict(tickformat=".1f"))
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# TAB 6: Compare Two Files (ΔR/R Transformation)
# -------------------------------------------------
with tab6:
    st.subheader("Compare Two Files: ΔR/R = (R - R₀) / R₀")

    # File upload
    c1, c2 = st.columns(2)
    with c1:
        f1 = st.file_uploader("File 1 (e.g., Control)", type=["csv", "xlsx"], key="c1")
    with c2:
        f2 = st.file_uploader("File 2 (e.g., Drug)", type=["csv", "xlsx"], key="c2")

    if f1 and f2:
        # Load and clean both files
        df1, sensor_cols1, time_col1, _, has_time_col1 = load_and_clean(f1)
        df2, sensor_cols2, time_col2, _, has_time_col2 = load_and_clean(f2)

        if df1 is None or df2 is None:
            st.error("Failed to load one or both files.")
            st.stop()

        # Check if both files have consistent time column usage
        if has_time_col1 != has_time_col2:
            st.error("Files are inconsistent: one has a time column, the other does not.")
            st.stop()

        # Find common sensor columns
        common = list(set(df1.columns) & set(df2.columns))
        if not common:
            st.error("No common sensor columns found between the files.")
            st.stop()

        # --- Baseline Method Selection ---
        baseline_method = st.radio(
            "Choose Baseline (R₀) Method",
            options=["Mean of first N points", "Single point at specific time"],
            horizontal=True,
            key="comp_drr_baseline_method"
        )

        # --- Option 1: Mean of first N points ---
        if baseline_method == "Mean of first N points":
            max_points = min(len(df1), len(df2), 100)
            baseline_points = st.slider(
                "Number of initial points to average (R₀)",
                min_value=1,
                max_value=max_points,
                value=min(10, max_points),
                key="comp_drr_base_points"
            )
            r0_df1 = df1.iloc[:baseline_points].mean()
            r0_df2 = df2.iloc[:baseline_points].mean()
            st.info(f"R₀ = mean of first **{baseline_points}** points for both files")

        # --- Option 2: Single point at specific time ---
        else:
            if has_time_col1:
                st.write("Enter the **exact time (s)** where the baseline point is located:")
                col1, col2 = st.columns([3, 1])
                with col1:
                    # Use the overlapping time range for the input
                    min_time = max(df1.index.min(), df2.index.min())
                    max_time = min(df1.index.max(), df2.index.max())
                    baseline_time = st.number_input(
                        "Baseline time (s)",
                        min_value=float(min_time),
                        max_value=float(max_time),
                        value=float(min_time),
                        step=0.1,
                        format="%.3f",
                        key="comp_drr_single_time"
                    )
                with col2:
                    st.caption("Time axis")

                # Find closest time point for both files
                time_diffs1 = np.abs(df1.index - baseline_time)
                closest_pos1 = time_diffs1.argmin()
                closest_idx1 = df1.index[closest_pos1]
                actual_time1 = closest_idx1
                r0_df1 = df1.loc[closest_idx1]

                time_diffs2 = np.abs(df2.index - baseline_time)
                closest_pos2 = time_diffs2.argmin()
                closest_idx2 = df2.index[closest_pos2]
                actual_time2 = closest_idx2
                r0_df2 = df2.loc[closest_idx2]

                st.info(f"File 1: R₀ at t = {actual_time1:.3f} s | File 2: R₀ at t = {actual_time2:.3f} s (closest points)")
            else:
                st.write("Enter the **index** where the baseline point is located:")
                col1, col2 = st.columns([3, 1])
                with col1:
                    # Use the overlapping index range
                    min_idx = 0
                    max_idx = min(len(df1) - 1, len(df2) - 1)
                    baseline_idx = st.number_input(
                        "Baseline index",
                        min_value=int(min_idx),
                        max_value=int(max_idx),
                        value=int(min_idx),
                        step=1,
                        key="comp_drr_single_idx"
                    )
                with col2:
                    st.caption("Index axis")

                # Use the specified index
                r0_df1 = df1.iloc[baseline_idx]
                r0_df2 = df2.iloc[baseline_idx]
                st.info(f"R₀ = value at index **{baseline_idx}** for both files")

        # --- Compute ΔR/R Transformation ---
        df_drr1 = (df1 - r0_df1) / r0_df1
        df_drr2 = (df2 - r0_df2) / r0_df2
        df_drr1 = df_drr1.dropna(how='all')  # Drop rows where all are NaN
        df_drr2 = df_drr2.dropna(how='all')

        # --- Plotting ---
        plot_cols = st.multiselect(
            "Select channels to plot",
            common,
            default=common[:min(2, len(common))],
            key="comp_drr_cols"
        )
        win = st.slider(
            "Rolling mean window (points)",
            1, 100, 10,
            key="comp_drr_roll"
        )

        # Compute rolling means
        roll1 = df_drr1[plot_cols].rolling(win, min_periods=1, center=True).mean()
        roll2 = df_drr2[plot_cols].rolling(win, min_periods=1, center=True).mean()

        # Create plot
        fig = go.Figure()
        for c in plot_cols:
            # File 1 traces
            fig.add_trace(go.Scatter(
                x=df_drr1.index, y=df_drr1[c],
                name=f"{f1.name.split('.')[0]}: {c} (ΔR/R)",
                mode='lines',
                line=dict(width=2)
            ))
            fig.add_trace(go.Scatter(
                x=roll1.index, y=roll1[c],
                name=f"{f1.name.split('.')[0]}: {c} (roll)",
                line=dict(dash='dot', width=1.5),
                mode='lines'
            ))
            # File 2 traces
            fig.add_trace(go.Scatter(
                x=df_drr2.index, y=df_drr2[c],
                name=f"{f2.name.split('.')[0]}: {c} (ΔR/R)",
                mode='lines',
                line=dict(width=2, dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=roll2.index, y=roll2[c],
                name=f"{f2.name.split('.')[0]}: {c} (roll)",
                line=dict(dash='dot', width=1.5),
                mode='lines'
            ))

        fig.update_layout(
            title=f"ΔR/R Comparison: {f1.name} vs {f2.name}",
            xaxis_title="Time (s)" if has_time_col1 else "Index",
            yaxis_title="ΔR/R",
            yaxis=dict(tickformat=".4f"),
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="simple_white",
            margin=dict(l=40, r=20, t=60, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Download ΔR/R Data ---
        combined_df = pd.DataFrame()
        for c in common:
            combined_df[f"{f1.name.split('.')[0]}_{c}_ΔR/R"] = df_drr1[c]
            combined_df[f"{f2.name.split('.')[0]}_{c}_ΔR/R"] = df_drr2[c]
        combined_df.index.name = 'Time' if has_time_col1 else 'Index'
        csv = combined_df.to_csv().encode()
        st.download_button(
            label="Download ΔR/R Data (CSV)",
            data=csv,
            file_name="compare_delta_r_over_r.csv",
            mime="text/csv"
        )

        # --- Show R₀ values ---
        with st.expander("View R₀ (Baseline) Values"):
            r0_combined = pd.DataFrame({
                f"{f1.name.split('.')[0]} R₀": r0_df1,
                f"{f2.name.split('.')[0]} R₀": r0_df2
            })
            st.dataframe(r0_combined.style.format("{:.4f}"))
    else:
        st.info("Please upload both files to compare.")

# -------------------------------------------------
# TAB 7: ΔR/R Transformation (Enhanced Baseline Options)
# -------------------------------------------------
with tab7:
    st.subheader("ΔR/R = (R - R₀) / R₀  →  All Channels")
    if sensor_cols:
        # --- Baseline Method Selection ---
        baseline_method = st.radio(
            "Choose Baseline (R₀) Method",
            options=["Mean of first N points", "Single point at specific time"],
            horizontal=True,
            key="drr_baseline_method"
        )

        # --- Option 1: Mean of first N points ---
        if baseline_method == "Mean of first N points":
            baseline_points = st.slider(
                "Number of initial points to average (R₀)",
                min_value=1,
                max_value=min(100, len(df_work)),
                value=min(10, len(df_work)),
                key="drr_base_points"
            )
            r0 = df_work.iloc[:baseline_points].mean()
            st.info(f"R₀ = mean of first **{baseline_points}** points")

        # --- Option 2: Single point at specific time ---
        else:
            if has_time_col:
                st.write("Enter the **exact time (s)** where the baseline point is located:")
                col1, col2 = st.columns([3, 1])
                with col1:
                    baseline_time = st.number_input(
                        "Baseline time (s)",
                        min_value=float(df_work.index.min()),
                        max_value=float(df_work.index.max()),
                        value=float(df_work.index.min()),
                        step=0.1,
                        format="%.3f",
                        key="drr_single_time"
                    )
                with col2:
                    st.caption("Time axis")

                # Find closest time point
                time_diffs = np.abs(df_work.index - baseline_time)
                closest_pos = time_diffs.argmin()
                closest_idx = df_work.index[closest_pos]
                r0 = df_work.loc[closest_idx]
                actual_time = closest_idx
                st.info(f"R₀ = value at **t = {actual_time:.3f} s** (closest point)")
            else:
                st.write("Enter the **index** where the baseline point is located:")
                col1, col2 = st.columns([3, 1])
                with col1:
                    baseline_idx = st.number_input(
                        "Baseline index",
                        min_value=0,
                        max_value=len(df_work) - 1,
                        value=0,
                        step=1,
                        key="drr_single_idx"
                    )
                with col2:
                    st.caption("Index axis")

                r0 = df_work.iloc[baseline_idx]
                st.info(f"R₀ = value at **index = {baseline_idx}**")

        # --- Apply ΔR/R Transformation ---
        df_drr = (df_work - r0) / r0
        df_drr = df_drr.dropna(how='all')  # Drop rows where all are NaN

        # --- Plotting ---
        plot_cols = st.multiselect(
            "Select channels to plot",
            sensor_cols,
            default=sensor_cols[:min(2, len(sensor_cols))],
            key="drr_cols"
        )
        win = st.slider(
            "Rolling mean window (points)",
            1, 100, 10,
            key="drr_roll"
        )
        roll = df_drr[plot_cols].rolling(win, min_periods=1).mean()

        fig = go.Figure()
        for c in plot_cols:
            fig.add_trace(go.Scatter(
                x=df_drr.index, y=df_drr[c],
                name=f"{c} (ΔR/R)",
                mode='lines'
            ))
            fig.add_trace(go.Scatter(
                x=roll.index, y=roll[c],
                name=f"{c} (roll)",
                line=dict(dash='dot', width=2),
                mode='lines'
            ))

        fig.update_layout(
            title="ΔR/R Transformation (Normalized Sensor Response)",
            xaxis_title="Time (s)" if has_time_col else "Index",
            yaxis_title="ΔR/R",
            yaxis=dict(tickformat=".4f"),
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Download ΔR/R Data ---
        csv = df_drr.to_csv().encode()
        st.download_button(
            label="Download ΔR/R Data",
            data=csv,
            file_name="delta_r_over_r.csv",
            mime="text/csv"
        )

        # --- Optional: Show R₀ values ---
        with st.expander("View R₀ (Baseline) Values"):
            r0_display = r0.to_frame(name="R₀ (Baseline)")
            st.dataframe(r0_display.style.format("{:.4f}"))

    else:
        st.info("No sensor data available.")

# -------------------------------------------------
# TAB 8: Export (CSV + Plot Images)
# -------------------------------------------------
with tab8:
    st.subheader("Export Data & Plots")

    # ------------------------------------------------------------------
    # 1. CSV exports
    # ------------------------------------------------------------------
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            label="Cleaned Data (CSV)",
            data=df_work.to_csv().encode(),
            file_name="cleaned_data.csv",
            mime="text/csv"
        )
    with c2:
        st.download_button(
            label="Raw Valid Rows (CSV)",
            data=df_clean.to_csv().encode(),
            file_name="raw_valid.csv",
            mime="text/csv"
        )
    with c3:
        try:
            st.download_button(
                label="Statistics (CSV)",
                data=stats_df.to_csv().encode(),
                file_name="statistics.csv",
                mime="text/csv"
            )
        except NameError:
            st.download_button(
                label="Statistics (CSV)",
                data=b"",
                file_name="statistics.csv",
                mime="text/csv",
                disabled=True
            )
            st.caption("Visit the **Stats** tab first")

    st.markdown("---")

    # ------------------------------------------------------------------
    # 2. Plot-image export
    # ------------------------------------------------------------------
    st.subheader("Export Plot as PNG")

    plot_type = st.selectbox(
        "Choose plot type",
        options=["Box Plot", "Time Series + Rolling Mean", "ΔR/R Transformation"],
        key="export_plot_type"
    )

    # ---------- Helper: generate the figure based on selection ----------
    def get_figure():
        if plot_type == "Box Plot":
            col = st.selectbox("Sensor", sensor_cols, key="export_box_col")
            fig = px.box(df_work, y=col, points="outliers",
                         title=f"Box Plot – {col}")
            fig.update_layout(yaxis=dict(tickformat=".1f"))

        elif plot_type == "Time Series + Rolling Mean":
            cols = st.multiselect("Sensors", sensor_cols,
                                  default=sensor_cols[:2],
                                  key="export_time_cols")
            win = st.slider("Rolling window (points)", 1, 100, 10,
                            key="export_time_roll")
            roll = df_work[cols].rolling(win).mean()

            fig = go.Figure()
            for c in cols:
                fig.add_trace(go.Scatter(x=df_work.index, y=df_work[c],
                                         name=c, mode='lines'))
                fig.add_trace(go.Scatter(x=roll.index, y=roll[c],
                                         name=f"{c} (roll)",
                                         line=dict(dash='dot'), mode='lines'))
            fig.update_layout(
                xaxis_title="Time (s)" if has_time_col else "Index",
                yaxis_title="Resistance (Ω)",
                yaxis=dict(tickformat=".1f", separatethousands=True),
                hovermode='x unified',
                title="Time Series + Rolling Mean"
            )

        else:  # ΔR/R Transformation
            # Baseline method
            baseline_method = st.radio(
                "Baseline (R₀) method",
                ["Mean of first N points", "Single point at specific time"],
                horizontal=True,
                key="export_drr_method"
            )

            if baseline_method == "Mean of first N points":
                n = st.slider("Points to average", 1, min(100, len(df_work)),
                              min(10, len(df_work)), key="export_drr_n")
                r0 = df_work.iloc[:n].mean()
            else:
                if has_time_col:
                    baseline_time = st.number_input(
                        "Baseline time (s)",
                        min_value=float(df_work.index.min()),
                        max_value=float(df_work.index.max()),
                        value=float(df_work.index.min()),
                        step=0.1,
                        format="%.3f",
                        key="export_drr_time"
                    )
                    time_diffs = np.abs(df_work.index - baseline_time)
                    closest_pos = time_diffs.argmin()
                    closest_idx = df_work.index[closest_pos]
                    r0 = df_work.loc[closest_idx]
                else:
                    idx = st.slider(
                        "Baseline index",
                        min_value=0,
                        max_value=len(df_work) - 1,
                        value=0,
                        key="export_drr_idx"
                    )
                    r0 = df_work.iloc[idx]

            df_drr = (df_work - r0) / r0

            cols = st.multiselect("Sensors", sensor_cols,
                                  default=sensor_cols[:min(2, len(sensor_cols))],
                                  key="export_drr_cols")
            win = st.slider("Rolling window (points)", 1, 100, 10,
                            key="export_drr_roll")
            roll = df_drr[cols].rolling(win, min_periods=1).mean()

            fig = go.Figure()
            for c in cols:
                fig.add_trace(go.Scatter(x=df_drr.index, y=df_drr[c],
                                         name=f"{c} (ΔR/R)", mode='lines'))
                fig.add_trace(go.Scatter(x=roll.index, y=roll[c],
                                         name=f"{c} (roll)",
                                         line=dict(dash='dot'), mode='lines'))
            fig.update_layout(
                title="ΔR/R Transformation",
                xaxis_title="Time (s)" if has_time_col else "Index",
                yaxis_title="ΔR/R",
                yaxis=dict(tickformat=".4f"),
                hovermode='x unified'
            )

        # Common styling
        fig.update_layout(
            template="simple_white",
            font=dict(size=12),
            margin=dict(l=40, r=20, t=60, b=40),
            height=550
        )
        return fig

    # ------------------------------------------------------------------
    # 3. Render the figure + PNG download button
    # ------------------------------------------------------------------
    if sensor_cols:
        fig = get_figure()
        st.plotly_chart(fig, use_container_width=True)

        # Convert to PNG (high DPI)
        img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
        st.download_button(
            label="Download Plot as PNG",
            data=img_bytes,
            file_name=f"{plot_type.lower().replace(' ', '_')}.png",
            mime="image/png"
        )
    else:
        st.info("No sensor columns available for plotting.")