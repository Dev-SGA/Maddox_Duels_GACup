import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
from streamlit_image_coordinates import streamlit_image_coordinates
from io import BytesIO
import numpy as np
from PIL import Image
from matplotlib.lines import Line2D

# ==========================
# Page Configuration
# ==========================
st.set_page_config(layout="wide", page_title="Duel Map Analysis")

st.title("Offensive Duel Map Analysis - Multiple Matches")
st.caption("Click on the icons on the pitch to play the corresponding video analysis.")

# ==========================
# Data Setup (All English labels with new data)
# ==========================
matches_data = {
    "Vs Los Angeles": [
        ("DUEL OFENSIVO WON", 65.15, 8.42, None),
        ("DUEL OFENSIVO WON", 84.27, 13.57, None),
        ("DUEL OFENSIVO LOST", 71.64, 25.04, None),
        ("DUEL OFENSIVO LOST", 95.08, 17.39, None),
        ("DUEL OFENSIVO LOST", 103.22, 4.26, None),
    ],
    "Vs Sockers": [
        ("DUEL OFENSIVO WON", 106.88, 2.10, None),
        ("DUEL OFENSIVO WON", 111.20, 7.75, None),
        ("DUEL OFENSIVO WON", 105.55, 14.57, None),
        ("DUEL OFENSIVO WON", 48.20, 13.07, None),
        ("DUEL OFENSIVO LOST", 39.22, 17.56, None),
    ],
    "Vs Slavia Praha": [
        ("DUEL OFENSIVO WON", 115.02, 14.90, None),
        ("DUEL OFENSIVO WON", 118.35, 21.22, None),
        ("DUEL OFENSIVO WON", 95.08, 10.41, None),
        ("DUEL OFENSIVO WON", 87.59, 5.09, None),
        ("DUEL OFENSIVO WON", 84.44, 8.75, None),
        ("DUEL OFENSIVO WON", 77.29, 11.41, None),
        ("DUEL OFENSIVO WON", 64.32, 6.42, None),
        ("DUEL OFENSIVO WON", 59.17, 19.55, None),
        ("DUEL OFENSIVO WON", 48.03, 12.74, None),
        ("DUEL OFENSIVO LOST", 102.06, 19.55, None),
        ("DUEL OFENSIVO LOST", 113.03, 16.73, None),
        ("DUEL OFENSIVO LOST", 105.71, 21.71, None),
    ],
}

# Create DataFrames for each match and combined
dfs_by_match = {}
for match_name, events in matches_data.items():
    dfs_by_match[match_name] = pd.DataFrame(events, columns=["type", "x", "y", "video"])

# All games combined
df_all = pd.concat(dfs_by_match.values(), ignore_index=True)
full_data = {"All games": df_all}
full_data.update(dfs_by_match)

def get_style(event_type, has_video):
    """Returns marker, color (rgba), size, and linewidth based on event type"""
    event_type = event_type.upper()
    
    # 1. DUELOS OFENSIVOS (Offensive Duels)
    if "OFENSIVO" in event_type:
        if "WON" in event_type:
            # Circle (Strong green)
            return 'o', (0.1, 0.95, 0.1, 0.95), 110, 0.5
        if "LOST" in event_type:
            # X marker (Strong red)
            alpha = 0.95 if has_video else 0.75
            return 'x', (0.95, 0.1, 0.1, alpha), 120, 3.0

    # 2. DUELOS DEFENSIVOS (Defensive Duels)
    if "DEFENSIVO" in event_type:
        if "WON" in event_type:
            # Square (Dark green)
            return 's', (0.0, 0.6, 0.0, 0.9), 110, 0.5
        if "LOST" in event_type:
            # Diamond (Dark red)
            alpha = 0.9 if has_video else 0.65
            return 'D', (0.7, 0.0, 0.0, alpha), 110, 2.5

    # 3. OTHER EVENTS
    if "BLOQUEIO" in event_type:
        # Star (Purple)
        return '*', (0.7, 0.3, 0.9, 0.8), 140, 0.5

    if "INTERCEPT" in event_type or "INTERCEPTACAO" in event_type:
        # Star (Larger and bright blue)
        return '*', (0.2, 0.6, 0.95, 0.9), 160, 0.8

    if "FOULED" in event_type:
        # Pentagon (Orange)
        return 'P', (1, 0.5, 0, 0.8), 110, 0.5

    # Default
    return 'o', (0.5, 0.5, 0.5, 0.8), 90, 0.5


def compute_stats(df: pd.DataFrame) -> dict:
    """Compute duel statistics"""
    total = len(df)
    
    # Duel counts
    is_duel = df['type'].str.contains('DUEL|AERIAL', case=False)
    is_won = df['type'].str.contains('WON', case=False)
    is_offensive = df['type'].str.contains('OFENSIVO|OFFENSIVE', case=False)
    is_defensive = df['type'].str.contains('DEFENSIVO|DEFENSIVE', case=False)
    
    all_duels = df[is_duel]
    total_duels = len(all_duels)
    won_duels = all_duels[is_won].shape[0]
    duel_rate = (won_duels / total_duels * 100) if total_duels > 0 else 0
    
    # Offensive
    off_duels = df[is_offensive & is_duel]
    off_total = len(off_duels)
    off_wins = off_duels[is_won].shape[0]
    off_rate = (off_wins / off_total * 100) if off_total > 0 else 0
    
    # Defensive
    def_duels = df[is_defensive & is_duel]
    def_total = len(def_duels)
    def_wins = def_duels[is_won].shape[0]
    def_rate = (def_wins / def_total * 100) if def_total > 0 else 0
    
    # Aerial
    aerial_duels = df[df['type'].str.contains('AERIAL', case=False)]
    aerial_total = len(aerial_duels)
    aerial_wins = aerial_duels[is_won].shape[0]
    aerial_rate = (aerial_wins / aerial_total * 100) if aerial_total > 0 else 0
    
    # Zone stats - Split into 3 zones
    # Left corridor: y < 26.6
    # Central corridor: 26.6 <= y <= 53.3
    # Right corridor: y > 53.3
    
    left_mask = df['y'] < 26.6
    left_duels = df[left_mask & is_duel]
    left_total = len(left_duels)
    left_wins = left_duels[is_won].shape[0]
    left_rate = (left_wins / left_total * 100) if left_total > 0 else 0
    
    central_mask = (df['y'] >= 26.6) & (df['y'] <= 53.3)
    central_duels = df[central_mask & is_duel]
    c_total = len(central_duels)
    c_wins = central_duels[is_won].shape[0]
    c_rate = (c_wins / c_total * 100) if c_total > 0 else 0
    
    # Final third duels
    final_third_mask = df['x'] > 80
    final_third_duels = df[final_third_mask & is_duel]
    final_third_total = len(final_third_duels)
    final_third_wins = final_third_duels[is_won].shape[0]
    final_third_rate = (final_third_wins / final_third_total * 100) if final_third_total > 0 else 0
    blocks = len(df[df['type'].str.contains('BLOQUEIO', case=False)])
    intercepts = len(df[df['type'].str.contains('INTERCEPT', case=False)])
    fouls = len(df[df['type'].str.contains('FOULED', case=False)])
    
    return {
        "total": total,
        "duel_total": total_duels,
        "duel_wins": won_duels,
        "duel_rate": duel_rate,
        "off_total": off_total,
        "off_wins": off_wins,
        "off_rate": off_rate,
        "def_total": def_total,
        "def_wins": def_wins,
        "def_rate": def_rate,
        "aerial_total": aerial_total,
        "aerial_wins": aerial_wins,
        "aerial_rate": aerial_rate,
        "central_total": c_total,
        "central_wins": c_wins,
        "central_rate": c_rate,
        "left_total": left_total,
        "left_wins": left_wins,
        "left_rate": left_rate,
        "central_total": c_total,
        "central_wins": c_wins,
        "central_rate": c_rate,
        "right_total": right_total,
        "right_wins": right_wins,
        "right_rate": right_rate,
        "final_third_total": final_third_total,
        "final_third_wins": final_third_wins,
        "final_third_rate": final_third_rate,
        "blocks": blocks,
        "intercepts": intercepts,
        "fouls": fouls,
    }


# ==========================
# Sidebar Configuration
# ==========================
st.sidebar.header("📋 Filter Configuration")
selected_match = st.sidebar.radio("Select a match", list(full_data.keys()), index=0)

st.sidebar.divider()

# Additional filter
filter_duel_type = st.sidebar.multiselect(
    "Duel Type",
    ["Offensive"],
    default=["Offensive"]
)

st.sidebar.divider()
st.sidebar.caption("Match filtered by selected options above")

# Get selected data
df = full_data[selected_match].copy()

# Apply duel type filter
if not all(x in filter_duel_type for x in ["Offensive", "Defensive", "Aerial", "Other"]):
    mask = pd.Series([False] * len(df))
    if "Offensive" in filter_duel_type:
        mask |= df['type'].str.contains('OFENSIVO', case=False)
    if "Defensive" in filter_duel_type:
        mask |= df['type'].str.contains('DEFENSIVO', case=False)
    if "Aerial" in filter_duel_type:
        mask |= df['type'].str.contains('AERIAL', case=False)
    if "Other" in filter_duel_type:
        mask |= ~df['type'].str.contains('OFENSIVO|DEFENSIVO|AERIAL', case=False)
    df = df[mask]

# Compute stats always from full match data
stats = compute_stats(full_data[selected_match])

# ==========================
# Main Layout
# ==========================
col_map, col_vid = st.columns([1, 1])

with col_map:
    st.subheader("Interactive Pitch Map")
    pitch = Pitch(pitch_type='statsbomb', pitch_color='black', line_color='white')
    fig, ax = pitch.draw(figsize=(10, 7))

    for _, row in df.iterrows():
        has_vid = row["video"] is not None
        marker, color, size, lw = get_style(row["type"], has_vid)
        # White border for all icons
        ec = 'white'
        pitch.scatter(row.x, row.y, marker=marker, s=size, color=color,
                      edgecolors=ec, linewidths=lw, ax=ax, zorder=3)

    # Attack Arrow
    ax.annotate('', xy=(70, 83), xytext=(50, 83),
        arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    ax.text(60, 86, "Attack Direction", ha='center', va='center',
        fontsize=9, color='white', fontweight='bold')

    # Legend
    legend_elements = [
        # --- Offensive Duels ---
        Line2D([0], [0], marker='o', color='w', label='Offensive Duel Won',
               markerfacecolor=(0.1, 0.95, 0.1, 0.95), markersize=10, linestyle='None'),

        Line2D([0], [0], marker='x', color='w', label='Offensive Duel Lost',
               markeredgecolor=(0.95, 0.1, 0.1, 0.95), markersize=10, markeredgewidth=2.5, linestyle='None'),
    ]

    # Apply legend to graphic
    legend = ax.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor='black',
        edgecolor='white',
        fontsize='small',
        title="Match Events",
        title_fontsize='medium',
        labelspacing=1.2,
        borderpad=1.0,
        framealpha=0.95
    )

    legend.get_title().set_fontweight('bold')

    # Convert plot to image for coordinate tracking
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_obj = Image.open(buf)

    # Use fixed width to ensure coordinate scaling works
    click = streamlit_image_coordinates(img_obj, width=700)

# ==========================
# Interaction Logic
# ==========================
selected_event = None

if click is not None:
    real_w, real_h = img_obj.size
    disp_w, disp_h = click["width"], click["height"]

    # Map pixel click to actual image pixels
    pixel_x = click["x"] * (real_w / disp_w)
    pixel_y = click["y"] * (real_h / disp_h)

    # Invert Y for Matplotlib logic and transform to pitch data coordinates
    mpl_pixel_y = real_h - pixel_y
    coords = ax.transData.inverted().transform((pixel_x, mpl_pixel_y))
    field_x, field_y = coords[0], coords[1]

    # Calculate distance to markers
    df["dist"] = np.sqrt((df["x"] - field_x)**2 + (df["y"] - field_y)**2)

    # Radius threshold for easier selection
    RADIUS = 5
    candidates = df[df["dist"] < RADIUS]

    if not candidates.empty:
        selected_event = candidates.loc[candidates["dist"].idxmin()]

# ==========================
# Video Display & Stats
# ==========================
with col_vid:
    st.subheader("Event Details")
    if selected_event is not None:
        st.success(f"**Selected Event:** {selected_event['type']}")
        st.info(f"**Position:** X: {selected_event['x']:.2f}, Y: {selected_event['y']:.2f}")

        if selected_event["video"]:
            try:
                st.video(selected_event["video"])
            except:
                st.error(f"Video file not found: {selected_event['video']}")
        else:
            st.warning("No video footage available for this specific event.")
    else:
        st.info("Select a marker on the pitch to view event details.")

    st.divider()
    st.subheader("Performance Statistics")

    col1, col2 = st.columns(2)
    col1.metric("Overall Duels", f"{stats['duel_wins']}/{stats['duel_total']}", f"{stats['duel_rate']:.1f}% Success")
    col2.metric("Duels in the Final Third", f"{stats['final_third_wins']}/{stats['final_third_total']}", f"{stats['final_third_rate']:.1f}% Success")

    st.divider()
    st.subheader("Zone Performance")

    zc1, zc2, zc3 = st.columns(3)
    zc1.metric("Left Corridor", f"{stats['left_wins']}/{stats['left_total']}", f"{stats['left_rate']:.1f}% Success")
    zc2.metric("Central Corridor", f"{stats['central_wins']}/{stats['central_total']}", f"{stats['central_rate']:.1f}% Success")
    zc3.metric("Right Corridor", f"{stats['right_wins']}/{stats['right_total']}", f"{stats['right_rate']:.1f}% Success")

    st.divider()
    st.subheader("Other Events")

    ec1, ec2, ec3 = st.columns(3)
    ec1.metric("Blocks", stats["blocks"])
    ec2.metric("Interceptions", stats["intercepts"])
    ec3.metric("Fouls", stats["fouls"])
