import umap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import scipy.io as spio
import scipy as sp
from scipy.stats import pearsonr, binom, linregress
import os
import cv2
from imageio import imread
from skimage.transform import resize
import re
import pandas as pd
import colorsys

# Plotly imports for 3D visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.colors as pc

import seaborn as sns
try:
    import distinctipy
    print("distinctipy available for optimal color generation")
except ImportError:
    print("distinctipy not installed - install with: pip install distinctipy")

try:
    from bokeh.plotting import figure, save, output_file
    from bokeh.models import HoverTool, ColumnDataSource
    print("Bokeh available for interactive visualization")
except ImportError:
    print("Bokeh not installed - install with: pip install bokeh")

def pairwise_corr_all(ground_truth, predictions):
    r = np.corrcoef(ground_truth, predictions)
    r = r[:len(ground_truth), len(ground_truth):]
    congruents = np.diag(r)
    success = r < congruents
    success_cnt = np.sum(success, 0)
    perf = np.mean(success_cnt) / (len(ground_truth) - 1)
    p = 1 - binom.cdf(perf * len(ground_truth) * (len(ground_truth) - 1), len(ground_truth) * (len(ground_truth) - 1),
                      0.5)
    return perf, p


def pairwise_corr_individuals(ground_truth, predictions):
    r = np.corrcoef(ground_truth, predictions)
    r = r[:len(ground_truth), len(ground_truth):]
    congruents = np.diag(r)
    success = r < congruents
    success_cnt = np.sum(success, 0)
    perf = success_cnt / (len(ground_truth) - 1)
    return perf


def load_categories_mapping(tsv_path):
    """Load the TSV file and create word to category mapping"""
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"Loaded categories file with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Number of unique categories: {df['category'].nunique()}")

    # Create mapping from word to category (primary mapping)
    word_to_category = dict(zip(df['Word'], df['category']))

    # Create mapping from uniqueID to category (fallback mapping)
    uniqueid_to_category = dict(zip(df['uniqueID'], df['category']))

    # Get unique categories
    unique_categories = df['category'].unique()

    return word_to_category, uniqueid_to_category, unique_categories, df


def match_labels_to_categories(test_labels, word_to_category, uniqueid_to_category):
    """Match test image labels to their categories"""
    matched_categories = []
    unmatched_words = []

    for label in test_labels:
        matched = False

        # First try: exact match with Words
        if label in word_to_category:
            matched_categories.append(word_to_category[label])
            matched = True
        else:
            # Second try: case-insensitive match with Words
            label_lower = label.lower().strip()
            for word, category in word_to_category.items():
                if word.lower().strip() == label_lower:
                    matched_categories.append(category)
                    matched = True
                    break

        # Third try: if not matched with Words, try uniqueID
        if not matched:
            if label in uniqueid_to_category:
                matched_categories.append(uniqueid_to_category[label])
                matched = True
            else:
                # Fourth try: case-insensitive match with uniqueID
                label_lower = label.lower().strip()
                for uniqueid, category in uniqueid_to_category.items():
                    if uniqueid.lower().strip() == label_lower:
                        matched_categories.append(category)
                        matched = True
                        break

        # If still not matched, mark as unknown
        if not matched:
            matched_categories.append('unknown')
            unmatched_words.append(label)

    if unmatched_words:
        print(f"Warning: {len(unmatched_words)} words couldn't be matched to categories:")
        print(f"Unmatched: {unmatched_words[:10]}...")  # Show first 10

    return matched_categories, unmatched_words


# def generate_golden_ratio_colors(n_categories, saturation=0.85, lightness=0.6):
#     """Generate distinct colors using golden ratio for optimal color separation"""
#     golden_ratio = (1 + 5 ** 0.5) / 2
#     golden_angle = 2 * np.pi / golden_ratio
#
#     colors = []
#     for i in range(n_categories):
#         # Use golden ratio to distribute hues evenly
#         hue = (i * golden_angle) % (2 * np.pi)
#         hue_normalized = hue / (2 * np.pi)
#
#         # Convert HSL to RGB
#         rgb = colorsys.hls_to_rgb(hue_normalized, lightness, saturation)
#         colors.append(rgb)
#
#     return colors


# def get_distinct_colors_golden_ratio(n_categories):
#     """Get distinct colors for categories using golden ratio"""
#     if n_categories == 1:
#         return ['#1f77b4']  # Single blue color
#
#     # Generate colors using golden ratio
#     rgb_colors = generate_golden_ratio_colors(n_categories, saturation=0.85, lightness=0.6)
#
#     # Convert to hex colors for plotly
#     hex_colors = []
#     for rgb in rgb_colors:
#         hex_color = '#{:02x}{:02x}{:02x}'.format(
#             int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
#         )
#         hex_colors.append(hex_color)
#
#     return hex_colors
#
#
# def get_distinct_colors_matplotlib(n_categories):
#     """Get distinct colors for matplotlib using golden ratio"""
#     if n_categories == 1:
#         return [(0.122, 0.467, 0.706)]  # Single blue color
#
#     # Generate colors using golden ratio
#     rgb_colors = generate_golden_ratio_colors(n_categories, saturation=0.85, lightness=0.6)
#     return rgb_colors


def create_3d_umap(data, n_components=3, random_state=42):
    """Create 3D UMAP embedding"""
    umap_3d = umap.UMAP(n_components=n_components, random_state=random_state,
                        n_neighbors=30, min_dist=0.1)
    embedding_3d = umap_3d.fit_transform(data)
    return embedding_3d


def plot_3d_category_visualization(embedding_3d, test_categories, test_images,
                                   sigmoid_normalized_clip_corrs, unique_categories, save_path):
    """Create interactive 3D scatter plot with categories using golden ratio colors"""

    # Get unique categories and assign colors
    unique_cats = [cat for cat in unique_categories if cat in test_categories]
    if 'unknown' in test_categories:
        unique_cats.append('unknown')

    n_categories = len(unique_cats)
    colors = get_distinct_colors_golden_ratio(n_categories)
    category_to_color = {cat: colors[i] for i, cat in enumerate(unique_cats)}

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('3D Category View', '2D Performance View'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}]],
        column_widths=[0.7, 0.3]
    )

    # Plot 3D scatter
    for i, cat in enumerate(unique_cats):
        # Ground truth points for this category
        gt_mask = np.array([c == cat for c in test_categories[:len(test_images)]])
        gt_indices = np.where(gt_mask)[0]

        if len(gt_indices) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=embedding_3d[gt_indices, 0],
                    y=embedding_3d[gt_indices, 1],
                    z=embedding_3d[gt_indices, 2],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=category_to_color[cat],
                        symbol='square',
                        line=dict(width=2, color='black'),
                        opacity=0.8
                    ),
                    name=f'{cat} (GT)',
                    text=[f'GT: {cat}' for _ in gt_indices],
                    hovertemplate='<b>%{text}</b><br>' +
                                  'X: %{x:.2f}<br>' +
                                  'Y: %{y:.2f}<br>' +
                                  'Z: %{z:.2f}<extra></extra>',
                    showlegend=True
                ),
                row=1, col=1
            )

            # For reconstructed points
            recon_indices = gt_indices + len(test_images)
            quality_scores = sigmoid_normalized_clip_corrs[gt_indices]
            avg_opacity = float(np.mean(0.4 + 0.6 * quality_scores))

            fig.add_trace(
                go.Scatter3d(
                    x=embedding_3d[recon_indices, 0],
                    y=embedding_3d[recon_indices, 1],
                    z=embedding_3d[recon_indices, 2],
                    mode='markers',
                    marker=dict(
                        size=[6 + 10 * q for q in quality_scores],
                        color=category_to_color[cat],
                        symbol='circle',
                        line=dict(width=1, color='black'),
                        opacity=avg_opacity
                    ),
                    name=f'{cat} (Recon)',
                    text=[f'Recon: {cat}<br>Quality: {q:.2f}' for q in quality_scores],
                    hovertemplate='<b>%{text}</b><br>' +
                                  'X: %{x:.2f}<br>' +
                                  'Y: %{y:.2f}<br>' +
                                  'Z: %{z:.2f}<extra></extra>',
                    showlegend=False
                ),
                row=1, col=1
            )

    # Add 2D performance view
    fig.add_trace(
        go.Scatter(
            x=embedding_3d[:len(test_images), 0],
            y=embedding_3d[:len(test_images), 1],
            mode='markers',
            marker=dict(
                size=8,
                color='lightgray',
                symbol='square',
                line=dict(width=1, color='black'),
                opacity=0.6
            ),
            name='Ground Truth',
            text=['GT' for _ in range(len(test_images))],
            hovertemplate='<b>Ground Truth</b><br>' +
                          'X: %{x:.2f}<br>' +
                          'Y: %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=embedding_3d[len(test_images):, 0],
            y=embedding_3d[len(test_images):, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=sigmoid_normalized_clip_corrs,
                colorscale='Viridis',
                colorbar=dict(
                    title="Quality Score",
                    x=1.02,
                    len=0.5
                ),
                symbol='circle',
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            name='Reconstructed',
            text=[f'Quality: {q:.2f}' for q in sigmoid_normalized_clip_corrs],
            hovertemplate='<b>Reconstructed</b><br>' +
                          'Quality: %{text}<br>' +
                          'X: %{x:.2f}<br>' +
                          'Y: %{y:.2f}<extra></extra>',
            showlegend=False
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title={
            'text': 'Interactive 3D UMAP: EEG-to-Image Reconstruction by Categories',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        width=1400,
        height=800,
        showlegend=True,
        legend=dict(
            x=1.05,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )

    # Update 3D scene
    fig.update_scenes(
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
        zaxis_title='UMAP Dimension 3',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )

    # Save the plot
    fig.write_html(f"{save_path}_3d_interactive.html")
    fig.write_image(f"{save_path}_3d_interactive.png", width=1400, height=800, scale=2)

    return fig




def create_separate_3d_plot_with_opacity_groups(embedding_3d, test_categories, test_images,
                                                sigmoid_normalized_clip_corrs, unique_categories, save_path):
    """Create 3D plot with opacity groups for better quality visualization using golden ratio colors"""

    # Prepare data
    unique_cats = [cat for cat in unique_categories if cat in test_categories]
    if 'unknown' in test_categories:
        unique_cats.append('unknown')

    colors = get_distinct_colors_golden_ratio(len(unique_cats))
    category_to_color = {cat: colors[i] for i, cat in enumerate(unique_cats)}

    # Create figure
    fig = go.Figure()

    # Define opacity groups
    opacity_thresholds = [0.0, 0.3, 0.6, 1.0]
    opacity_values = [0.3, 0.5, 0.7, 0.9]
    #opacity = opacity_val

    # Add ground truth points for each category
    for cat in unique_cats:
        cat_mask = np.array([c == cat for c in test_categories[:len(test_images)]])
        cat_indices = np.where(cat_mask)[0]

        if len(cat_indices) > 0:
            # Ground truth
            fig.add_trace(
                go.Scatter3d(
                    x=embedding_3d[cat_indices, 0],
                    y=embedding_3d[cat_indices, 1],
                    z=embedding_3d[cat_indices, 2],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=category_to_color[cat],
                        symbol='square',
                        line=dict(width=2, color='black'),
                        opacity=0.9
                    ),
                    name=f'{cat} (GT)',
                    legendgroup=cat,
                    text=[f'Ground Truth: {cat}' for _ in cat_indices],
                    hovertemplate='<b>%{text}</b><br>' +
                                  'Coordinates: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
                )
            )

            # Reconstructed points grouped by quality/opacity
            recon_indices = cat_indices + len(test_images)
            quality_scores = sigmoid_normalized_clip_corrs[cat_indices]

            # Group points by opacity level
            for i, (low_thresh, high_thresh, opacity_val) in enumerate(
                    zip(opacity_thresholds[:-1], opacity_thresholds[1:], opacity_values)):
                mask = (quality_scores >= low_thresh) & (quality_scores < high_thresh)
                if np.any(mask):
                    group_indices = recon_indices[mask]
                    group_quality = quality_scores[mask]

                    fig.add_trace(
                        go.Scatter3d(
                            x=embedding_3d[group_indices, 0],
                            y=embedding_3d[group_indices, 1],
                            z=embedding_3d[group_indices, 2],
                            mode='markers',
                            marker=dict(
                                size=[5 + 15 * q for q in group_quality],
                                color=category_to_color[cat],
                                symbol='circle',
                                line=dict(width=1, color='black'),
                                opacity=opacity_val
                            ),
                            name=f'{cat} (Q{i + 1})',
                            legendgroup=cat,
                            text=[f'Reconstructed: {cat}<br>Quality: {q:.3f}' for q in group_quality],
                            hovertemplate='<b>%{text}</b><br>' +
                                          'Coordinates: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>',
                            showlegend=False
                        )
                    )

    # Update layout
    fig.update_layout(
        title={
            'text': '3D UMAP Visualization: EEG-to-Image Reconstruction (Quality-based Opacity)<br>' +
                    '<sub>Squares: Ground Truth | Circles: Reconstructed (size & opacity = quality)</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        width=1200,
        height=900,
        scene=dict(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            zaxis_title='UMAP Dimension 3',
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=1.2)
            ),
            bgcolor='rgba(240,240,240,0.1)'
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )

    # Save the plot
    fig.write_html(f"{save_path}_3d_opacity_groups.html")
    fig.write_image(f"{save_path}_3d_opacity_groups.png", width=1200, height=900, scale=2)

    return fig


def generate_distinct_colors_multiple_methods(n_categories):
    """Generate distinct colors using multiple methods for maximum distinction"""

    if n_categories <= 10:
        # Use high-quality predefined palettes for small numbers
        colors = plt.cm.tab10(np.linspace(0, 1, n_categories))
        return [f'#{int(c[0] * 255):02x}{int(c[1] * 255):02x}{int(c[2] * 255):02x}' for c in colors]

    elif n_categories <= 20:
        # Combine tab10 and tab20
        colors1 = plt.cm.tab10(np.linspace(0, 1, 10))
        colors2 = plt.cm.tab20b(np.linspace(0, 1, n_categories - 10))
        colors = np.vstack([colors1, colors2])
        return [f'#{int(c[0] * 255):02x}{int(c[1] * 255):02x}{int(c[2] * 255):02x}' for c in colors]

    else:
        # For larger numbers, use distinctipy library
        try:
            import distinctipy
            colors_rgb = distinctipy.get_colors(n_categories, pastel_factor=0.2)
            return [f'#{int(c[0] * 255):02x}{int(c[1] * 255):02x}{int(c[2] * 255):02x}' for c in colors_rgb]
        except ImportError:
            print("distinctipy not installed, falling back to improved golden ratio method")
            return generate_improved_golden_ratio_colors(n_categories)


def create_separate_3d_plot(embedding_3d, test_categories, test_images,
                            sigmoid_normalized_clip_corrs, unique_categories, save_path):
    """Enhanced 3D plot with distinctipy colors - FIXED opacity issue"""

    # Get unique categories
    unique_cats = [cat for cat in unique_categories if cat in test_categories]
    if 'unknown' in test_categories and 'unknown' not in unique_cats:
        unique_cats.append('unknown')

    # Get the best colors using enhanced method
    colors, method = get_best_distinct_colors_enhanced(len(unique_cats))
    print(f"Using {method} for 3D visualization with {len(unique_cats)} categories")

    # Verify color uniqueness
    verify_color_uniqueness(colors)

    category_to_color = {cat: colors[i] for i, cat in enumerate(unique_cats)}

    fig = go.Figure()

    # Add traces for each category
    for cat in unique_cats:
        cat_mask = np.array([c == cat for c in test_categories[:len(test_images)]])
        cat_indices = np.where(cat_mask)[0]

        if len(cat_indices) > 0:
            # Ground truth points
            fig.add_trace(
                go.Scatter3d(
                    x=embedding_3d[cat_indices, 0],
                    y=embedding_3d[cat_indices, 1],
                    z=embedding_3d[cat_indices, 2],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=category_to_color[cat],
                        symbol='square',
                        line=dict(width=2, color='black'),
                        opacity=0.9  # Single value, not a list
                    ),
                    name=f'{cat} (GT)',
                    text=[f'GT: {cat}' for _ in cat_indices],
                    hovertemplate='<b>%{text}</b><br>' +
                                  'Coords: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
                                  '<extra></extra>'
                )
            )

            # Reconstructed points - ADD EACH POINT INDIVIDUALLY for variable opacity
            recon_indices = cat_indices + len(test_images)
            quality_scores = sigmoid_normalized_clip_corrs[cat_indices]

            # Plot each reconstructed point individually to allow different opacities
            for i, (recon_idx, quality) in enumerate(zip(recon_indices, quality_scores)):
                individual_opacity = float(0.3 + 0.7 * quality)  # Convert to float
                individual_size = float(8 + 15 * quality)  # Convert to float

                fig.add_trace(
                    go.Scatter3d(
                        x=[embedding_3d[recon_idx, 0]],  # Single point as list
                        y=[embedding_3d[recon_idx, 1]],
                        z=[embedding_3d[recon_idx, 2]],
                        mode='markers',
                        marker=dict(
                            size=individual_size,
                            color=category_to_color[cat],
                            symbol='circle',
                            line=dict(width=1, color='black'),
                            opacity=individual_opacity  # Single float value
                        ),
                        name=f'{cat} (Recon)',
                        text=[f'Recon: {cat}<br>Quality: {quality:.3f}'],
                        hovertemplate='<b>%{text}</b><br>' +
                                      'Coords: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
                                      '<extra></extra>',
                        showlegend=False,  # Don't show in legend (too many traces)
                        legendgroup=f'{cat}_recon'  # Group them together
                    )
                )

    # Enhanced layout
    fig.update_layout(
        title={
            'text': f'Enhanced 3D UMAP: {len(unique_cats)} Categories ({method})<br>' +
                    '<sub>Squares: Ground Truth | Circles: Reconstructed (size/opacity = quality)</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        width=1400,
        height=1000,
        scene=dict(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            zaxis_title='UMAP Dimension 3',
            camera=dict(
                eye=dict(x=1.3, y=1.3, z=1.3),
                center=dict(x=0, y=0, z=0)
            ),
            bgcolor='rgba(245,245,245,0.1)',
            aspectmode='cube'  # Equal aspect ratio
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=10)
        )
    )

    # Save with high resolution
    fig.write_html(f"{save_path}_3d_separate.html")
    fig.write_image(f"{save_path}_3d_separate.png", width=1400, height=1000, scale=2)

    return fig

# def generate_improved_golden_ratio_colors(n_categories, saturation_range=(0.6, 0.9), lightness_range=(0.4, 0.8)):
#     """Improved golden ratio colors with varied saturation and lightness"""
#     golden_ratio = (1 + 5 ** 0.5) / 2
#     golden_angle = 2 * np.pi / golden_ratio
#
#     colors = []
#     for i in range(n_categories):
#         # Use golden ratio for hue
#         hue = (i * golden_angle) % (2 * np.pi)
#         hue_normalized = hue / (2 * np.pi)
#
#         # Vary saturation and lightness to increase distinctiveness
#         saturation = saturation_range[0] + (saturation_range[1] - saturation_range[0]) * ((i * 0.618) % 1)
#         lightness = lightness_range[0] + (lightness_range[1] - lightness_range[0]) * ((i * 0.382) % 1)
#
#         # Convert HSL to RGB
#         rgb = colorsys.hls_to_rgb(hue_normalized, lightness, saturation)
#         hex_color = '#{:02x}{:02x}{:02x}'.format(
#             int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
#         )
#         colors.append(hex_color)
#
#     return colors


def generate_enhanced_golden_ratio_colors(n_categories):
    """Enhanced golden ratio specifically for large numbers like 53"""
    golden_ratio = (1 + 5 ** 0.5) / 2
    golden_angle = 2 * np.pi / golden_ratio

    colors = []
    # Use multiple strategies to ensure distinctness
    for i in range(n_categories):
        # Primary hue from golden ratio
        hue = (i * golden_angle) % (2 * np.pi)
        hue_normalized = hue / (2 * np.pi)

        # Vary saturation in cycles to avoid similar colors
        sat_cycle = 0.5 + 0.4 * np.sin(i * 2 * np.pi / 7)  # 7-cycle for saturation

        # Vary lightness in different cycle
        light_cycle = 0.4 + 0.3 * np.cos(i * 2 * np.pi / 11)  # 11-cycle for lightness

        # Ensure minimum differences
        saturation = max(0.3, min(0.9, sat_cycle))
        lightness = max(0.2, min(0.8, light_cycle))

        rgb = colorsys.hls_to_rgb(hue_normalized, lightness, saturation)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
        colors.append(hex_color)

    return colors


# def get_best_distinct_colors(n_categories):
#     """Get the best distinct colors for the given number of categories"""
#
#     # Method 1: Try distinctipy (best for large numbers)
#     try:
#         import distinctipy
#         colors_rgb = distinctipy.get_colors(n_categories,
#                                             pastel_factor=0.2,
#                                             rng=42)  # Fixed seed for reproducibility
#         hex_colors = [f'#{int(c[0] * 255):02x}{int(c[1] * 255):02x}{int(c[2] * 255):02x}' for c in colors_rgb]
#         return hex_colors, "distinctipy"
#     except ImportError:
#         pass
#
#     # Method 2: Use qualitative palettes for smaller numbers
#     if n_categories <= 12:
#         import seaborn as sns
#         palette = sns.color_palette("Set3", n_categories)
#         hex_colors = [f'#{int(c[0] * 255):02x}{int(c[1] * 255):02x}{int(c[2] * 255):02x}' for c in palette]
#         return hex_colors, "seaborn_Set3"
#
#     elif n_categories <= 20:
#         import seaborn as sns
#         palette = sns.color_palette("tab20", n_categories)
#         hex_colors = [f'#{int(c[0] * 255):02x}{int(c[1] * 255):02x}{int(c[2] * 255):02x}' for c in palette]
#         return hex_colors, "seaborn_tab20"
#
#     else:
#         # Method 3: Improved golden ratio with varied parameters
#         hex_colors = generate_improved_golden_ratio_colors(n_categories)
#         return hex_colors, "improved_golden_ratio"


def get_best_distinct_colors_enhanced(n_categories):
    """Enhanced color generation with better parameters for large numbers"""

    # Method 1: distinctipy with optimized parameters
    try:
        import distinctipy

        # For large numbers, use optimized parameters
        if n_categories > 20:
            colors_rgb = distinctipy.get_colors(
                n_categories,
                pastel_factor=0.0,  # More saturated colors
                rng=42,  # Reproducible
                colorblind_type="Deuteranomaly",  # Consider colorblind users
                exclude_colors=[(0, 0, 0), (1, 1, 1)]  # Exclude black/white
            )
        else:
            colors_rgb = distinctipy.get_colors(
                n_categories,
                pastel_factor=0.2,
                rng=42
            )

        hex_colors = [f'#{int(c[0] * 255):02x}{int(c[1] * 255):02x}{int(c[2] * 255):02x}' for c in colors_rgb]
        return hex_colors, f"distinctipy_optimized"

    except ImportError:
        print("Warning: distinctipy not installed. Install with: pip install distinctipy")
        print("Falling back to less optimal color generation...")

    # Method 2: Fallback methods
    if n_categories <= 12:
        import seaborn as sns
        palette = sns.color_palette("Set3", n_categories)
        hex_colors = [f'#{int(c[0] * 255):02x}{int(c[1] * 255):02x}{int(c[2] * 255):02x}' for c in palette]
        return hex_colors, "seaborn_Set3"

    elif n_categories <= 20:
        import seaborn as sns
        palette = sns.color_palette("tab20", n_categories)
        hex_colors = [f'#{int(c[0] * 255):02x}{int(c[1] * 255):02x}{int(c[2] * 255):02x}' for c in palette]
        return hex_colors, "seaborn_tab20"

    else:
        # Enhanced golden ratio for 53 categories
        hex_colors = generate_enhanced_golden_ratio_colors(n_categories)
        return hex_colors, "enhanced_golden_ratio"


def verify_color_uniqueness(colors):
    """Verify that all colors are unique"""
    unique_colors = list(set(colors))
    print(f"Total colors requested: {len(colors)}")
    print(f"Unique colors generated: {len(unique_colors)}")
    print(f"All colors unique: {len(colors) == len(unique_colors)}")

    if len(colors) != len(unique_colors):
        duplicates = [color for color in colors if colors.count(color) > 1]
        print(f"Duplicate colors found: {set(duplicates)}")

    return len(colors) == len(unique_colors)


# UPDATED WRAPPER FUNCTIONS:
def get_distinct_colors_golden_ratio(n_categories):
    """Updated function using the enhanced method"""
    colors, method = get_best_distinct_colors_enhanced(n_categories)
    print(f"Using {method} for {n_categories} categories")
    return colors


def get_distinct_colors_matplotlib(n_categories):
    """Updated function for matplotlib"""
    hex_colors, method = get_best_distinct_colors_enhanced(n_categories)
    print(f"Using {method} for {n_categories} categories")

    # Convert hex to RGB tuples for matplotlib
    rgb_colors = []
    for hex_color in hex_colors:
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        rgb_colors.append(rgb)

    return rgb_colors




# def create_bokeh_visualization(embedding_2d, test_categories, test_images,
#                                sigmoid_normalized_clip_corrs, unique_categories, save_path):
#     """Create interactive Bokeh visualization"""
#     try:
#         from bokeh.plotting import figure, save, output_file
#         from bokeh.models import HoverTool, ColumnDataSource
#         from bokeh.layouts import column
#         from bokeh.models.widgets import Div
#     except ImportError:
#         print("Bokeh not installed. Install with: pip install bokeh")
#         return None
#
#     # Get colors
#     unique_cats = [cat for cat in unique_categories if cat in test_categories]
#     if 'unknown' in test_categories and 'unknown' not in unique_cats:
#         unique_cats.append('unknown')
#
#     #colors, method = get_best_distinct_colors(len(unique_cats))
#     colors, method = get_best_distinct_colors_enhanced(len(unique_cats))
#     category_to_color = {cat: colors[i] for i, cat in enumerate(unique_cats)}
#
#     # VERIFY COLOR UNIQUENESS
#     print(f"\n=== Color Verification for Bokeh Visualization ===")
#     verify_color_uniqueness(colors)
#     print(f"Method used: {method}")
#     print(f"First 10 colors: {colors[:10]}")
#     print(f"Categories: {len(unique_cats)}")
#
#     # Create figure
#     p = figure(width=1200, height=800,
#                title=f"Interactive UMAP Visualization ({method} colors)",
#                tools="pan,wheel_zoom,box_zoom,reset,save")
#
#     # Prepare ground truth data
#     gt_colors = [category_to_color.get(cat, '#888888') for cat in test_categories[:len(test_images)]]
#     gt_source = ColumnDataSource(data={
#         'x': embedding_2d[:len(test_images), 0],
#         'y': embedding_2d[:len(test_images), 1],
#         'category': test_categories[:len(test_images)],
#         'type': ['Ground Truth'] * len(test_images),
#         'color': gt_colors,
#         'quality': [1.0] * len(test_images)  # GT has quality 1
#     })
#
#     # Add ground truth points (squares)
#     gt_glyph = p.square('x', 'y', size=10, color='color', alpha=0.8,
#                         line_color='black', line_width=1, source=gt_source)
#
#     # Prepare reconstructed data
#     recon_colors = [category_to_color.get(cat, '#888888') for cat in test_categories[:len(test_images)]]
#     recon_sizes = [8 + 12 * q for q in sigmoid_normalized_clip_corrs]
#     recon_alphas = [0.4 + 0.6 * q for q in sigmoid_normalized_clip_corrs]
#
#     recon_source = ColumnDataSource(data={
#         'x': embedding_2d[len(test_images):, 0],
#         'y': embedding_2d[len(test_images):, 1],
#         'category': test_categories[:len(test_images)],
#         'type': ['Reconstructed'] * len(test_images),
#         'quality': sigmoid_normalized_clip_corrs,
#         'color': recon_colors,
#         'size': recon_sizes,
#         'alpha': recon_alphas
#     })
#
#     # Add reconstructed points (circles)
#     recon_glyph = p.circle('x', 'y', size='size', color='color', alpha='alpha',
#                            line_color='black', line_width=1, source=recon_source)
#
#     # Add hover tools
#     gt_hover = HoverTool(renderers=[gt_glyph], tooltips=[
#         ("Category", "@category"),
#         ("Type", "@type"),
#         ("Position", "(@x{0.00}, @y{0.00})")
#     ])
#
#     recon_hover = HoverTool(renderers=[recon_glyph], tooltips=[
#         ("Category", "@category"),
#         ("Type", "@type"),
#         ("Quality", "@quality{0.000}"),
#         ("Position", "(@x{0.00}, @y{0.00})")
#     ])
#
#     p.add_tools(gt_hover, recon_hover)
#
#     # Style the plot
#     p.xaxis.axis_label = "UMAP Dimension 1"
#     p.yaxis.axis_label = "UMAP Dimension 2"
#     p.title.text_font_size = "16pt"
#     p.axis.axis_label_text_font_size = "12pt"
#
#     # Create a title div with explanation
#     title_div = Div(text=f"""
#     <h2>Interactive UMAP Visualization</h2>
#     <p><strong>Squares:</strong> Ground Truth Images | <strong>Circles:</strong> Reconstructed Images</p>
#     <p><strong>Size/Opacity:</strong> Reconstruction Quality | <strong>Colors:</strong> {len(unique_cats)} Categories</p>
#     <p><strong>Color Method:</strong> {method}</p>
#     """, width=1200, height=100)
#
#     # Combine title and plot
#     layout = column(title_div, p)
#
#     # Save
#     output_file(f"{save_path}_bokeh_interactive.html")
#     save(layout)
#
#     print(f"Bokeh visualization saved to: {save_path}_bokeh_interactive.html")
#     return p

def create_bokeh_visualization(embedding_2d, test_categories, test_images,
                               sigmoid_normalized_clip_corrs, unique_categories, save_path):
    """Create interactive Bokeh visualization with enhanced colors"""
    try:
        from bokeh.plotting import figure, save, output_file
        from bokeh.models import HoverTool, ColumnDataSource
        from bokeh.layouts import column
        from bokeh.models.widgets import Div
    except ImportError:
        print("Bokeh not installed. Install with: pip install bokeh")
        return None

    # Get colors using enhanced method
    unique_cats = [cat for cat in unique_categories if cat in test_categories]
    if 'unknown' in test_categories and 'unknown' not in unique_cats:
        unique_cats.append('unknown')

    colors, method = get_best_distinct_colors_enhanced(len(unique_cats))

    # Verify color uniqueness
    print(f"\n=== Color Verification for Bokeh Visualization ===")
    verify_color_uniqueness(colors)
    print(f"Method used: {method}")

    category_to_color = {cat: colors[i] for i, cat in enumerate(unique_cats)}

    # Create figure
    p = figure(width=1200, height=800,
               title=f"Interactive UMAP Visualization ({method} colors)",
               tools="pan,wheel_zoom,box_zoom,reset,save")

    # Prepare ground truth data
    gt_colors = [category_to_color.get(cat, '#888888') for cat in test_categories[:len(test_images)]]
    gt_source = ColumnDataSource(data={
        'x': embedding_2d[:len(test_images), 0],
        'y': embedding_2d[:len(test_images), 1],
        'category': test_categories[:len(test_images)],
        'type': ['Ground Truth'] * len(test_images),
        'color': gt_colors,
        'quality': [1.0] * len(test_images)  # GT has quality 1
    })

    # Add ground truth points (squares)
    gt_glyph = p.square('x', 'y', size=10, color='color', alpha=0.8,
                        line_color='black', line_width=1, source=gt_source)

    # Prepare reconstructed data
    recon_colors = [category_to_color.get(cat, '#888888') for cat in test_categories[:len(test_images)]]
    recon_sizes = [8 + 12 * q for q in sigmoid_normalized_clip_corrs]
    recon_alphas = [0.4 + 0.6 * q for q in sigmoid_normalized_clip_corrs]

    recon_source = ColumnDataSource(data={
        'x': embedding_2d[len(test_images):, 0],
        'y': embedding_2d[len(test_images):, 1],
        'category': test_categories[:len(test_images)],
        'type': ['Reconstructed'] * len(test_images),
        'quality': sigmoid_normalized_clip_corrs,
        'color': recon_colors,
        'size': recon_sizes,
        'alpha': recon_alphas
    })

    # Add reconstructed points (circles)
    recon_glyph = p.circle('x', 'y', size='size', color='color', alpha='alpha',
                           line_color='black', line_width=1, source=recon_source)

    # Add hover tools
    gt_hover = HoverTool(renderers=[gt_glyph], tooltips=[
        ("Category", "@category"),
        ("Type", "@type"),
        ("Position", "(@x{0.00}, @y{0.00})")
    ])

    recon_hover = HoverTool(renderers=[recon_glyph], tooltips=[
        ("Category", "@category"),
        ("Type", "@type"),
        ("Quality", "@quality{0.000}"),
        ("Position", "(@x{0.00}, @y{0.00})")
    ])

    p.add_tools(gt_hover, recon_hover)

    # Style the plot
    p.xaxis.axis_label = "UMAP Dimension 1"
    p.yaxis.axis_label = "UMAP Dimension 2"
    p.title.text_font_size = "16pt"
    p.axis.axis_label_text_font_size = "12pt"

    # Create a title div with explanation
    title_div = Div(text=f"""
    <h2>Interactive UMAP Visualization</h2>
    <p><strong>Squares:</strong> Ground Truth Images | <strong>Circles:</strong> Reconstructed Images</p>
    <p><strong>Size/Opacity:</strong> Reconstruction Quality | <strong>Colors:</strong> {len(unique_cats)} Categories</p>
    <p><strong>Color Method:</strong> {method} | <strong>All {len(unique_cats)} colors are unique</strong></p>
    """, width=1200, height=120)

    # Combine title and plot
    layout = column(title_div, p)

    # Save
    output_file(f"{save_path}_bokeh_interactive.html")
    save(layout)

    print(f"Bokeh visualization saved to: {save_path}_bokeh_interactive.html")
    return p


# # Network evaluation setup
net_list = [
    ('inceptionv3', 'avgpool'),
    ('clip', 'final'),
    ('alexnet', 2),
    ('alexnet', 5),
    ('efficientnet', 'avgpool'),
    ('swav', 'avgpool')
]

num_test = 200
test_dir = '/mnt/data12_16T/tanaya/results/thingseeg2_test_images_eval_features'
feats_dir = '/mnt/data12_16T/tanaya/data/eeg_dataset/results_5/thingseeg2_preproc/eval_features/sub-08/versatile_diffusion'
distance_fn = sp.spatial.distance.correlation
pairwise_corrs = []

for (net_name, layer) in net_list:
    file_name = '{}/{}_{}.npy'.format(test_dir, net_name, layer)
    gt_feat = np.load(file_name)

    file_name = '{}/{}_{}.npy'.format(feats_dir, net_name, layer)
    eval_feat = np.load(file_name)

    gt_feat = gt_feat.reshape((len(gt_feat), -1))
    eval_feat = eval_feat.reshape((len(eval_feat), -1))

    print(net_name, layer)
    if net_name in ['efficientnet', 'swav']:
        print('distance: ', np.array([distance_fn(gt_feat[i], eval_feat[i]) for i in range(num_test)]).mean())
    else:
        pairwise_corrs.append(pairwise_corr_individuals(gt_feat[:num_test], eval_feat[:num_test]))

clip_corrs = pairwise_corrs[1]
normalized_clip_corrs = (clip_corrs - np.mean(clip_corrs)) / np.std(clip_corrs)
sigmoid_normalized_clip_corrs = 1 / (1 + np.exp(-normalized_clip_corrs))

# Load data files
print("Loading data files...")

# Load test image labels (object names)
test_labels = np.load('/mnt/data12_16T/tanaya/data/thingseeg2_metadata/test_concepts.npy')
print(f"Loaded {len(test_labels)} test image labels")
print(f"First 10 labels: {test_labels[:10]}")

# Load categories mapping from TSV
tsv_path = '/mnt/data12_16T/tanaya/data/thingseeg2_metadata/category53_long-format.tsv'
word_to_category, uniqueid_to_category, unique_categories, df = load_categories_mapping(tsv_path)

# Match test labels to categories
test_categories, unmatched = match_labels_to_categories(test_labels, word_to_category, uniqueid_to_category)

print(f"\nCategory matching results:")
print(f"Total test labels: {len(test_labels)}")
print(f"Matched to categories: {len(test_categories) - test_categories.count('unknown')}")
print(f"Unmatched: {test_categories.count('unknown')}")

# Load CLIP features
test_clip = np.load('/mnt/data12_16T/tanaya/results/thingseeg2_test_images_eval_features/clip_final.npy')
pred_clip = np.load(
    '/mnt/data12_16T/tanaya/data/eeg_dataset/results_5/thingseeg2_preproc/eval_features/sub-08/versatile_diffusion/clip_final.npy')
test_images = np.load('/mnt/data12_16T/tanaya/data/thingseeg2_metadata/test_images.npy')

# Combine features for UMAP
combined = np.concatenate((test_clip, pred_clip), axis=0)

# UMAP embeddings - both 2D and 3D
print("Computing 2D UMAP embedding...")
reducer_2d = umap.UMAP(random_state=1, n_neighbors=15, min_dist=0.1, n_components=2)
embedding_2d = reducer_2d.fit_transform(combined)

print("Computing 3D UMAP embedding...")
embedding_3d = create_3d_umap(combined, n_components=3, random_state=1)


# Fixed matplotlib visualization with golden ratio colors
def plot_category_visualization():
    # Get unique categories and assign colors using golden ratio
    unique_cats = [cat for cat in unique_categories if cat in test_categories]
    if 'unknown' in test_categories:
        unique_cats.append('unknown')

    n_categories = len(unique_cats)
    print(f"Plotting {n_categories} categories with golden ratio colors")

    # Use golden ratio colors for better distinction
    colors = get_distinct_colors_matplotlib(n_categories)
    category_to_color = {cat: colors[i] for i, cat in enumerate(unique_cats)}

    fig, ax = plt.subplots(figsize=(20, 16), dpi=500)

    # Plot ground truth as squares
    for i, category in enumerate(test_categories[:len(test_images)]):
        color = category_to_color.get(category, (0.5, 0.5, 0.5))  # Gray for unknown
        ax.scatter(embedding_2d[i, 0], embedding_2d[i, 1],
                   c=[color], marker='s', s=100, alpha=0.7,
                   edgecolors='black', linewidths=0.3)

    # Plot reconstructed as circles
    for i, category in enumerate(test_categories[:len(test_images)]):
        color = category_to_color.get(category, (0.5, 0.5, 0.5))  # Gray for unknown
        size = 60 + 120 * sigmoid_normalized_clip_corrs[i]
        alpha = 0.4 + 0.6 * sigmoid_normalized_clip_corrs[i]
        ax.scatter(embedding_2d[i + len(test_images), 0], embedding_2d[i + len(test_images), 1],
                   c=[color], marker='o', s=size, alpha=alpha,
                   edgecolors='black', linewidths=0.3)

    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = []

    # Shape legend
    legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                                  markersize=8, label='Ground Truth', markeredgecolor='black'))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                                  markersize=8, label='Reconstructed', markeredgecolor='black'))
    legend_elements.append(Line2D([0], [0], color='w', label=''))

    # Category legend (show top categories by frequency)
    category_counts = pd.Series(test_categories).value_counts()
    top_categories = category_counts.head(53).index.tolist()

    for cat in top_categories:
        if cat in category_to_color:
            count = category_counts[cat]
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=category_to_color[cat], markersize=8,
                                          label=f'{cat} ({count})', markeredgecolor='black'))

    if len(category_counts) > 53:
        remaining = len(category_counts) - 53
        legend_elements.append(Line2D([0], [0], color='w',
                                      label=f'... and {remaining} more categories'))

    # Add legend with proper positioning
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
              frameon=True, fancybox=True, shadow=True, ncol=1, fontsize=10)

    # Set title and labels
    ax.set_title('2D UMAP Visualization: EEG-to-Image Reconstruction by Categories\n'
                 'Squares: Ground Truth | Circles: Reconstructed (size/opacity = quality)',
                 fontsize=16, pad=20)
    ax.set_xlabel('UMAP Dimension 1', fontsize=14)
    ax.set_ylabel('UMAP Dimension 2', fontsize=14)

    # Set grid and styling
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    # Save the plot
    save_path = '/mnt/data12_16T/tanaya/data/eeg_dataset/results_5/thingseeg2_preproc/sub-08/plots/visualization_2d_categories'
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', facecolor='white')
    plt.show()

    return fig, ax


# Execute the visualization functions (FIXED INDENTATION)
print("Creating 2D category visualization...")
fig_2d, ax_2d = plot_category_visualization()

print("Creating 3D interactive visualization...")
save_path_3d = '/mnt/data12_16T/tanaya/data/eeg_dataset/results_5/thingseeg2_preproc/sub-08/plots/3dcategories'

# Create interactive 3D plot with subplots
fig_interactive = plot_3d_category_visualization(
    embedding_3d, test_categories, test_images,
    sigmoid_normalized_clip_corrs, unique_categories, save_path_3d
)

print("Creating separate 3D visualization...")
fig_3d_separate = create_separate_3d_plot(
    embedding_3d, test_categories, test_images,
    sigmoid_normalized_clip_corrs, unique_categories, save_path_3d
)

print("Creating 3D visualization with opacity groups...")
fig_3d_opacity = create_separate_3d_plot_with_opacity_groups(
    embedding_3d, test_categories, test_images,
    sigmoid_normalized_clip_corrs, unique_categories, save_path_3d
)

print("Creating Bokeh interactive visualization...")
bokeh_save_path = '/mnt/data12_16T/tanaya/data/eeg_dataset/results_5/thingseeg2_preproc/sub-08/plots/bokeh_visualization'
bokeh_fig = create_bokeh_visualization(
    embedding_2d, test_categories, test_images,
    sigmoid_normalized_clip_corrs, unique_categories, bokeh_save_path
)


# Print summary statistics
print("\n=== Summary Statistics ===")
print(f"Total test images: {len(test_images)}")
print(f"Total categories: {len(unique_categories)}")
print(f"Categories present in test set: {len(set(test_categories))}")

# Category distribution
category_counts = pd.Series(test_categories).value_counts()
print(f"\nTop 10 most frequent categories:")
for i, (cat, count) in enumerate(category_counts.head(10).items()):
    print(f"{i + 1:2d}. {cat:20s}: {count:3d} images")

# Quality statistics
print(f"\nReconstruction Quality Statistics:")
print(f"Mean CLIP correlation: {np.mean(clip_corrs):.4f}")
print(f"Std CLIP correlation:  {np.std(clip_corrs):.4f}")
print(f"Min CLIP correlation:  {np.min(clip_corrs):.4f}")
print(f"Max CLIP correlation:  {np.max(clip_corrs):.4f}")

print(f"\nNormalized quality scores (sigmoid):")
print(f"Mean: {np.mean(sigmoid_normalized_clip_corrs):.4f}")
print(f"Std:  {np.std(sigmoid_normalized_clip_corrs):.4f}")
print(f"Min:  {np.min(sigmoid_normalized_clip_corrs):.4f}")
print(f"Max:  {np.max(sigmoid_normalized_clip_corrs):.4f}")

# Per-category quality analysis
print(f"\nPer-category quality analysis:")
category_quality = {}
for cat in unique_categories:
    if cat in test_categories:
        cat_mask = np.array([c == cat for c in test_categories[:len(test_images)]])
        if np.any(cat_mask):
            cat_quality = sigmoid_normalized_clip_corrs[cat_mask]
            category_quality[cat] = {
                'count': len(cat_quality),
                'mean_quality': np.mean(cat_quality),
                'std_quality': np.std(cat_quality) if len(cat_quality) > 1 else 0
            }

# Sort by mean quality (descending)
sorted_categories = sorted(category_quality.items(),
                           key=lambda x: x[1]['mean_quality'], reverse=True)

print(f"\nTop 10 categories by reconstruction quality:")
for i, (cat, stats) in enumerate(sorted_categories[:10]):
    print(
        f"{i + 1:2d}. {cat:20s}: {stats['mean_quality']:.4f} ± {stats['std_quality']:.4f} (n={stats['count']})")

print(f"\nBottom 10 categories by reconstruction quality:")
for i, (cat, stats) in enumerate(sorted_categories[-10:]):
    rank = len(sorted_categories) - 9 + i
    print(
        f"{rank:2d}. {cat:20s}: {stats['mean_quality']:.4f} ± {stats['std_quality']:.4f} (n={stats['count']})")

# Save quality analysis to CSV
quality_df = pd.DataFrame([
    {
        'category': cat,
        'count': stats['count'],
        'mean_quality': stats['mean_quality'],
        'std_quality': stats['std_quality']
    }
    for cat, stats in sorted_categories
])

quality_df.to_csv('/mnt/data12_16T/tanaya/data/eeg_dataset/results_5/thingseeg2_preproc/sub-08/plots/category_quality_analysis.csv', index=False)
print(f"\nQuality analysis saved to: /mnt/data12_16T/tanaya/data/eeg_dataset/results_5/thingseeg2_preproc/sub-08/plots/category_quality_analysis.csv")

# Create a quality vs frequency scatter plot
plt.figure(figsize=(12, 8))
categories_in_both = []
counts_list = []
qualities_list = []

for cat in category_quality.keys():
    if cat in category_counts.index:
        categories_in_both.append(cat)
        counts_list.append(category_counts[cat])
        qualities_list.append(category_quality[cat]['mean_quality'])

plt.scatter(counts_list, qualities_list, alpha=0.7, s=60)
plt.xlabel('Category Frequency (Number of Images)', fontsize=12)
plt.ylabel('Mean Reconstruction Quality', fontsize=12)
plt.title('Category Frequency vs. Reconstruction Quality', fontsize=14)
plt.grid(True, alpha=0.3)

# Add correlation coefficient
if len(counts_list) > 1:
    corr_coef = np.corrcoef(counts_list, qualities_list)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: r = {corr_coef:.3f}',
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig('/mnt/data12_16T/tanaya/data/eeg_dataset/results_5/thingseeg2_preproc/sub-08/plots/frequency_vs_quality.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Visualization Complete ===")
print("Generated files:")
print("- 2D matplotlib visualization: visualization_2d_categories.png/.pdf")
print("- 3D interactive plot: 3dcategories_3d_interactive.html/.png")
print("- 3D separate plot: 3dcategories_3d_separate.html/.png")
print("- 3D opacity groups: 3dcategories_3d_opacity_groups.html/.png")
print("- Quality analysis: category_quality_analysis.csv")
print("- Frequency vs Quality: frequency_vs_quality.png")