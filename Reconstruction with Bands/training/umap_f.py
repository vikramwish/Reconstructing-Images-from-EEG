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
    from bokeh.plotting import figure, save, output_file, show
    from bokeh.models import HoverTool, ColumnDataSource, Tabs, Panel
    from bokeh.layouts import gridplot, column, row
    from bokeh.models.widgets import Div
    from bokeh.io import export_png, export_svgs

    print("Bokeh available for interactive visualization")
except ImportError:
    print("Bokeh not installed - install with: pip install bokeh")

import warnings
warnings.filterwarnings("ignore")

# Or for specific warning types
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


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


def create_3d_umap(data, n_components=3, random_state=42):
    """Create 3D UMAP embedding"""
    umap_3d = umap.UMAP(n_components=n_components, random_state=random_state,
                        n_neighbors=30, min_dist=0.1)
    embedding_3d = umap_3d.fit_transform(data)
    return embedding_3d


def generate_colors_for_all_categories(all_unique_categories):
    """Generate distinct colors for ALL 53 categories, not just detected ones"""
    n_categories = len(all_unique_categories)

    # Method 1: Try distinctipy (best for large numbers)
    try:
        import distinctipy
        if n_categories > 20:
            colors_rgb = distinctipy.get_colors(
                n_categories,
                pastel_factor=0.0,  # More saturated colors
                rng=42,  # Reproducible
                colorblind_type="Deuteranomaly",
                exclude_colors=[(0, 0, 0), (1, 1, 1)]
            )
        else:
            colors_rgb = distinctipy.get_colors(n_categories, pastel_factor=0.2, rng=42)

        hex_colors = [f'#{int(c[0] * 255):02x}{int(c[1] * 255):02x}{int(c[2] * 255):02x}' for c in colors_rgb]
        method = "distinctipy_optimized"

    except ImportError:
        print("Warning: distinctipy not installed. Using fallback color generation...")
        if n_categories <= 12:
            palette = sns.color_palette("Set3", n_categories)
            hex_colors = [f'#{int(c[0] * 255):02x}{int(c[1] * 255):02x}{int(c[2] * 255):02x}' for c in palette]
            method = "seaborn_Set3"
        elif n_categories <= 20:
            palette = sns.color_palette("tab20", n_categories)
            hex_colors = [f'#{int(c[0] * 255):02x}{int(c[1] * 255):02x}{int(c[2] * 255):02x}' for c in palette]
            method = "seaborn_tab20"
        else:
            # Enhanced golden ratio for large numbers
            hex_colors = generate_enhanced_golden_ratio_colors(n_categories)
            method = "enhanced_golden_ratio"

    # Create mapping for ALL categories
    category_to_color = {cat: hex_colors[i] for i, cat in enumerate(all_unique_categories)}

    # Add unknown category color
    if 'unknown' not in category_to_color:
        category_to_color['unknown'] = '#888888'  # Gray for unknown

    print(f"Generated {len(hex_colors)} colors using {method} for all {n_categories} categories")
    print(f"All colors are unique: {len(set(hex_colors)) == len(hex_colors)}")

    return category_to_color, method


def generate_enhanced_golden_ratio_colors(n_categories):
    """Enhanced golden ratio specifically for large numbers like 53"""
    golden_ratio = (1 + 5 ** 0.5) / 2
    golden_angle = 2 * np.pi / golden_ratio

    colors = []
    for i in range(n_categories):
        # Primary hue from golden ratio
        hue = (i * golden_angle) % (2 * np.pi)
        hue_normalized = hue / (2 * np.pi)

        # Vary saturation and lightness in cycles
        sat_cycle = 0.5 + 0.4 * np.sin(i * 2 * np.pi / 7)
        light_cycle = 0.4 + 0.3 * np.cos(i * 2 * np.pi / 11)

        saturation = max(0.3, min(0.9, sat_cycle))
        lightness = max(0.2, min(0.8, light_cycle))

        rgb = colorsys.hls_to_rgb(hue_normalized, lightness, saturation)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
        colors.append(hex_color)

    return colors


def create_bokeh_2d_umap(embedding_2d, test_categories, test_images,
                         sigmoid_normalized_clip_corrs, category_to_color, save_path, method):
    """Create interactive 2D UMAP visualization with Bokeh"""

    # Create figure
    p = figure(
        width=1200, height=800,
        title=f"Interactive 2D UMAP Visualization )",
        #tools="pan,wheel_zoom,box_zoom,reset,save,tap",
        toolbar_location="above"
    )

    # Prepare ground truth data
    gt_colors = [category_to_color.get(cat, '#888888') for cat in test_categories[:len(test_images)]]
    gt_source = ColumnDataSource(data={
        'x': embedding_2d[:len(test_images), 0],
        'y': embedding_2d[:len(test_images), 1],
        'category': test_categories[:len(test_images)],
        'type': ['Ground Truth'] * len(test_images),
        'color': gt_colors,
        'quality': [1.0] * len(test_images),
        'image_idx': list(range(len(test_images)))
    })

    # Add ground truth points (squares)
    gt_glyph = p.square('x', 'y', size=12, color='color', alpha=0.8,
                        line_color='black', line_width=1, source=gt_source,
                        selection_color='red', nonselection_alpha=0.6)

    # Prepare reconstructed data
    recon_colors = [category_to_color.get(cat, '#888888') for cat in test_categories[:len(test_images)]]
    recon_sizes = [8 + 15 * q for q in sigmoid_normalized_clip_corrs]
    recon_alphas = [0.4 + 0.6 * q for q in sigmoid_normalized_clip_corrs]

    recon_source = ColumnDataSource(data={
        'x': embedding_2d[len(test_images):, 0],
        'y': embedding_2d[len(test_images):, 1],
        'category': test_categories[:len(test_images)],
        'type': ['Reconstructed'] * len(test_images),
        'quality': sigmoid_normalized_clip_corrs,
        'color': recon_colors,
        'size': recon_sizes,
        'alpha': recon_alphas,
        'image_idx': list(range(len(test_images)))
    })

    # Add reconstructed points (circles)
    recon_glyph = p.circle('x', 'y', size='size', color='color', alpha='alpha',
                           line_color='black', line_width=1, source=recon_source,
                           selection_color='red', nonselection_alpha=0.4)

    # Add hover tools
    gt_hover = HoverTool(renderers=[gt_glyph], tooltips=[
        ("Category", "@category"),
        ("Type", "@type"),
        ("Image Index", "@image_idx"),
        ("Position", "(@x{0.00}, @y{0.00})")
    ])

    recon_hover = HoverTool(renderers=[recon_glyph], tooltips=[
        ("Category", "@category"),
        ("Type", "@type"),
        ("Quality", "@quality{0.000}"),
        ("Image Index", "@image_idx"),
        ("Position", "(@x{0.00}, @y{0.00})")
    ])

    p.add_tools(gt_hover, recon_hover)

    # Style the plot
    p.xaxis.axis_label = "UMAP Dimension 1"
    p.yaxis.axis_label = "UMAP Dimension 2"
    p.title.text_font_size = "16pt"
    p.axis.axis_label_text_font_size = "12pt"
    p.grid.grid_line_alpha = 0.3

    return p


def create_bokeh_3d_multiview(embedding_3d, test_categories, test_images,
                              sigmoid_normalized_clip_corrs, category_to_color, save_path, method):
    """Create Bokeh 3D-like visualization using multiple 2D projections"""

    # Create three 2D projections: XY, XZ, YZ
    projections = [
        ("XY View (Dim 1 vs Dim 2)", embedding_3d[:, 0], embedding_3d[:, 1], "UMAP Dimension 1", "UMAP Dimension 2"),
        ("XZ View (Dim 1 vs Dim 3)", embedding_3d[:, 0], embedding_3d[:, 2], "UMAP Dimension 1", "UMAP Dimension 3"),
        ("YZ View (Dim 2 vs Dim 3)", embedding_3d[:, 1], embedding_3d[:, 2], "UMAP Dimension 2", "UMAP Dimension 3")
    ]

    plots = []

    for view_name, x_coords, y_coords, x_label, y_label in projections:
        # Create figure
        p = figure(
            width=450, height=450,
            title=view_name,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            toolbar_location="above"
        )

        # Prepare ground truth data
        gt_colors = [category_to_color.get(cat, '#888888') for cat in test_categories[:len(test_images)]]
        gt_source = ColumnDataSource(data={
            'x': x_coords[:len(test_images)],
            'y': y_coords[:len(test_images)],
            'category': test_categories[:len(test_images)],
            'type': ['Ground Truth'] * len(test_images),
            'color': gt_colors,
            'quality': [1.0] * len(test_images),
            'image_idx': list(range(len(test_images)))
        })

        # Add ground truth points
        gt_glyph = p.square('x', 'y', size=10, color='color', alpha=0.8,
                            line_color='black', line_width=1, source=gt_source)

        # Prepare reconstructed data
        recon_colors = [category_to_color.get(cat, '#888888') for cat in test_categories[:len(test_images)]]
        recon_sizes = [8 + 12 * q for q in sigmoid_normalized_clip_corrs]
        recon_alphas = [0.4 + 0.6 * q for q in sigmoid_normalized_clip_corrs]

        recon_source = ColumnDataSource(data={
            'x': x_coords[len(test_images):],
            'y': y_coords[len(test_images):],
            'category': test_categories[:len(test_images)],
            'type': ['Reconstructed'] * len(test_images),
            'quality': sigmoid_normalized_clip_corrs,
            'color': recon_colors,
            'size': recon_sizes,
            'alpha': recon_alphas,
            'image_idx': list(range(len(test_images)))
        })

        # Add reconstructed points
        recon_glyph = p.circle('x', 'y', size='size', color='color', alpha='alpha',
                               line_color='black', line_width=1, source=recon_source)

        # Add hover tools
        gt_hover = HoverTool(renderers=[gt_glyph], tooltips=[
            ("Category", "@category"),
            ("Type", "@type"),
            ("Image Index", "@image_idx"),
            ("Position", f"({x_label}: @x{{0.00}}, {y_label}: @y{{0.00}})")
        ])

        recon_hover = HoverTool(renderers=[recon_glyph], tooltips=[
            ("Category", "@category"),
            ("Type", "@type"),
            ("Quality", "@quality{0.000}"),
            ("Image Index", "@image_idx"),
            ("Position", f"({x_label}: @x{{0.00}}, {y_label}: @y{{0.00}})")
        ])

        p.add_tools(gt_hover, recon_hover)

        # Style
        p.xaxis.axis_label = x_label
        p.yaxis.axis_label = y_label
        p.title.text_font_size = "12pt"
        p.grid.grid_line_alpha = 0.3

        plots.append(p)

    return plots

def create_bokeh_static_2d_umap(embedding_2d, test_categories, test_images,
                                sigmoid_normalized_clip_corrs, category_to_color, save_path, method):
    """Create static 2D UMAP visualization with Bokeh for export"""

    # Create figure without interactive tools
    p = figure(
        width=1000, height=800,
        title=f"2D UMAP Visualization )",
        toolbar_location=None
    )

    # Add all categories present in data
    present_categories = list(set(test_categories[:len(test_images)]))

    for cat in present_categories:
        if cat == 'unknown':
            continue

        # Get indices for this category
        cat_mask = np.array([c == cat for c in test_categories[:len(test_images)]])
        cat_indices = np.where(cat_mask)[0]

        if len(cat_indices) > 0:
            color = category_to_color.get(cat, '#888888')

            # Ground truth points
            p.square(embedding_2d[cat_indices, 0], embedding_2d[cat_indices, 1],
                     size=10, color=color, alpha=0.8, line_color='black', line_width=1,
                     legend_label=f'{cat} (GT)')

            # Reconstructed points
            recon_indices = cat_indices + len(test_images)
            quality_scores = sigmoid_normalized_clip_corrs[cat_indices]
            sizes = [8 + 12 * q for q in quality_scores]
            alphas = [0.4 + 0.6 * q for q in quality_scores]

            for i, (recon_idx, size, alpha) in enumerate(zip(recon_indices, sizes, alphas)):
                p.circle(embedding_2d[recon_idx, 0], embedding_2d[recon_idx, 1],
                         size=size, color=color, alpha=alpha, line_color='black', line_width=1)

    # Handle unknown category
    if 'unknown' in test_categories:
        unknown_mask = np.array([c == 'unknown' for c in test_categories[:len(test_images)]])
        unknown_indices = np.where(unknown_mask)[0]

        if len(unknown_indices) > 0:
            p.square(embedding_2d[unknown_indices, 0], embedding_2d[unknown_indices, 1],
                     size=10, color='#888888', alpha=0.8, line_color='black', line_width=1,
                     legend_label='unknown (GT)')

            # Reconstructed unknown
            recon_indices = unknown_indices + len(test_images)
            quality_scores = sigmoid_normalized_clip_corrs[unknown_indices]
            for i, (recon_idx, quality) in enumerate(zip(recon_indices, quality_scores)):
                size = 8 + 12 * quality
                alpha = 0.4 + 0.6 * quality
                p.circle(embedding_2d[recon_idx, 0], embedding_2d[recon_idx, 1],
                         size=size, color='#888888', alpha=alpha, line_color='black', line_width=1)

    # Style the plot
    p.xaxis.axis_label = "UMAP Dimension 1"
    p.yaxis.axis_label = "UMAP Dimension 2"
    p.title.text_font_size = "16pt"
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    p.grid.grid_line_alpha = 0.3

    return p

def create_matplotlib_2d_umap(embedding_2d, test_categories, test_images,
                              sigmoid_normalized_clip_corrs, category_to_color, save_path, method):
    """Create matplotlib 2D UMAP visualization"""

    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)

    # Convert hex colors to RGB for matplotlib
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Plot each category
    present_categories = list(set(test_categories[:len(test_images)]))

    for cat in present_categories:
        cat_mask = np.array([c == cat for c in test_categories[:len(test_images)]])
        cat_indices = np.where(cat_mask)[0]

        if len(cat_indices) > 0:
            color_rgb = hex_to_rgb(category_to_color.get(cat, '#888888'))

            # Ground truth points (squares)
            ax.scatter(embedding_2d[cat_indices, 0], embedding_2d[cat_indices, 1],
                       c=[color_rgb], marker='s', s=80, alpha=0.8,
                       edgecolors='black', linewidths=0.5, label=f'{cat} (GT)')

            # Reconstructed points (circles)
            recon_indices = cat_indices + len(test_images)
            quality_scores = sigmoid_normalized_clip_corrs[cat_indices]
            sizes = [60 + 100 * q for q in quality_scores]
            alphas = [0.4 + 0.6 * q for q in quality_scores]

            for i, (recon_idx, size, alpha) in enumerate(zip(recon_indices, sizes, alphas)):
                ax.scatter(embedding_2d[recon_idx, 0], embedding_2d[recon_idx, 1],
                           c=[color_rgb], marker='o', s=size, alpha=alpha,
                           edgecolors='black', linewidths=0.3)

    # Set title and labels
    ax.set_title(f'2D UMAP Visualization: EEG-to-Image Reconstruction\n'
                 'Squares: Ground Truth | Circles: Reconstructed (size/opacity = quality)',
                 fontsize=14, pad=15)
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{save_path}_matplotlib_2d.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}_matplotlib_2d.pdf', bbox_inches='tight')

    return fig, ax



def create_plotly_3d_umap(embedding_3d, test_categories, test_images,
                          sigmoid_normalized_clip_corrs, category_to_color, save_path, method):
    """Create Plotly 3D UMAP visualization with grouped legend interaction"""

    fig = go.Figure()

    # Add traces for each category present in data
    present_categories = list(set(test_categories[:len(test_images)]))

    for cat in present_categories:
        cat_mask = np.array([c == cat for c in test_categories[:len(test_images)]])
        cat_indices = np.where(cat_mask)[0]

        if len(cat_indices) > 0:
            color = category_to_color.get(cat, '#888888')

            # Ground truth points - MAIN legend entry
            fig.add_trace(
                go.Scatter3d(
                    x=embedding_3d[cat_indices, 0],
                    y=embedding_3d[cat_indices, 1],
                    z=embedding_3d[cat_indices, 2],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=color,
                        symbol='square',
                        line=dict(width=2, color='black'),
                        opacity=0.9
                    ),
                    name=f'{cat}',  # Simplified name - represents both GT and Recon
                    legendgroup=cat,  # Group GT and Recon together
                    showlegend=True,  # Show in legend
                    text=[f'GT: {cat}' for _ in cat_indices],
                    hovertemplate='<b>%{text}</b><br>' +
                                  'Coords: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
                                  '<extra></extra>'
                )
            )

            # Reconstructed points - grouped with GT, no separate legend entry
            recon_indices = cat_indices + len(test_images)
            quality_scores = sigmoid_normalized_clip_corrs[cat_indices]

            # Option 1: Add all reconstructed points as individual traces (for variable opacity)
            for i, (recon_idx, quality) in enumerate(zip(recon_indices, quality_scores)):
                individual_opacity = float(0.3 + 0.7 * quality)
                individual_size = float(6 + 12 * quality)

                fig.add_trace(
                    go.Scatter3d(
                        x=[embedding_3d[recon_idx, 0]],
                        y=[embedding_3d[recon_idx, 1]],
                        z=[embedding_3d[recon_idx, 2]],
                        mode='markers',
                        marker=dict(
                            size=individual_size,
                            color=color,
                            symbol='circle',
                            line=dict(width=1, color='black'),
                            opacity=individual_opacity
                        ),
                        name=f'{cat}',  # Same name as GT
                        legendgroup=cat,  # Same legendgroup as GT
                        showlegend=False,  # Don't show in legend - controlled by GT
                        text=[f'Recon: {cat}<br>Quality: {quality:.3f}'],
                        hovertemplate='<b>%{text}</b><br>' +
                                      'Coords: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
                                      '<extra></extra>'
                    )
                )

    # Update layout
    fig.update_layout(
        title={
            'text': f'Interactive 3D UMAP: EEG-to-Image Reconstruction <br>' +
                    '<sub>Click legend to show/hide categories | Squares: GT, Circles: Reconstructed</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        width=1200,
        height=900,
        scene=dict(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            zaxis_title='UMAP Dimension 3',
            camera=dict(
                eye=dict(x=1.3, y=1.3, z=1.3)
            ),
            aspectmode='cube'
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=1,
            itemclick="toggle",  # Enable click to toggle
            itemdoubleclick="toggleothers"  # Enable double-click to show only this category
        )
    )

    return fig




# Network evaluation setup
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
feats_dir = '/mnt/data12_16T/tanaya/data/eeg_dataset/results_3a/thingseeg2_preproc/eval_features/sub-08/versatile_diffusion'
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

# Load categories mapping from TSV
tsv_path = '/mnt/data12_16T/tanaya/data/thingseeg2_metadata/category53_long-format.tsv'
word_to_category, uniqueid_to_category, unique_categories, df = load_categories_mapping(tsv_path)

# Match test labels to categories
test_categories, unmatched = match_labels_to_categories(test_labels, word_to_category, uniqueid_to_category)

print(f"\nCategory matching results:")
print(f"Total test labels: {len(test_labels)}")
print(f"Matched to categories: {len(test_categories) - test_categories.count('unknown')}")
print(f"Unmatched: {test_categories.count('unknown')}")

# Generate colors for ALL 53 categories (not just detected ones)
print(f"\nGenerating colors for ALL {len(unique_categories)} categories...")
category_to_color, method = generate_colors_for_all_categories(unique_categories)

# Load CLIP features
test_clip = np.load('/mnt/data12_16T/tanaya/results/thingseeg2_test_images_eval_features/clip_final.npy')
pred_clip = np.load(
    '/mnt/data12_16T/tanaya/data/eeg_dataset/results_3a/thingseeg2_preproc/eval_features/sub-08/versatile_diffusion/clip_final.npy')
test_images = np.load('/mnt/data12_16T/tanaya/data/thingseeg2_metadata/test_images.npy')

# Combine features for UMAP
combined = np.concatenate((test_clip, pred_clip), axis=0)

# UMAP embeddings - both 2D and 3D
print("Computing 2D UMAP embedding...")
reducer_2d = umap.UMAP(random_state=1, n_neighbors=15, min_dist=0.1, n_components=2)
embedding_2d = reducer_2d.fit_transform(combined)

print("Computing 3D UMAP embedding...")
embedding_3d = create_3d_umap(combined, n_components=3, random_state=1)

# Set up save paths
save_base_path = '/mnt/data12_16T/tanaya/data/eeg_dataset/results_3a/thingseeg2_preproc/sub-08/umaps53'

print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

# 1. Create Interactive Bokeh 2D UMAP
print("\n1. Creating Interactive Bokeh 2D UMAP...")
bokeh_2d_interactive = create_bokeh_2d_umap(
    embedding_2d, test_categories, test_images,
    sigmoid_normalized_clip_corrs, category_to_color, save_base_path, method
)

title_div_2d = Div(text=f"""
    <h2>Interactive 2D UMAP Visualization</h2>
    
    <p><strong>Squares:</strong> Ground Truth Images | <strong>Circles:</strong> Reconstructed Images</p>
    <p><strong>Size oar Opacity:</strong> Reconstruction Quality </p> 
   
    """, width=1200, height=120)

layout_2d_interactive = column(title_div_2d, bokeh_2d_interactive)
output_file(f"{save_base_path}/bokeh_2d_interactive.html")
save(layout_2d_interactive)
print(f"✓ Saved: {save_base_path}/bokeh_2d_interactive.html")

# 2. Create Static Bokeh 2D UMAP
print("\n2. Creating Static Bokeh 2D UMAP...")
bokeh_2d_static = create_bokeh_static_2d_umap(
    embedding_2d, test_categories, test_images,
    sigmoid_normalized_clip_corrs, category_to_color, save_base_path, method
)

title_div_2d_static = Div(text=f"""
    <h2>Static 2D UMAP Visualization</h2>
    
    <p><strong>Squares:</strong> Ground Truth | <strong>Circles:</strong> Reconstructed (size or opacity = quality)</p>
    """, width=1000, height=100)

layout_2d_static = column(title_div_2d_static, bokeh_2d_static)
output_file(f"{save_base_path}/bokeh_2d_static.html")
save(layout_2d_static)
print(f"✓ Saved: {save_base_path}/bokeh_2d_static.html")

# 3. Create Interactive Bokeh 3D Multi-view
print("\n3. Creating Interactive Bokeh 3D Multi-view...")
bokeh_3d_plots = create_bokeh_3d_multiview(
    embedding_3d, test_categories, test_images,
    sigmoid_normalized_clip_corrs, category_to_color, save_base_path, method
)

title_div_3d = Div(text=f"""
    <h2>Interactive 3D UMAP </h2>
    
    <p><strong>Three synchronized 2D projections of 3D UMAP:</strong> XY, XZ, and YZ views</p>
    <p><strong>Squares:</strong> Ground Truth | <strong>Circles:</strong> Reconstructed (size or opacity = quality)</p>
    
    """, width=1400, height=120)

grid_3d = gridplot(bokeh_3d_plots, ncols=3, sizing_mode="scale_width")
layout_3d_interactive = column(title_div_3d, grid_3d)
output_file(f"{save_base_path}/bokeh_3d_interactive.html")
save(layout_3d_interactive)
print(f"✓ Saved: {save_base_path}/bokeh_3d_interactive.html")

# 4. Create Static Matplotlib 2D UMAP
print("\n4. Creating Static Matplotlib 2D UMAP...")
fig_2d_matplotlib, ax_2d = create_matplotlib_2d_umap(
    embedding_2d, test_categories, test_images,
    sigmoid_normalized_clip_corrs, category_to_color, save_base_path, method
)
plt.close(fig_2d_matplotlib)  # Close to save memory
print(f"✓ Saved: {save_base_path}/matplotlib_2d.png and .pdf")

# 5. Create Interactive Plotly 3D UMAP
fig_3d_plotly = create_plotly_3d_umap(  # or create_plotly_3d_umap_efficient
    embedding_3d, test_categories, test_images,
    sigmoid_normalized_clip_corrs, category_to_color, save_base_path, method
)

fig_3d_plotly.write_html(f"{save_base_path}/plotly_3d_interactive.html")
fig_3d_plotly.write_image(f"{save_base_path}/plotly_3d_interactive.png", width=1200, height=900, scale=2)
print(f"✓ Saved: {save_base_path}/plotly_3d_interactive.html and .png")
# 6. Create Enhanced Static Bokeh Plots for Export
print("\n6. Creating Enhanced Static Bokeh Plots for Export...")

def create_bokeh_publication_ready_2d(embedding_2d, test_categories, test_images,
                                     sigmoid_normalized_clip_corrs, category_to_color, method):
    """Create publication-ready 2D plot with legend"""

    p = figure(
        width=800, height=600,
        title=f"2D UMAP: EEG-to-Image Reconstruction ",
        toolbar_location=None,
        title_location="above"
    )

    # Get category counts for legend ordering
    category_counts = pd.Series(test_categories[:len(test_images)]).value_counts()
    present_categories = category_counts.index.tolist()

    # Plot top 15 most frequent categories with legend
    legend_categories = present_categories[:15]

    for cat in legend_categories:
        if cat == 'unknown':
            continue

        cat_mask = np.array([c == cat for c in test_categories[:len(test_images)]])
        cat_indices = np.where(cat_mask)[0]

        if len(cat_indices) > 0:
            color = category_to_color.get(cat, '#888888')
            count = category_counts[cat]

            # Ground truth
            p.square(embedding_2d[cat_indices, 0], embedding_2d[cat_indices, 1],
                    size=8, color=color, alpha=0.8, line_color='black', line_width=0.5,
                    legend_label=f'{cat} ({count})')

            # Reconstructed
            recon_indices = cat_indices + len(test_images)
            quality_scores = sigmoid_normalized_clip_corrs[cat_indices]

            for i, (recon_idx, quality) in enumerate(zip(recon_indices, quality_scores)):
                size = 6 + 8 * quality
                alpha = 0.4 + 0.6 * quality
                p.circle(embedding_2d[recon_idx, 0], embedding_2d[recon_idx, 1],
                        size=size, color=color, alpha=alpha, line_color='black', line_width=0.5)

    # Plot remaining categories without legend
    remaining_categories = present_categories[15:]
    for cat in remaining_categories:
        cat_mask = np.array([c == cat for c in test_categories[:len(test_images)]])
        cat_indices = np.where(cat_mask)[0]

        if len(cat_indices) > 0:
            color = category_to_color.get(cat, '#888888')

            p.square(embedding_2d[cat_indices, 0], embedding_2d[cat_indices, 1],
                    size=8, color=color, alpha=0.8, line_color='black', line_width=0.5)

            recon_indices = cat_indices + len(test_images)
            quality_scores = sigmoid_normalized_clip_corrs[cat_indices]

            for i, (recon_idx, quality) in enumerate(zip(recon_indices, quality_scores)):
                size = 6 + 8 * quality
                alpha = 0.4 + 0.6 * quality
                p.circle(embedding_2d[recon_idx, 0], embedding_2d[recon_idx, 1],
                        size=size, color=color, alpha=alpha, line_color='black', line_width=0.5)

    # Style
    p.xaxis.axis_label = "UMAP Dimension 1"
    p.yaxis.axis_label = "UMAP Dimension 2"
    p.title.text_font_size = "14pt"
    p.legend.location = "top_left"
    p.legend.label_text_font_size = "8pt"
    p.legend.glyph_height = 15
    p.legend.spacing = 1
    p.grid.grid_line_alpha = 0.3

    return p

bokeh_publication_2d = create_bokeh_publication_ready_2d(
    embedding_2d, test_categories, test_images,
    sigmoid_normalized_clip_corrs, category_to_color, method
)

output_file(f"{save_base_path}/bokeh_2d_publication.html")
save(bokeh_publication_2d)
print(f"✓ Saved: {save_base_path}/bokeh_2d_publication.html")

# 7. Create Summary Dashboard
print("\n7. Creating Summary Dashboard...")

def create_summary_dashboard(embedding_2d, embedding_3d, test_categories, test_images,
                           sigmoid_normalized_clip_corrs, category_to_color, method):
    """Create a comprehensive dashboard with multiple views"""

    # Small 2D plot
    p1 = figure(width=400, height=300, title="2D UMAP Overview", toolbar_location=None)

    present_categories = list(set(test_categories[:len(test_images)]))[:10]  # Top 10 for overview

    for cat in present_categories:
        if cat == 'unknown':
            continue

        cat_mask = np.array([c == cat for c in test_categories[:len(test_images)]])
        cat_indices = np.where(cat_mask)[0]

        if len(cat_indices) > 0:
            color = category_to_color.get(cat, '#888888')

            p1.square(embedding_2d[cat_indices, 0], embedding_2d[cat_indices, 1],
                     size=6, color=color, alpha=0.8)

            recon_indices = cat_indices + len(test_images)
            p1.circle(embedding_2d[recon_indices, 0], embedding_2d[recon_indices, 1],
                     size=4, color=color, alpha=0.6)

    p1.xaxis.axis_label = "UMAP Dim 1"
    p1.yaxis.axis_label = "UMAP Dim 2"

    # 3D XY projection
    p2 = figure(width=400, height=300, title="3D UMAP XY View", toolbar_location=None)

    for cat in present_categories:
        if cat == 'unknown':
            continue

        cat_mask = np.array([c == cat for c in test_categories[:len(test_images)]])
        cat_indices = np.where(cat_mask)[0]

        if len(cat_indices) > 0:
            color = category_to_color.get(cat, '#888888')

            p2.square(embedding_3d[cat_indices, 0], embedding_3d[cat_indices, 1],
                     size=6, color=color, alpha=0.8)

            recon_indices = cat_indices + len(test_images)
            p2.circle(embedding_3d[recon_indices, 0], embedding_3d[recon_indices, 1],
                     size=4, color=color, alpha=0.6)

    p2.xaxis.axis_label = "UMAP Dim 1"
    p2.yaxis.axis_label = "UMAP Dim 2"

    # Quality distribution histogram
    p3 = figure(width=400, height=300, title="Quality Distribution", toolbar_location=None)

    hist, edges = np.histogram(sigmoid_normalized_clip_corrs, bins=20)
    p3.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="navy", line_color="white", alpha=0.7)

    p3.xaxis.axis_label = "Reconstruction Quality"
    p3.yaxis.axis_label = "Count"

    # Category frequency bar chart
    p4 = figure(width=400, height=300, title="Category Frequency (Top 10)",
               toolbar_location=None, x_range=present_categories[:10])

    category_counts = pd.Series(test_categories[:len(test_images)]).value_counts()
    top_10_counts = [category_counts.get(cat, 0) for cat in present_categories[:10]]
    colors_for_bars = [category_to_color.get(cat, '#888888') for cat in present_categories[:10]]

    p4.vbar(x=present_categories[:10], top=top_10_counts, width=0.8,
           color=colors_for_bars, alpha=0.8)

    p4.xaxis.major_label_orientation = 45
    p4.yaxis.axis_label = "Count"

    return gridplot([[p1, p2], [p3, p4]], sizing_mode="scale_width")

summary_grid = create_summary_dashboard(
    embedding_2d, embedding_3d, test_categories, test_images,
    sigmoid_normalized_clip_corrs, category_to_color, method
)

title_div_summary = Div(text=f"""
<h2>EEG-to-Image Reconstruction</h2>
<p><strong>Top Left:</strong> 2D UMAP | <strong>Top Right:</strong> 3D UMAP XY View | 
   <strong>Bottom Left:</strong> Quality Distribution | <strong>Bottom Right:</strong> Category Frequency</p>
""", width=800, height=120)

layout_summary = column(title_div_summary, summary_grid)
output_file(f"{save_base_path}/summary_dashboard.html")
save(layout_summary)
print(f"✓ Saved: {save_base_path}/summary_dashboard.html")

# Print comprehensive statistics
print("\n" + "="*60)
print("COMPREHENSIVE ANALYSIS RESULTS")
print("="*60)

print(f"\n📊 Dataset Statistics:")
print(f"  • Total test images: {len(test_images)}")
print(f"  • Total possible categories: {len(unique_categories)}")
print(f"  • Categories present in test set: {len(set(test_categories))}")
print(f"  • Unknown/unmatched labels: {test_categories.count('unknown')}")

print(f"\n🎨 Color Generation:")
print(f"  • Method used: {method}")
print(f"  • Colors generated for ALL {len(unique_categories)} categories")
print(f"  • All colors are unique: {len(set(category_to_color.values())) == len(category_to_color)}")

print(f"\n📈 Reconstruction Quality:")
print(f"  • Mean CLIP correlation: {np.mean(clip_corrs):.4f}")
print(f"  • Std CLIP correlation: {np.std(clip_corrs):.4f}")
print(f"  • Quality range: {np.min(sigmoid_normalized_clip_corrs):.3f} - {np.max(sigmoid_normalized_clip_corrs):.3f}")

# Category analysis
category_counts = pd.Series(test_categories[:len(test_images)]).value_counts()
print(f"\n📂 Category Distribution (Top 10):")
for i, (cat, count) in enumerate(category_counts.head(10).items()):
    quality_mask = np.array([c == cat for c in test_categories[:len(test_images)]])
    if np.any(quality_mask):
        avg_quality = np.mean(sigmoid_normalized_clip_corrs[quality_mask])
        print(f"  {i+1:2d}. {cat:20s}: {count:3d} images (avg quality: {avg_quality:.3f})")

# Per-category quality analysis
category_quality = {}
for cat in set(test_categories[:len(test_images)]):
    if cat != 'unknown':
        cat_mask = np.array([c == cat for c in test_categories[:len(test_images)]])
        if np.any(cat_mask):
            cat_quality = sigmoid_normalized_clip_corrs[cat_mask]
            category_quality[cat] = {
                'count': len(cat_quality),
                'mean_quality': np.mean(cat_quality),
                'std_quality': np.std(cat_quality) if len(cat_quality) > 1 else 0
            }

# Sort by quality
sorted_by_quality = sorted(category_quality.items(), key=lambda x: x[1]['mean_quality'], reverse=True)

print(f"\n🏆 Best Reconstructed Categories (Top 5):")
for i, (cat, stats) in enumerate(sorted_by_quality[:5]):
    print(f"  {i+1}. {cat:20s}: {stats['mean_quality']:.3f} ± {stats['std_quality']:.3f} (n={stats['count']})")

print(f"\n🔻 Most Challenging Categories (Bottom 5):")
for i, (cat, stats) in enumerate(sorted_by_quality[-5:]):
    rank = len(sorted_by_quality) - 4 + i
    print(f"  {rank}. {cat:20s}: {stats['mean_quality']:.3f} ± {stats['std_quality']:.3f} (n={stats['count']})")

# Save detailed analysis
analysis_results = {
    'dataset_stats': {
        'total_images': len(test_images),
        'total_categories': len(unique_categories),
        'present_categories': len(set(test_categories)),
        'unknown_count': test_categories.count('unknown')
    },
    'color_info': {
        'method': method,
        'total_colors_generated': len(category_to_color),
        'all_unique': len(set(category_to_color.values())) == len(category_to_color)
    },
    'quality_stats': {
        'mean_clip_corr': float(np.mean(clip_corrs)),
        'std_clip_corr': float(np.std(clip_corrs)),
        'min_quality': float(np.min(sigmoid_normalized_clip_corrs)),
        'max_quality': float(np.max(sigmoid_normalized_clip_corrs)),
        'mean_quality': float(np.mean(sigmoid_normalized_clip_corrs))
    }
}

# Save category analysis to CSV
quality_df = pd.DataFrame([
    {
        'category': cat,
        'count': stats['count'],
        'mean_quality': stats['mean_quality'],
        'std_quality': stats['std_quality'],
        'color': category_to_color.get(cat, '#888888')
    }
    for cat, stats in sorted_by_quality
])

quality_df.to_csv(f'{save_base_path}/detailed_category_analysis.csv', index=False)

# Create quality vs frequency analysis
plt.figure(figsize=(12, 8))
categories_in_both = []
counts_list = []
qualities_list = []

for cat in category_quality.keys():
    if cat in category_counts.index:
        categories_in_both.append(cat)
        counts_list.append(category_counts[cat])
        qualities_list.append(category_quality[cat]['mean_quality'])

plt.scatter(counts_list, qualities_list, alpha=0.7, s=80,
           c=[category_to_color.get(cat, '#888888') for cat in categories_in_both])
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
plt.savefig(f'{save_base_path}/frequency_vs_quality_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n💾 Files Generated:")
print(f"  Interactive Visualizations:")
print(f"    • {save_base_path}/bokeh_2d_interactive.html")
print(f"    • {save_base_path}/bokeh_3d_interactive.html")
print(f"    • {save_base_path}/plotly_3d_interactive.html")
print(f"    • {save_base_path}/summary_dashboard.html")
print(f"  Static Visualizations:")
print(f"    • {save_base_path}/bokeh_2d_static.html")
print(f"    • {save_base_path}/bokeh_2d_publication.html")
print(f"    • {save_base_path}/matplotlib_2d.png/.pdf")
print(f"    • {save_base_path}/plotly_3d_interactive.png")
print(f"  Analysis Files:")
print(f"    • {save_base_path}/detailed_category_analysis.csv")
print(f"    • {save_base_path}/frequency_vs_quality_analysis.png")


print(f"All {len(unique_categories)} categories have unique, distinct colors using {method}")