# HD-MEA Neuron Tracking

A Python toolkit for tracking neurons across recordings using high-density microelectrode array (HD-MEA) data from cortical organoids. Implements graph-based algorithms to identify and track individual neurons over time based on waveform similarity and spatial location.

## Key Features

- **Graph-Based Tracking**: Track neurons across time using spatial and waveform features
- **Connectivity Analysis**: Identify sender, receiver, and relay neurons 
- **Trajectory Visualization**: Plot neuron movement and persistence over time
- **Comprehensive Metrics**: Waveform similarity, spatial drift, duration, and channel occupancy

## Repository Structure

```
├── base.py                                      # Core data processing functions
├── functions.py                                 # Data loading and plotting utilities
├── analyis_graph.ipynb                          # Main tracking analysis notebook
├── analyis_graph_connectivity.ipynb             # Connectivity analysis notebook
├── umap_with_recs_res1.5_loc.csv                # UMAP embeddings with locations
├── umap_with_recs_res1.5_loc_added.csv          # Extended UMAP with features
└── umap_with_recs_res1.5_loc_added_conn.csv     # UMAP with connectivity
```

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy networkx
pip install braingeneers  # For data loading
pip install fastdtw  # For waveform comparison
```

## Quick Start

```python
import pandas as pd
import numpy as np
import functions as func
from analyis_graph import NeuronTracker, visualize_tracking, visualize_trackable_units

# 1. Load data
umap_df_added = func.load_umap_df("umap_with_recs_res1.5_loc_added.csv")
data = umap_df_added.copy()
data['timestamp'] = pd.to_datetime(data['timestamp'])

# 2. Track neurons
tracker = NeuronTracker(
    spatial_threshold=17.5*np.sqrt(2),  # ~24.7 μm (diagonal neighbor)
    waveform_threshold=0.9,
    time_window="48:00:00",             # 48 hours
    dtw_threshold=0.7,                  # NOT USED - kept for compatibility
    use_cluster=True
)
tracker.construct_graph(data)
trackable_units = tracker.find_trackable_units(duration=10, min_weight=0.5)

trackable_units = tracker.find_trackable_units(duration=10, min_weight=0.5)

# 3. Visualize
visualize_trackable_units(tracker.graph, trackable_units, node_size=20)

# 4. Save results
np.save("trackable_units.npy", trackable_units)
```

## Algorithm Overview

### Tracking Method

1. **Nodes**: Each neuron observation becomes a graph node with position, waveform, timestamp, and cluster ID
2. **Edges**: Connect nodes within time window based on:
   - Spatial weight: `1.0 - distance / spatial_threshold`
   - Waveform weight: `(cosine_similarity - 0.9) / 0.1`
   - Combined: `0.5 × spatial + 0.5 × waveform`
3. **Tracking**: Connected components represent the same neuron over time
3. **Tracking**: Connected components represent the same neuron over time
4. **Filtering**: Require minimum appearances (duration) and edge weight

### Key Parameters

- `spatial_threshold`: 24.7 μm (diagonal neighbor distance)
- `waveform_threshold`: 0.9 (minimum similarity)
- `time_window`: "48:00:00" (format: "HH:MM:SS")
- `dtw_threshold`: NOT USED (kept for backward compatibility)
- `duration`: 10 (minimum appearances)
- `min_weight`: 0.5 (minimum avg edge weight)

### Analysis Metrics

- **Waveform Similarity**: Cosine similarity (0.9-1.0 typical)
- **Spatial Drift**: Euclidean distance (0-80 μm)
- **Channel Occupancy**: Unique electrodes (1-8 typical)
- **Duration**: Time span (hours to days)

## Core Modules

### `base.py`
- `load_curation()`: Load spike sorting results
- `waveform_feature()`: Extract waveform features
- `population_rate()`: Calculate firing rates
- `plot_inset()`: Visualize waveform footprints

### `functions.py`
- `load_umap_df()`: Parse UMAP embeddings from CSV
- `load_connectivity()`: Load functional connectivity
- `sender_and_receiver_id()`: Classify neurons by role
- `plot_functional_map()`: Create connectivity visualizations

### `analyis_graph.ipynb`
Main tracking analysis with:
- NeuronTracker implementation
- Visualization functions
- Trajectory plotting
- Statistical analysis (similarity, drift, duration, occupancy)

### `analyis_graph_connectivity.ipynb`
Connectivity analysis for tracked neurons:
- Transmission probability calculation
- Connectivity maps with latency
- Trackable vs non-trackable comparison

## Data Format

UMAP CSV files contain:
- `pos_x`, `pos_y`: Spatial coordinates (μm)
- `waveform`: Average waveform (50 samples)
- `neighbor_waveforms`: Neighboring electrode waveforms
- `neighbor_positions`: Neighboring electrode positions
- `timestamp`: Recording time
- `color`: Cluster assignment
- `dataset`: Recording session ID

## Dataset

Developed and tested on HD-MEA recordings from mouse cortical organoids:
- Multi-day recordings
- 17.5 μm electrode pitch
- Spike-sorted units with quality metrics
- Functional connectivity between units

## Citation

If you use this code in your research, please cite:

Geng J, Gonzalez-Ferrer J, Voitiuk K, Seiler ST, Schweiger HE, Hernandez S, Salama SR, Haussler D, Mostajo-Radji MA, Teodorescu M. Weighted Graph for Longitudinal Tracking of Neurons in Cortical Organoids on a High-Density Microelectrode Array. *Annu Int Conf IEEE Eng Med Biol Soc.* 2025 Jul;2025:1-5. doi: [10.1109/EMBC58623.2025.11253576](https://doi.org/10.1109/EMBC58623.2025.11253576). PMID: 41335812.

Please also acknowledge the Braingeneers project.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). 

**You are free to:**
- Share and adapt this code for non-commercial purposes
- Use it for research and educational purposes

**You must:**
- Give appropriate credit and cite the publication above
- Indicate if changes were made

**You cannot:**
- Use this code for commercial purposes without explicit permission

For commercial use inquiries, please contact the Braingeneers team.

See the [LICENSE](LICENSE) file for full details.

## Contact

For questions or issues, please open an issue on the GitHub repository or contact the Braingeneers team.