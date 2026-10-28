import json
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Union, List, Callable
from scipy.optimize import curve_fit
import numpy as np
class PlottingAnalyser:
    """
    Loads and provides filtered access to the experiment analysis results.
    """
    def __init__(self, results_path='analysis_results.json'):
        """
        Initialises the analyser by loading data from the results file.
        The data is loaded once and cached in a pandas DataFrame for speed.

        Args:
            results_path (str): The path to the JSON results file.
        """
        self.results_path = results_path
        self._data_cache = None

    @property
    def data(self) -> pd.DataFrame:
        """
        Loads and parses data from the JSON file. It handles standard JSON
        and tinydb's storage format. Caches data for performance.
        """
        if self._data_cache is None:
            try:
                with open(self.results_path, 'r') as f:
                    raw_data = json.load(f)

                # Check if it's a tinydb-like structure
                if isinstance(raw_data, dict) and '_default' in raw_data:
                    # This looks like a tinydb file.
                    # The records are the values of the inner dictionary.
                    records = list(raw_data['_default'].values())
                    self._data_cache = pd.DataFrame(records)
                else:
                    # If it's not a  tinydb format, something is unexpected.
                    raise ValueError("JSON format not recognized. Expected a list of records or a tinydb structure.")

                print(f"Successfully loaded and parsed data from {self.results_path}")

            except FileNotFoundError:
                print(f"Error: Results file not found at {self.results_path}")
                self._data_cache = pd.DataFrame()
            except Exception as e:
                print(f"An error occurred while loading or parsing {self.results_path}: {e}")
                self._data_cache = pd.DataFrame()
        return self._data_cache

    def get_data(self, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Filters the data based on the provided criteria.

        Args:
            filters (dict, optional): A dictionary where keys are column names 
                                      and values are the values to filter by.
                                      The value can be a single item or a list.

        Returns:
            pd.DataFrame: A filtered pandas DataFrame.
        """
        if self.data.empty:
            return pd.DataFrame()

        filtered_df = self.data.copy()
        if filters:
            for col, value in filters.items():
                if col in filtered_df.columns:
                    if isinstance(value, (list, tuple, np.ndarray)):
                        filtered_df = filtered_df[filtered_df[col].isin(value)]
                    else:
                        filtered_df = filtered_df[filtered_df[col] == value]
                else:
                    print(f"Warning: Filter column '{col}' not found in data. Potential columns: ", filtered_df.columns)
        return filtered_df

def _get_palette_maps(data, hue_by, style_by, palette):
    """
    Pre-computes the color and style mappings to ensure consistency.
    """
    hue_order = None
    hue_map = {}
    if hue_by and not data.empty:
        hue_order = sorted(data[hue_by].unique())
        
        # Determine the palette to use
        palette_name = palette
        if isinstance(palette, dict):
            palette_name = palette.get(hue_by, list(palette.values())[0] if palette else 'viridis')

        colors = sns.color_palette(palette_name, n_colors=len(hue_order))
        hue_map = dict(zip(hue_order, colors))

    style_order = None
    style_map = {}
    if style_by and not data.empty:
        style_order = sorted(data[style_by].unique())
        linestyles = ['-', '--', '-.', ':']
        style_map = {val: linestyles[i % len(linestyles)] for i, val in enumerate(style_order)}

    # Determine the final palette for seaborn
    # If hue is used, hue_map is the palette. Otherwise, it's the original string.
    plot_palette = hue_map if hue_by and hue_map else palette

    return plot_palette, hue_order, style_order, hue_map, style_map

def flexible_plot(
    analyser: PlottingAnalyser,
    x_col: str,
    y_col: str,
    filters: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    hue_by: Optional[str] = None,
    style_by: Optional[str] = None,
    plot_title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    log_x: bool = False,
    log_y: bool = False,
    y_err_col: Optional[str] = None,
    cubic_x: bool = False,
    palette: Optional[Union[str, Dict[str, Any]]] = 'viridis',
    save_path: Optional[str] = None,
    show_plot: bool = True,
    plot_style: str = 'seaborn-v0_8-whitegrid',
    classical_baseline = True,
    fit_func: Optional[Callable] = None,
    show_fit_params: bool = False,
):
    """
    A flexible plotting function. If 'filters' is a list of dicts, it creates subplots.
    """
    plt.style.use(plot_style)

    def _plot_on_ax(ax, data_filters):
        """Helper function to plot data on a given axes object."""
        print("data_filters: ",data_filters)
        data = analyser.get_data(data_filters)

        if classical_baseline:
            print("plotting classical baseline")
            #plot the classical baseline (ie. m_quantum_replicas = 0)
            if "n_spins" in data_filters:
                classical_data = analyser.get_data({"m_quantum_replicas": 0, "n_spins": data_filters["n_spins"]})
            else:
                classical_data = analyser.get_data({"m_quantum_replicas": 0})
            if not classical_data.empty:
                sns.scatterplot(
                    data=classical_data,
                    x=x_col,
                    y=y_col,
                    marker='o',
                    color = "lightgreen")

        if data.empty:
            ax.text(0.5, 0.5, 'No data for these filters', ha='center', va='center', transform=ax.transAxes)
            if data_filters:
                ax.set_title(", ".join([f"{k}={v}" for k, v in data_filters.items()]))
            return

        sort_cols = [col for col in [style_by, hue_by, x_col] if col]
        if sort_cols:
            data = data.sort_values(by=sort_cols)

        # --- Centralized Palette and Style Generation ---
        plot_palette, hue_order, style_order, hue_map, style_map = _get_palette_maps(data, hue_by, style_by, palette)

        sns.scatterplot(
            data=data,
            x=x_col,
            y=y_col,
            hue=hue_by,
            hue_order=hue_order,
            style=style_by,
            style_order=style_order,
            legend='full',
            palette=plot_palette,
            ax=ax,
        )

        # --- Error Bar Logic ---
        if y_err_col and y_err_col in data.columns:
            group_cols = [col for col in [hue_by, style_by] if col]
            if not group_cols: # No grouping, plot all error bars with default color
                ax.errorbar(data[x_col], data[y_col], yerr=data[y_err_col], fmt='none', ecolor='gray', alpha=0.5, capsize=2)
            else:
                for name, group_df in data.groupby(group_cols):
                    color = 'C0' # Default color
                    if hue_by:
                        hue_val = name if not isinstance(name, tuple) else name[group_cols.index(hue_by)]
                        color = hue_map.get(hue_val)
                    
                    ax.errorbar(group_df[x_col], group_df[y_col], yerr=group_df[y_err_col], fmt='none', ecolor=color, alpha=0.5, capsize=2)


        if hue_by or style_by:
            ax.legend(title=' & '.join(filter(None, [hue_by, style_by])).replace('_', ' ').title())


        # --- Fitting Logic ---
        if fit_func and not data.empty:
            group_cols = [col for col in [hue_by, style_by] if col]
            
            if show_fit_params:
                fit_results_data = []

            if group_cols:
                for name, group_df in data.groupby(group_cols):
                    if len(group_df) >= 3:
                        x_data = group_df[x_col]
                        y_data = group_df[y_col]
                        
                        # Use errors for fitting if available
                        sigma = None
                        if y_err_col and y_err_col in group_df.columns:
                            # Filter out zero/nan errors to avoid issues with curve_fit
                            sigma = group_df[y_err_col].values.copy() # Use .copy() to ensure it's mutable
                            sigma[sigma <= 0] = np.inf # Effectively ignore points with no/invalid error

                        try:
                            popt, _ = curve_fit(fit_func, x_data, y_data, sigma=sigma, maxfev=5000)
                            
                            # Generate smooth x for plotting the fit
                            x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
                            y_fit = fit_func(x_smooth, *popt)

                            # --- Determine color and style directly from our pre-computed maps ---
                            color = 'C0' # Default color
                            if hue_by:
                                hue_val = name if not isinstance(name, tuple) else name[group_cols.index(hue_by)]
                                color = hue_map.get(hue_val)

                            linestyle = '-' # Default style
                            if style_by:
                                style_val = name if not isinstance(name, tuple) else name[group_cols.index(style_by)]
                                linestyle = style_map.get(style_val, '-')
                            ax.plot(x_smooth, y_fit, color=color, linestyle=linestyle, alpha=0.6)
                            
                            if show_fit_params:
                                fit_results_data.append({'group': name, 'params': popt})

                        except (RuntimeError, ValueError) as e:
                            print(f"Warning: Could not fit data for group {name}. Reason: {e}")
            else:
                # No grouping, fit the whole dataset
                
                if len(data) >= 3:
                    sigma = None
                    if y_err_col and y_err_col in data.columns:
                        sigma = data[y_err_col].values.copy() # Use .copy() to ensure it's mutable
                        sigma[sigma <= 0] = np.inf

                    try: # Use a try-except block for safety
                        popt, _ = curve_fit(fit_func, data[x_col], data[y_col], sigma=sigma, maxfev=5000)
                        x_smooth = np.linspace(data[x_col].min(), data[x_col].max(), 200)
                        ax.plot(x_smooth, fit_func(x_smooth, *popt), linestyle='--', alpha=0.8)
                        if show_fit_params:
                            fit_results_data.append({'group': 'Full Dataset', 'params': popt})
                    except (RuntimeError, ValueError) as e:
                        print(f"Warning: Could not fit data for the whole dataset. Reason: {e}")
                        print("dataset:", data[x_col], data[y_col])
            
            if show_fit_params and fit_results_data:
                print("\n--- Fit Parameters ---")
                
                # Create header
                header_parts = []
                if group_cols:
                    header_parts.extend(group_cols)
                else:
                    header_parts.append("Group")
                
                num_params = len(fit_results_data[0]['params'])
                param_headers = [f"p{i}" for i in range(num_params)]
                header_parts.extend(param_headers)
                print(" & ".join(header_parts) + " \\\\")
                print("\\hline")

                # Create rows
                for result in fit_results_data:
                    group_name = result['group']
                    params = result['params']
                    row_parts = []
                    if isinstance(group_name, tuple):
                        row_parts.extend(map(str, group_name))
                    else:
                        row_parts.append(str(group_name))
                    
                    row_parts.extend([f"{p:.4f}" for p in params])
                    print(" & ".join(row_parts) + " \\\\")
                print("----------------------\n")

        # Subplot titles and labels
        ax.set_title(", ".join([f"{k}={v}" for k, v in data_filters.items()]) if data_filters else "Full Dataset")
        ax.set_xlabel(x_label if x_label else x_col.replace('_', ' ').title())
        ax.set_ylabel(y_label if y_label else y_col.replace('_', ' ').title())

        if log_x: ax.set_xscale('log', base=10 if data[x_col].min() > 0 else 2)
        if log_y: ax.set_yscale('log', base=10 if data[y_col].min() > 0 else 2)
        if cubic_x: ax.set_xscale('function', functions=(lambda x: x**(1/3), lambda x: x**3))

    # Main logic of flexible_plot
    if isinstance(filters, list):
        n = len(filters)
        if n == 0:
            print("Warning: filters list is empty.")
            return

        ncols = int(math.ceil(math.sqrt(n)))
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False, sharex=True, sharey=True)
        
        for i, f_dict in enumerate(filters):
            ax = axes.flatten()[i]
            _plot_on_ax(ax, f_dict)

        for i in range(n, len(axes.flatten())):
            axes.flatten()[i].set_visible(False)
        
        if plot_title:
            fig.suptitle(plot_title, fontsize=16, weight='bold')
        
        fig.tight_layout(rect=[0, 0, 1, 0.96] if plot_title else None)

    else:
        # Single plot mode
        fig, ax = plt.subplots(figsize=(12, 7))
        _plot_on_ax(ax, filters)
        if plot_title:
            ax.set_title(plot_title, fontsize=16, weight='bold')



    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()

    plt.close(fig)
