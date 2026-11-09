import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_simulation_results(csv_filename, title="Simulation Results", ylabel="Voltage [p.u.]", output_pdf=None, phase=None, upperlimit=1.15, lowerlimit=0.90):
    """
    Reads a simulation result CSV and plots all columns against the index (time).
    
    Args:
        csv_filename (str): Path to the CSV file.
        title (str): Title of the graph.
        ylabel (str): Label for the Y-axis.
    """
    if not os.path.exists(csv_filename):
        print(f"Error: File {csv_filename} not found.")
        return

    # Load the data
    df = pd.read_csv(csv_filename, index_col=0, skiprows=[1]) # Assumes first column is 'time' or index
    
    fig = plt.figure(figsize=(12, 6))
    
    # Plot each column (each bus/phase)
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.ylim(lowerlimit, upperlimit)
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot as a PDF
    pdf_filename = f"{phase}_voltage_profile.pdf"
    pdf_location = os.path.join(output_pdf, "pdfs", pdf_filename)
    if not os.path.exists(os.path.dirname(pdf_location)):
        os.makedirs(os.path.dirname(pdf_location), exist_ok=True)
    
    # Show x-axis ticks only on hour marks (00:00, 01:00, ..., 24:00)
    try:
        # Only show ticks exactly on the hour marks
        tick_positions = [i for i, t in enumerate(df.index) if str(t).endswith(':00:00')]
        if not tick_positions:
            tick_positions = list(range(0, len(df.index), max(1, len(df.index) // 24)))
            
        if 0 not in tick_positions:
            tick_positions.insert(0, 0)
            
        tick_labels = [str(df.index[i])[:5] for i in tick_positions]
        tick_labels[0] = '00:00'
        
        plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45)
    except Exception:
        # Fall back to matplotlib's automatic ticks
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(pdf_location, format='pdf')
    plt.close(fig) # Close the figure to free up memory
    



def plot_combined_phases(file_a, file_b, file_c, output_pdf, title="Combined Phase Voltages", ylabel="Voltage [p.u.]", filename="combined_phases_plot.pdf", upperlimit=1.15, lowerlimit=0.90):
    """
    Reads three CSV files (one for each phase) and plots them on the same graph.
    Assumes the CSVs have the same time index and bus columns.
    """
    fig = plt.figure(figsize=(14, 7))
    files = [file_a, file_b, file_c]
    colors = ['r', 'g', 'b']
    labels = ['Phase A', 'Phase B', 'Phase C']

    # We will store one dataframe to use its index for the x-ticks at the end
    last_df = None 

    for file, color, label in zip(files, colors, labels):
        if os.path.exists(file):
            # Read CSV, skip the second row if it contains metadata like 'unit'
            df = pd.read_csv(file, index_col=0)
            last_df = df
            
            # Plot each bus in the file but use the same color per phase to avoid clutter
            for i, column in enumerate(df.columns):
                # Use range(len(df)) instead of df.index to prevent wrap-around lines
                plt.plot(range(len(df)), df[column], color=color, alpha=0.5,
                         label=f"{label} - Bus {column}" if i == 0 else "")
        else:
            print(f"Warning: {file} not found.")

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.ylim(lowerlimit, upperlimit)
    plt.legend(loc='upper right', fontsize='x-small', ncol=3)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Show x-axis ticks only on hour marks (00:00, 01:00, ..., 24:00)
    if last_df is not None:
        try:
            # Only show ticks exactly on the hour marks
            tick_positions = [i for i, t in enumerate(last_df.index) if str(t).endswith(':00:00')]
            if not tick_positions:
                tick_positions = list(range(0, len(last_df.index), max(1, len(last_df.index) // 24)))
                
            if 0 not in tick_positions:
                tick_positions.insert(0, 0)
                
            tick_labels = [str(last_df.index[i])[:5] for i in tick_positions]
            tick_labels[0] = '00:00'
            
            plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45)
        except Exception:
            plt.xticks(rotation=45)
    else:
        plt.xticks(rotation=45) # Fallback if no files loaded
        
    plt.tight_layout()
        
    output_pdf = os.path.join(output_pdf, filename)
    if not os.path.exists(os.path.dirname(output_pdf)):
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    plt.savefig(output_pdf, format='pdf')
    plt.close(fig) # Close the figure to free up memory