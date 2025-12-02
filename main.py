#!/usr/bin/env python3
"""
Main script to generate all plots for potential analysis.

This script generates loop balance, detailed balance, potential distribution,
and potential graph plots for different databases.
"""

import os
import sys
from V_hand import V_hand

# Add the current directory to path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot import (
    plot_loop_balance,
    plot_detailed_balance,
    draw_potential_graph,
    plot_potential_distribution
)
from build_potential import PotentialBuilder


def main():
    """Generate all plots for the two databases."""
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.join(script_dir, 'data')
    
    # Define paths relative to script directory
    word_database = os.path.join(script_dir, 'word_database_gpt5-nano.json')
    idea_database = os.path.join(script_dir, 'ideasearchfitter_database.json')
    # output_dir = os.path.join(script_dir, 'figures')
    word_output_dir = os.path.join(script_dir, 'figures', 'word_gpt5-nano')
    idea_output_dir = os.path.join(script_dir, 'figures', 'ideasearchfitter')
    
    # Ensure output directory exists
    os.makedirs(word_output_dir, exist_ok=True)
    os.makedirs(idea_output_dir, exist_ok=True)
    
    print("=" * 80)
    print("PART 1: Processing word_database_gpt5-nano.json")
    print("=" * 80)
    print(f"Database: {word_database}")
    print(f"Settings: high_reject, N_threshold=0, sample_ratio=1")
    print()
    
    # Part 1: word_database_gpt5-nano.json with high_reject, N_threshold=0, sample_ratio=1
    
    print("-" * 80)
    print("1.1 Generating Loop Balance Plot")
    print("-" * 80)
    try:
        plot_loop_balance(
            word_database,
            output_dir=word_output_dir,
            loop_threshold=0,
            sample_ratio=1,
            random_seed=42
        )
        print("✓ Loop balance plot completed\n")
    except Exception as e:
        print(f"✗ Error generating loop balance plot: {e}\n")
    
    print("-" * 80)
    print("1.2 Generating Detailed Balance Plot")
    print("-" * 80)
    try:
        plot_detailed_balance(
            word_database,
            output_dir=word_output_dir,
            reject_mode='high_reject',
            N_threshold=0,
            sample_ratio=1,
            random_seed=42
        )
        print("✓ Detailed balance plot completed\n")
    except Exception as e:
        print(f"✗ Error generating detailed balance plot: {e}\n")
    
    print("-" * 80)
    print("1.3 Generating Potential Distribution Plot")
    print("-" * 80)
    try:
        # Build potential for distribution plot
        builder1 = PotentialBuilder(word_database, reject_mode='high_reject', N_threshold=0)
        optimized_potentials1 = builder1.optimize_potentials(max_iter=20000)
        
        def V1(f):
            return optimized_potentials1.get(f, 0.0)
        
        plot_potential_distribution(word_database, V1, output_dir=word_output_dir)
        print("✓ Potential distribution plot completed\n")
    except Exception as e:
        print(f"✗ Error generating potential distribution plot: {e}\n")
    
    print("\n" + "=" * 80)
    print("PART 2: Processing ideasearchfitter_database.json")
    print("=" * 80)
    print(f"Database: {idea_database}")
    print(f"Settings: low_reject, N_threshold=1, sample_ratio=5")
    print()
    
    # Part 2: ideasearchfitter_database.json with low_reject, N_threshold=1, sample_ratio=5
    
    print("-" * 80)
    print("2.1 Generating Loop Balance Plot")
    print("-" * 80)
    try:
        plot_loop_balance(
            idea_database,
            output_dir=idea_output_dir,
            loop_threshold=1,
            sample_ratio=5,
            random_seed=42
        )
        print("✓ Loop balance plot completed\n")
    except Exception as e:
        print(f"✗ Error generating loop balance plot: {e}\n")
    
    print("-" * 80)
    print("2.2 Generating Detailed Balance Plot")
    print("-" * 80)
    try:
        plot_detailed_balance(
            idea_database,
            output_dir=idea_output_dir,
            reject_mode='low_reject',
            N_threshold=1,
            sample_ratio=5,
            random_seed=42
        )
        print("✓ Detailed balance plot completed\n")
    except Exception as e:
        print(f"✗ Error generating detailed balance plot: {e}\n")
    
    print("-" * 80)
    print("2.3 Generating Potential Distribution Plot")
    print("-" * 80)
    try:
        # Build potential for distribution and graph plots
        builder2 = PotentialBuilder(idea_database, reject_mode='low_reject', N_threshold=1)
        optimized_potentials2 = builder2.optimize_potentials(max_iter=20000)
        
        def V2(f):
            return optimized_potentials2.get(f, 0.0)
        
        plot_potential_distribution(idea_database, V2, output_dir=idea_output_dir)
        print("✓ Potential distribution plot completed\n")
    except Exception as e:
        print(f"✗ Error generating potential distribution plot: {e}\n")
    
    print("-" * 80)
    print("2.4 Generating Potential Graph")
    print("-" * 80)
    try:
        draw_potential_graph(
            idea_database,
            V_hand,
            output_dir=idea_output_dir,
            p0_threshold=0.05,
            num_nodes_to_plot=70,
            random_seed=42
        )
        print("✓ Potential graph completed\n")
    except Exception as e:
        print(f"✗ Error generating potential graph: {e}\n")
    
    print("=" * 80)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Output directories: {word_output_dir}, {idea_output_dir}")
    print("\nGenerated files:")
    print("  - loop_balance_plot.pdf")
    print("  - detailed_balance_plot.pdf")
    print("  - potential_distribution.pdf")
    print("  - potential_graph.pdf")
    print()


if __name__ == "__main__":
    main()
