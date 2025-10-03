#!/usr/bin/env python3
"""
Script to visualize skill hierarchies from tasks as pretty hierarchical trees.

This script reads the skills_hierarchy.json files from each task directory
and displays them as formatted tree structures.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import textwrap
from datetime import datetime

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("⚠️  Warning: graphviz not installed. Install with: pip install graphviz")
    print("   Text output will be used instead of image generation.")


class SkillHierarchyVisualizer:
    """Class to visualize skill hierarchies as pretty trees."""
    
    def __init__(self, tasks_base_path: str, save_output: bool = False, output_dir: Optional[str] = None, 
                 output_format: str = "text", layout_style: str = "compact"):
        """
        Initialize the visualizer.
        
        Args:
            tasks_base_path: Path to the base tasks directory
            save_output: Whether to save output to files instead of printing
            output_dir: Directory to save output files (default: writing_images/hierarchies)
            output_format: Output format - "text", "image", or "both" (default: "text")
            layout_style: Layout style for images - "compact", "staggered", or "wide" (default: "compact")
        """
        self.tasks_base_path = Path(tasks_base_path)
        self.save_output = save_output
        self.output_format = output_format
        self.layout_style = layout_style
        self.output_lines = []  # For capturing output when saving
        
        # Validate output format
        if output_format == "image" and not GRAPHVIZ_AVAILABLE:
            print("⚠️  Graphviz not available, falling back to text format")
            self.output_format = "text"
        
        if save_output:
            if output_dir:
                self.output_dir = Path(output_dir)
            else:
                # Default to writing_images/hierarchies relative to the script location
                script_dir = Path(__file__).parent
                project_root = script_dir.parent.parent  # Go up to GenHRL_v2
                self.output_dir = project_root / "writing_images" / "hierarchies"
            
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color scheme for different levels
        self.level_colors = [
            "#FF6B6B",  # Level 1 - Red
            "#4ECDC4",  # Level 2 - Teal  
            "#45B7D1",  # Level 3 - Blue
            "#96CEB4",  # Level 4 - Green
            "#FFEAA7",  # Level 5 - Yellow
            "#DDA0DD",  # Level 6 - Plum
        ]
    
    def output(self, text: str = "") -> None:
        """Output text either to console or capture for file saving."""
        if self.save_output:
            self.output_lines.append(text)
        else:
            print(text)
    
    def save_task_output(self, task_name: str) -> None:
        """Save captured output for a task to a file."""
        if not self.save_output or not self.output_lines:
            return
            
        filename = f"{task_name}_hierarchy.txt"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.output_lines))
        
        print(f"💾 Saved {task_name} hierarchy to: {filepath}")
        self.output_lines.clear()  # Clear for next task
    
    def create_graph_diagram(self, hierarchy: Dict[str, Any], task_name: str, layout_style: str = "compact") -> Optional[graphviz.Digraph]:
        """Create a graphviz diagram from the hierarchy."""
        if not GRAPHVIZ_AVAILABLE:
            return None
            
        # Create a new directed graph with layout-specific settings
        dot = graphviz.Digraph(comment=f'{task_name} Skill Hierarchy')
        
        if layout_style == "compact":
            # Compact layout - minimize width
            dot.attr(rankdir='TB', size='10,16', dpi='300')
            dot.attr('graph', ranksep='0.5', nodesep='0.3', concentrate='true')
            dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='9', 
                    width='1.5', height='0.5')
        elif layout_style == "staggered":
            # Staggered layout - staircase effect
            dot.attr(rankdir='TB', size='12,14', dpi='300')
            dot.attr('graph', ranksep='0.8', nodesep='0.4')
            dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='10')
        else:  # wide (original)
            dot.attr(rankdir='TB', size='16,12', dpi='300')
            dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='10')
        
        dot.attr('edge', color='gray60', arrowsize='0.7')
        
        # Add title node
        title_text = f"{task_name}\\n{hierarchy.get('description', '')[:60]}{'...' if len(hierarchy.get('description', '')) > 60 else ''}"
        dot.node('title', title_text, shape='plaintext', fontsize='12', fontname='Arial Bold')
        
        # Track nodes by level for better layout
        self.node_counter = 0
        self.level_nodes = {}
        self.layout_style = layout_style
        
        # Add all nodes recursively
        self._add_nodes_to_graph(dot, hierarchy, level=0, parent_id=None)
        
        # Apply layout-specific arrangements
        if layout_style == "compact":
            self._apply_compact_layout(dot)
        elif layout_style == "staggered":
            self._apply_staggered_layout(dot)
        else:
            self._apply_standard_layout(dot)
        
        return dot
    
    def _apply_compact_layout(self, dot: graphviz.Digraph):
        """Apply compact layout with grouped children."""
        for level, nodes in self.level_nodes.items():
            if level > 0:  # Skip root level
                # Group nodes into smaller clusters to reduce width
                cluster_size = 3  # Max 3 nodes per cluster
                for i in range(0, len(nodes), cluster_size):
                    cluster_nodes = nodes[i:i+cluster_size]
                    with dot.subgraph(name=f'cluster_l{level}_g{i//cluster_size}') as s:
                        s.attr(style='invis')  # Invisible cluster
                        s.attr(rank='same')
                        for node_id in cluster_nodes:
                            s.node(node_id)
    
    def _apply_staggered_layout(self, dot: graphviz.Digraph):
        """Apply smart staggered layout - wave pattern with limited depth."""
        for level, nodes in self.level_nodes.items():
            if level > 0 and len(nodes) > 2:  # Apply stagger for groups with 3+ nodes
                # Create wave pattern: group nodes into "tiers" with max 4 tiers per level
                max_tiers = 4  # Limit vertical depth
                nodes_per_tier = max(1, len(nodes) // max_tiers)
                
                # If we have too many nodes, group more per tier
                if len(nodes) > max_tiers * 3:
                    nodes_per_tier = max(2, len(nodes) // max_tiers)
                
                tiers = []
                for i in range(0, len(nodes), nodes_per_tier):
                    tier_nodes = nodes[i:i + nodes_per_tier]
                    tiers.append(tier_nodes)
                
                # Limit to max_tiers to prevent infinite staircase
                if len(tiers) > max_tiers:
                    # Merge excess tiers into the last tier
                    last_tier = []
                    for tier in tiers[max_tiers-1:]:
                        last_tier.extend(tier)
                    tiers = tiers[:max_tiers-1] + [last_tier]
                
                # Create each tier as a separate rank
                for tier_idx, tier_nodes in enumerate(tiers):
                    tier_rank = f"tier_{level}_{tier_idx}"
                    
                    with dot.subgraph() as s:
                        s.attr(rank='same')
                        s.attr(name=tier_rank)
                        
                        for node_id in tier_nodes:
                            s.node(node_id)
                    
                    # Add invisible spacer between tiers for visual separation
                    if tier_idx > 0:
                        spacer_id = f"tier_spacer_{level}_{tier_idx}"
                        dot.node(spacer_id, '', style='invis', width='0', height='0.2')
                        
                        # Connect previous tier to spacer and spacer to current tier
                        if tiers[tier_idx-1]:
                            dot.edge(tiers[tier_idx-1][0], spacer_id, style='invis')
                        if tier_nodes:
                            dot.edge(spacer_id, tier_nodes[0], style='invis')
                            
            else:
                # Small groups or single nodes - place in same rank
                with dot.subgraph() as s:
                    s.attr(rank='same')
                    for node_id in nodes:
                        s.node(node_id)
    
    def _apply_standard_layout(self, dot: graphviz.Digraph):
        """Apply standard layout with level labels."""
        for level, nodes in self.level_nodes.items():
            if level > 0:  # Skip root level
                with dot.subgraph() as s:
                    s.attr(rank='same')
                    s.node(f'level_label_{level}', f'Level {level}', 
                           shape='plaintext', fontsize='12', fontname='Arial Bold', color='gray50')
                    for node_id in nodes:
                        s.node(node_id)
    
    def _add_nodes_to_graph(self, dot: graphviz.Digraph, node: Dict[str, Any], 
                           level: int, parent_id: Optional[str]) -> str:
        """Recursively add nodes to the graph."""
        # Generate unique node ID
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1
        
        # Track nodes by level
        if level not in self.level_nodes:
            self.level_nodes[level] = []
        self.level_nodes[level].append(node_id)
        
        # Get node properties
        name = node.get('name', 'Unknown')
        
        # Truncate long names for display
        display_name = name if len(name) <= 30 else name[:27] + "..."
        
        # Create label with just the name (no description for cleaner diagrams)
        label = display_name
        
        # Choose color based on level
        color = self.level_colors[min(level, len(self.level_colors) - 1)]
        
        # Add special styling for leaf nodes (those with object_config_paths)
        if 'object_config_paths' in node:
            dot.node(node_id, label, fillcolor=color, style='rounded,filled,bold', penwidth='2')
        else:
            dot.node(node_id, label, fillcolor=color)
        
        # Connect to parent if exists
        if parent_id:
            dot.edge(parent_id, node_id)
        
        # Process children
        children = node.get('children', [])
        for child in children:
            self._add_nodes_to_graph(dot, child, level + 1, node_id)
        
        return node_id
    
    def save_graph_image(self, dot: graphviz.Digraph, task_name: str, layout_style: str = "compact") -> None:
        """Save the graph as an image file."""
        if not dot:
            return
            
        try:
            # Add layout suffix to filename for clarity
            suffix = f"_{layout_style}" if layout_style != "compact" else ""
            
            # Save as PNG
            png_path = self.output_dir / f"{task_name}_hierarchy{suffix}"
            dot.render(png_path, format='png', cleanup=True)
            print(f"🖼️  Saved {task_name} hierarchy image ({layout_style}) to: {png_path}.png")
            
            # Also save as SVG for scalability
            svg_path = self.output_dir / f"{task_name}_hierarchy{suffix}_vector"
            dot.render(svg_path, format='svg', cleanup=True)
            print(f"📐 Saved {task_name} hierarchy vector ({layout_style}) to: {svg_path}.svg")
            
        except Exception as e:
            print(f"❌ Error saving image for {task_name}: {e}")
            print("   Make sure Graphviz is installed on your system: https://graphviz.org/download/")
        
    def find_task_directories(self) -> List[Path]:
        """Find all task directories containing skills_hierarchy.json files."""
        task_dirs = []
        
        if not self.tasks_base_path.exists():
            self.output(f"Error: Tasks directory not found: {self.tasks_base_path}")
            return task_dirs
            
        for item in self.tasks_base_path.iterdir():
            if item.is_dir():
                # Look for skills_hierarchy.json or skill_hierarchy.json
                hierarchy_file = item / "skills_hierarchy.json"
                alt_hierarchy_file = item / "skill_hierarchy.json"
                
                if hierarchy_file.exists() or alt_hierarchy_file.exists():
                    task_dirs.append(item)
                    
        return sorted(task_dirs)
    
    def load_hierarchy(self, task_dir: Path) -> Dict[str, Any]:
        """Load the skills hierarchy from a task directory."""
        # Try both possible filenames
        hierarchy_file = task_dir / "skills_hierarchy.json"
        if not hierarchy_file.exists():
            hierarchy_file = task_dir / "skill_hierarchy.json"
            
        if not hierarchy_file.exists():
            raise FileNotFoundError(f"No hierarchy file found in {task_dir}")
            
        with open(hierarchy_file, 'r') as f:
            return json.load(f)
    
    def format_description(self, description: str, indent_level: int = 0, max_width: int = 80) -> str:
        """Format description text with proper wrapping and indentation."""
        if not description:
            return ""
            
        # Calculate available width considering the tree structure symbols
        tree_symbols_width = indent_level * 4 + 4  # 4 chars per level + symbols
        available_width = max(30, max_width - tree_symbols_width)
        
        # Wrap the text
        wrapped = textwrap.fill(description, width=available_width)
        
        # Add indentation to all lines except the first
        lines = wrapped.split('\n')
        if len(lines) <= 1:
            return wrapped
            
        # Indent continuation lines
        continuation_indent = " " * (tree_symbols_width + 2)
        formatted_lines = [lines[0]]
        for line in lines[1:]:
            formatted_lines.append(continuation_indent + line)
            
        return '\n'.join(formatted_lines)
    
    def print_node(self, node: Dict[str, Any], prefix: str = "", is_last: bool = True, 
                   indent_level: int = 0, max_width: int = 100) -> None:
        """
        Recursively print a node and its children as a tree structure.
        
        Args:
            node: The current node to print
            prefix: The prefix string for tree formatting
            is_last: Whether this is the last child at this level
            indent_level: Current indentation level
            max_width: Maximum width for text wrapping
        """
        # Choose the appropriate connector
        connector = "└── " if is_last else "├── "
        
        # Output the node name
        node_name = node.get('name', 'Unknown')
        self.output(f"{prefix}{connector}{node_name}")
        
        # Output the description if it exists
        description = node.get('description', '')
        if description:
            # Calculate the prefix for description lines
            desc_prefix = prefix + ("    " if is_last else "│   ") + "    "
            formatted_desc = self.format_description(description, indent_level, max_width)
            
            # Add the description prefix to each line
            desc_lines = formatted_desc.split('\n')
            for line in desc_lines:
                self.output(f"{desc_prefix}📝 {line}")
        
        # Output object config paths if they exist (for leaf nodes)
        if 'object_config_paths' in node:
            desc_prefix = prefix + ("    " if is_last else "│   ") + "    "
            self.output(f"{desc_prefix}🔧 Object config: {len(node['object_config_paths'])} path(s)")
        
        # Process children if they exist
        children = node.get('children', [])
        if children:
            # Calculate the new prefix for children
            new_prefix = prefix + ("    " if is_last else "│   ")
            
            for i, child in enumerate(children):
                is_last_child = (i == len(children) - 1)
                self.print_node(child, new_prefix, is_last_child, indent_level + 1, max_width)
    
    def visualize_task(self, task_dir: Path, max_width: int = 100) -> None:
        """Visualize the hierarchy for a single task."""
        try:
            hierarchy = self.load_hierarchy(task_dir)
            
            # Generate image if requested
            if self.output_format in ["image", "both"] and self.save_output:
                dot = self.create_graph_diagram(hierarchy, task_dir.name, self.layout_style)
                if dot:
                    self.save_graph_image(dot, task_dir.name, self.layout_style)
            
            # Generate text output if requested
            if self.output_format in ["text", "both"]:
                self.output(f"\n{'='*max_width}")
                self.output(f"📁 TASK: {task_dir.name}")
                self.output(f"{'='*max_width}")
                
                # Output task description if available
                task_description = hierarchy.get('description', '')
                if task_description:
                    wrapped_desc = textwrap.fill(task_description, width=max_width-4)
                    self.output(f"📋 {wrapped_desc}")
                    self.output()
                
                # Visualize the hierarchy
                self.print_node(hierarchy, max_width=max_width)
                
                # Save text output if requested
                if self.save_output:
                    self.save_task_output(task_dir.name)
            
        except Exception as e:
            error_msg = f"\n❌ Error processing task {task_dir.name}: {e}"
            if self.save_output and self.output_format in ["text", "both"]:
                self.output(error_msg)
                self.save_task_output(f"{task_dir.name}_ERROR")
            else:
                print(error_msg)
    
    def visualize_all_tasks(self, max_width: int = 100) -> None:
        """Visualize hierarchies for all tasks."""
        if self.save_output:
            print("🌳 SKILL HIERARCHY VISUALIZER - SAVING TO FILES")
            print(f"📂 Scanning tasks in: {self.tasks_base_path}")
            print(f"💾 Output directory: {self.output_dir}")
        else:
            print("🌳 SKILL HIERARCHY VISUALIZER")
            print(f"📂 Scanning tasks in: {self.tasks_base_path}")
        
        task_dirs = self.find_task_directories()
        
        if not task_dirs:
            print(f"❌ No task directories with hierarchy files found in {self.tasks_base_path}")
            return
            
        print(f"✅ Found {len(task_dirs)} task(s) with hierarchy files")
        
        for task_dir in task_dirs:
            self.visualize_task(task_dir, max_width)
        
        # Summary
        if self.save_output:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            summary_content = [
                f"🌳 SKILL HIERARCHY VISUALIZATION SUMMARY",
                f"📅 Generated: {timestamp}",
                f"📂 Source: {self.tasks_base_path}",
                f"📊 Total tasks processed: {len(task_dirs)}",
                f"🎨 Output format: {self.output_format}",
                f"📐 Layout style: {self.layout_style}",
                f"",
                f"📁 Generated files:",
            ]
            
            for task_dir in task_dirs:
                if self.output_format in ["text", "both"]:
                    summary_content.append(f"  - {task_dir.name}_hierarchy.txt")
                if self.output_format in ["image", "both"]:
                    suffix = f"_{self.layout_style}" if self.layout_style != "compact" else ""
                    summary_content.append(f"  - {task_dir.name}_hierarchy{suffix}.png")
                    summary_content.append(f"  - {task_dir.name}_hierarchy{suffix}_vector.svg")
            
            summary_file = self.output_dir / "summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary_content))
            
            print(f"\n{'='*max_width}")
            print(f"✅ Visualization complete! Processed {len(task_dirs)} task(s)")
            print(f"💾 Files saved to: {self.output_dir}")
            print(f"📋 Summary saved to: {summary_file}")
            if self.output_format in ["image", "both"]:
                print(f"🖼️  Generated {len(task_dirs)} PNG images and {len(task_dirs)} SVG vectors")
            print(f"{'='*max_width}")
        else:
            print(f"\n{'='*max_width}")
            print(f"✅ Visualization complete! Processed {len(task_dirs)} task(s)")
            print(f"{'='*max_width}")


def main():
    """Main function to run the visualization."""
    # Default path to tasks directory
    default_tasks_path = Path(__file__).parent.parent.parent / "IsaacLab" / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "manager_based" / "G1_generated" / "tasks"
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize skill hierarchies from task directories. By default, generates image files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate compact images for all tasks (default - saves to writing_images/hierarchies/)
  python visualize_skill_hierarchies.py
  
  # Try different layout styles for better report formatting
  python visualize_skill_hierarchies.py --layout compact     # Clustered, narrow
  python visualize_skill_hierarchies.py --layout staggered  # Smart wave pattern (max 4 tiers)
  python visualize_skill_hierarchies.py --layout wide       # Original wide layout
  
  # Generate text output to console only
  python visualize_skill_hierarchies.py --format text
  
  # Generate both images and text files
  python visualize_skill_hierarchies.py --format both
  
  # Visualize a specific task with compact layout
  python visualize_skill_hierarchies.py --task build_stairs_seed42 --layout compact
  
  # Generate text files instead of images
  python visualize_skill_hierarchies.py --format text --save
  
  # Custom output directory with staggered layout
  python visualize_skill_hierarchies.py --layout staggered --output-dir /path/to/output
        """
    )
    parser.add_argument(
        "--tasks-path", 
        type=str, 
        default=str(default_tasks_path),
        help=f"Path to tasks directory (default: auto-detected)"
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=100,
        help="Maximum width for text formatting (default: 100)"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Visualize only a specific task (task directory name). If not specified, all tasks are visualized."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save visualizations to files instead of printing to console"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save output files (default: writing_images/hierarchies)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "image", "both"],
        default="image",
        help="Output format: text, image, or both (default: image)"
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["compact", "staggered", "wide"],
        default="compact",
        help="Layout style for images: compact (tight clusters), staggered (smart wave pattern), or wide (original) (default: compact)"
    )
    
    args = parser.parse_args()
    
    # Auto-enable save when image format is requested
    save_output = args.save or args.format in ["image", "both"]
    
    # Create visualizer
    visualizer = SkillHierarchyVisualizer(
        args.tasks_path, 
        save_output=save_output, 
        output_dir=args.output_dir,
        output_format=args.format,
        layout_style=args.layout
    )
    
    if args.task:
        # Visualize specific task
        task_dir = Path(args.tasks_path) / args.task
        if task_dir.exists():
            visualizer.visualize_task(task_dir, args.max_width)
        else:
            print(f"❌ Task directory not found: {task_dir}")
    else:
        # Default behavior: Visualize all tasks automatically
        print("🚀 Auto-discovering and visualizing all available tasks...")
        visualizer.visualize_all_tasks(args.max_width)


if __name__ == "__main__":
    main()