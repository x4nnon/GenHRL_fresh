#!/usr/bin/env python3
"""
Results Analysis Script for GenHRL Scientific Testing

Analyzes the JSON results from scientific testing and generates reports.

Usage:
    python -m genhrl.testing.analyze_results results/scientific_test_results_*.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import statistics
from collections import defaultdict


class ResultsAnalyzer:
    """Analyzer for scientific testing results."""
    
    def __init__(self, results_file: str):
        self.results_file = Path(results_file)
        self.data = self.load_results()
        
    def load_results(self) -> Dict[str, Any]:
        """Load results from JSON file."""
        try:
            with open(self.results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading results file: {e}")
            sys.exit(1)
    
    def print_detailed_analysis(self):
        """Print detailed analysis of results."""
        print("🔬 DETAILED SCIENTIFIC ANALYSIS")
        print("=" * 80)
        
        metadata = self.data.get("metadata", {})
        results = self.data.get("results", [])
        
        print(f"📊 Experiment Metadata:")
        print(f"  Robot: {metadata.get('robot', 'Unknown')}")
        print(f"  Provider: {metadata.get('provider', 'Unknown')}")
        print(f"  Timestamp: {metadata.get('timestamp', 'Unknown')}")
        print(f"  Total experiments: {len(results)}")
        
        # Group results by task
        task_groups = defaultdict(list)
        for result in results:
            task_base = result['task_name'].rsplit('_seed', 1)[0]
            task_groups[task_base].append(result)
        
        print(f"\n📋 Task Analysis ({len(task_groups)} unique tasks):")
        
        for task_name, task_results in task_groups.items():
            print(f"\n  🎯 {task_name}:")
            
            # Basic statistics
            gen_success = [r for r in task_results if r['generation_success']]
            
            print(f"    Generation: {len(gen_success)}/{len(task_results)} success")
            
            # Time statistics
            if gen_success:
                gen_times = [r['generation_time'] for r in gen_success]
                print(f"    Generation time: {statistics.mean(gen_times):.2f}s ± {statistics.stdev(gen_times) if len(gen_times) > 1 else 0:.2f}s")
            
            # Skill consistency analysis
            skills_by_seed = {}
            for result in gen_success:
                skills_by_seed[result['seed']] = set(result['generated_skills'])
            
            if len(skills_by_seed) > 1:
                all_skills = set()
                for skills in skills_by_seed.values():
                    all_skills.update(skills)
                
                # Find common skills across all seeds
                common_skills = set.intersection(*skills_by_seed.values()) if skills_by_seed else set()
                
                print(f"    Skill consistency: {len(common_skills)}/{len(all_skills)} common skills")
                print(f"    Skill variations:")
                
                for seed, skills in skills_by_seed.items():
                    unique_to_seed = skills - common_skills
                    if unique_to_seed:
                        print(f"      Seed {seed}: +{list(unique_to_seed)}")
                
                if common_skills:
                    print(f"    Common skills: {sorted(list(common_skills))}")
            
            # Error analysis
            errors = [r for r in task_results if r.get('error_message')]
            if errors:
                print(f"    Errors ({len(errors)}):")
                for error_result in errors:
                    print(f"      Seed {error_result['seed']}: {error_result['error_message']}")
    
    def generate_csv_report(self, output_file: Optional[str] = None):
        """Generate a CSV report of results."""
        if output_file is None:
            output_file = str(self.results_file.with_suffix('.csv'))
        
        import csv
        
        results = self.data.get("results", [])
        
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = [
                'task_name', 'seed', 'generation_success',
                'generation_time', 'num_skills', 'skills_list', 'error_message'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'task_name': result['task_name'],
                    'seed': result['seed'],
                    'generation_success': result['generation_success'],
                    'generation_time': result['generation_time'],
                    'num_skills': len(result['generated_skills']),
                    'skills_list': ','.join(result['generated_skills']),
                    'error_message': result.get('error_message', '')
                })
        
        print(f"📊 CSV report saved to: {output_file}")
    
    def print_statistical_summary(self):
        """Print statistical summary."""
        print("\n📈 STATISTICAL SUMMARY")
        print("=" * 50)
        
        results = self.data.get("results", [])
        
        # Overall success rates
        total = len(results)
        gen_success = sum(1 for r in results if r['generation_success'])
        
        print(f"Overall Success Rates:")
        print(f"  Generation: {gen_success}/{total} ({100*gen_success/total:.1f}%)")
        
        # Time statistics
        gen_times = [r['generation_time'] for r in results if r['generation_success']]
        
        if gen_times:
            print(f"\nGeneration Times:")
            print(f"  Mean: {statistics.mean(gen_times):.2f}s")
            print(f"  Median: {statistics.median(gen_times):.2f}s")
            print(f"  Std Dev: {statistics.stdev(gen_times) if len(gen_times) > 1 else 0:.2f}s")
            print(f"  Range: {min(gen_times):.2f}s - {max(gen_times):.2f}s")
        
        # Skill statistics
        all_skills = []
        for result in results:
            if result['generation_success']:
                all_skills.extend(result['generated_skills'])
        
        if all_skills:
            skill_counts = defaultdict(int)
            for skill in all_skills:
                skill_counts[skill] += 1
            
            print(f"\nSkill Statistics:")
            print(f"  Total skills generated: {len(all_skills)}")
            print(f"  Unique skills: {len(skill_counts)}")
            print(f"  Most common skills:")
            for skill, count in sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {skill}: {count} times")


def main():
    """Main entry point for analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze GenHRL Scientific Testing Results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("results_file", help="Path to JSON results file")
    parser.add_argument("--csv", action="store_true", help="Generate CSV report")
    parser.add_argument("--csv-output", help="Custom CSV output filename")
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not Path(args.results_file).exists():
        print(f"Error: Results file not found: {args.results_file}")
        sys.exit(1)
    
    # Create analyzer
    analyzer = ResultsAnalyzer(args.results_file)
    
    # Run analysis
    analyzer.print_detailed_analysis()
    analyzer.print_statistical_summary()
    
    # Generate CSV if requested
    if args.csv:
        analyzer.generate_csv_report(args.csv_output)


if __name__ == "__main__":
    main()