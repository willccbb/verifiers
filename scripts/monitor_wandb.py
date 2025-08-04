#!/usr/bin/env python3
"""
Monitor W&B training runs and provide analysis.
"""

import os
import wandb
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional
import argparse

def get_recent_runs(project: str = "verifiers", entity: Optional[str] = None, hours: int = 24) -> List[wandb.apis.public.Run]:
    """Get recent W&B runs from the last N hours."""
    api = wandb.Api()
    
    # Calculate the cutoff time
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    # Build the path
    path = f"{entity}/{project}" if entity else project
    
    try:
        runs = api.runs(path, filters={"created_at": {"$gte": cutoff_time.isoformat()}})
        return list(runs)
    except Exception as e:
        print(f"Error fetching runs: {e}")
        print(f"Trying without time filter...")
        try:
            runs = api.runs(path)
            # Manually filter recent runs
            recent_runs = []
            for run in runs:
                if run.created_at:
                    run_time = datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
                    if run_time >= cutoff_time:
                        recent_runs.append(run)
            return recent_runs[:10]  # Limit to 10 most recent
        except Exception as e2:
            print(f"Error: {e2}")
            return []

def analyze_run(run: wandb.apis.public.Run) -> Dict:
    """Analyze a single W&B run and extract key metrics."""
    analysis = {
        "name": run.name,
        "id": run.id,
        "state": run.state,
        "created_at": run.created_at,
        "url": run.url,
        "config": dict(run.config) if run.config else {},
        "summary": dict(run.summary) if run.summary else {},
        "metrics": {}
    }
    
    # Get history for detailed metrics
    try:
        history = run.history(samples=1000)
        if not history.empty:
            # Get latest metrics
            latest = history.iloc[-1]
            
            # Extract key training metrics
            metric_keys = [
                "train/reward", "train/reward_std", 
                "train/loss", "train/num_tokens",
                "train/rewards/total_tool_calls", 
                "train/rewards/function_call_reward",
                "eval/reward", "eval/reward_std",
                "completions/mean_length",
                "kl", "clip_ratio/region_mean"
            ]
            
            for key in metric_keys:
                if key in latest:
                    analysis["metrics"][key] = float(latest[key])
            
            # Calculate training progress
            if "train/global_step" in latest:
                analysis["metrics"]["global_step"] = int(latest["train/global_step"])
            
            # Analyze reward progression
            if "train/reward" in history.columns:
                rewards = history["train/reward"].dropna()
                if len(rewards) > 1:
                    analysis["metrics"]["reward_improvement"] = float(rewards.iloc[-1] - rewards.iloc[0])
                    analysis["metrics"]["reward_trend"] = "improving" if rewards.iloc[-1] > rewards.iloc[0] else "declining"
    except Exception as e:
        print(f"Error analyzing history for run {run.name}: {e}")
    
    return analysis

def print_run_analysis(analysis: Dict):
    """Print a formatted analysis of a run."""
    print(f"\n{'='*80}")
    print(f"Run: {analysis['name']} ({analysis['state']})")
    print(f"URL: {analysis['url']}")
    print(f"Created: {analysis['created_at']}")
    
    if analysis['config']:
        print(f"\nConfig highlights:")
        important_config = ["learning_rate", "beta", "temperature", "num_generations", "max_steps"]
        for key in important_config:
            if key in analysis['config']:
                print(f"  {key}: {analysis['config'][key]}")
    
    if analysis['metrics']:
        print(f"\nKey Metrics:")
        for key, value in analysis['metrics'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Provide recommendations
    print(f"\nAnalysis:")
    metrics = analysis['metrics']
    
    # Check reward
    if 'train/reward' in metrics:
        reward = metrics['train/reward']
        if reward < 0.1:
            print("  ⚠️  Low reward - model may not be learning the task well")
            print("     Consider: increasing temperature, adjusting reward function, or checking data quality")
        elif reward > 0.8:
            print("  ✅ High reward - model is performing well on the task")
    
    # Check KL divergence
    if 'kl' in metrics:
        kl = metrics['kl']
        if kl > 10:
            print("  ⚠️  High KL divergence - model deviating significantly from reference")
            print("     Consider: reducing learning rate or increasing beta")
        elif kl < 0.1:
            print("  ⚠️  Very low KL divergence - model may not be exploring enough")
            print("     Consider: decreasing beta or increasing temperature")
    
    # Check training progress
    if 'global_step' in metrics and 'max_steps' in analysis['config']:
        progress = metrics['global_step'] / analysis['config']['max_steps'] * 100
        print(f"  📊 Training progress: {progress:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Monitor W&B training runs")
    parser.add_argument("--project", default="verifiers", help="W&B project name")
    parser.add_argument("--entity", default=None, help="W&B entity/username")
    parser.add_argument("--hours", type=int, default=24, help="Look back N hours")
    parser.add_argument("--run-id", help="Analyze specific run ID")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    # Set up W&B API key from environment
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])
    
    if args.run_id:
        # Analyze specific run
        api = wandb.Api()
        path = f"{args.entity}/{args.project}/{args.run_id}" if args.entity else f"{args.project}/{args.run_id}"
        try:
            run = api.run(path)
            analysis = analyze_run(run)
            if args.json:
                print(json.dumps(analysis, indent=2))
            else:
                print_run_analysis(analysis)
        except Exception as e:
            print(f"Error fetching run {args.run_id}: {e}")
    else:
        # Get recent runs
        runs = get_recent_runs(args.project, args.entity, args.hours)
        
        if not runs:
            print("No recent runs found")
            return
        
        print(f"Found {len(runs)} recent runs in the last {args.hours} hours")
        
        analyses = []
        for run in runs:
            analysis = analyze_run(run)
            analyses.append(analysis)
            if not args.json:
                print_run_analysis(analysis)
        
        if args.json:
            print(json.dumps(analyses, indent=2))
        else:
            # Summary statistics
            print(f"\n{'='*80}")
            print("SUMMARY")
            print(f"{'='*80}")
            
            # Find best performing run
            best_reward = -float('inf')
            best_run = None
            for analysis in analyses:
                if 'train/reward' in analysis['metrics']:
                    if analysis['metrics']['train/reward'] > best_reward:
                        best_reward = analysis['metrics']['train/reward']
                        best_run = analysis
            
            if best_run:
                print(f"\nBest performing run: {best_run['name']}")
                print(f"  Reward: {best_reward:.4f}")
                print(f"  URL: {best_run['url']}")

if __name__ == "__main__":
    main()