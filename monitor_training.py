#!/usr/bin/env python3
"""Monitor training progress until convergence."""

import time
import subprocess
import json
import sys
from datetime import datetime

def get_modal_app_id():
    """Get the latest Modal app ID for BFCL training."""
    result = subprocess.run(
        ["modal", "app", "list"], 
        capture_output=True, 
        text=True
    )
    for line in result.stdout.split('\n'):
        if 'verifiers' in line and 'running' in line:
            parts = line.split('│')
            if len(parts) > 1:
                return parts[1].strip()
    return None

def get_training_metrics(app_id):
    """Get reward metrics from Modal logs."""
    result = subprocess.run(
        ["modal", "app", "logs", app_id, "--tail", "100"],
        capture_output=True,
        text=True
    )
    
    rewards = []
    current_step = 0
    
    for line in result.stdout.split('\n'):
        if "│   " in line and " │ │" in line:
            # Extract reward value
            parts = line.split('│')
            if len(parts) >= 4:
                try:
                    reward = float(parts[-2].strip())
                    rewards.append(reward)
                except:
                    pass
        elif "global_step" in line:
            try:
                current_step = int(line.split()[-1])
            except:
                pass
                
    return {
        'rewards': rewards[-32:] if rewards else [],  # Last batch
        'avg_reward': sum(rewards[-32:]) / len(rewards[-32:]) if rewards else 0,
        'max_reward': max(rewards[-32:]) if rewards else 0,
        'current_step': current_step
    }

def main():
    print("🔍 Starting training monitor...")
    
    convergence_threshold = 0.8
    check_interval = 60  # Check every minute
    max_steps = 200
    
    while True:
        app_id = get_modal_app_id()
        if not app_id:
            print("❌ No running BFCL training found. Waiting...")
            time.sleep(check_interval)
            continue
            
        metrics = get_training_metrics(app_id)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{timestamp}] Step {metrics['current_step']}/{max_steps}")
        print(f"📊 Avg Reward: {metrics['avg_reward']:.3f}")
        print(f"🎯 Max Reward: {metrics['max_reward']:.3f}")
        print(f"📈 Reward Distribution: {len([r for r in metrics['rewards'] if r >= 1.0])}/32 perfect")
        
        # Check convergence
        if metrics['avg_reward'] >= convergence_threshold:
            print(f"\n✅ CONVERGED! Average reward {metrics['avg_reward']:.3f} >= {convergence_threshold}")
            break
            
        if metrics['current_step'] >= max_steps:
            print(f"\n⏱️ Reached max steps ({max_steps})")
            break
            
        # Show reward histogram
        if metrics['rewards']:
            print("📊 Reward histogram:")
            for threshold in [0.0, 0.2, 0.3, 0.5, 0.8, 1.0]:
                count = len([r for r in metrics['rewards'] if r >= threshold])
                bar = "█" * (count // 2)
                print(f"  >={threshold:.1f}: {bar} ({count})")
        
        time.sleep(check_interval)

if __name__ == "__main__":
    main()