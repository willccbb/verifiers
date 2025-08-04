#!/usr/bin/env python3
"""Continuous monitoring until convergence."""

import time
import subprocess
import json
import sys
import re
from datetime import datetime

def get_latest_wandb_run():
    """Get the latest BFCL training run from W&B."""
    result = subprocess.run(
        ["python", "scripts/monitor_wandb.py"],
        capture_output=True,
        text=True
    )
    
    # Find the latest running BFCL run
    for line in result.stdout.split('\n'):
        if 'bfcl-grpo_qwen2.5-0.5b-instruct (running)' in line:
            # Get the next lines for URL and metrics
            lines = result.stdout.split('\n')
            idx = lines.index(line)
            if idx + 1 < len(lines) and 'URL:' in lines[idx + 1]:
                url = lines[idx + 1].split('URL: ')[1].strip()
                return url
    return None

def monitor_modal_logs(app_id="ap-bnItgjMWmiV7CKAXzHUOWg"):
    """Monitor Modal logs for training progress."""
    print(f"\n{'='*60}")
    print(f"🔍 Monitoring BFCL Training - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")
    
    convergence_threshold = 0.8
    max_no_progress_steps = 50
    
    last_step = 0
    no_progress_count = 0
    best_reward = 0
    
    while True:
        try:
            # Get latest logs
            result = subprocess.run(
                ["modal", "app", "logs", app_id],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                print("❌ Failed to get logs, retrying...")
                time.sleep(10)
                continue
            
            # Parse rewards and steps
            rewards = []
            current_step = 0
            
            lines = result.stdout.split('\n')[-500:]  # Last 500 lines
            
            for line in lines:
                # Check for reward values
                if "│   " in line and " │" in line:
                    try:
                        parts = line.split('│')
                        if len(parts) >= 4:
                            reward_str = parts[-2].strip()
                            reward = float(reward_str)
                            rewards.append(reward)
                    except:
                        pass
                
                # Check for step number
                if "it/s]" in line:
                    match = re.search(r'(\d+)/200', line)
                    if match:
                        current_step = int(match.group(1))
                
                # Check for our custom progress callback
                if "📊 Step" in line and "Avg Reward" in line:
                    match = re.search(r'Step (\d+): Avg Reward = ([\d.]+)', line)
                    if match:
                        current_step = int(match.group(1))
                        avg_reward = float(match.group(2))
                        print(f"📊 Step {current_step}: Avg Reward = {avg_reward:.3f}")
            
            # Calculate metrics
            if rewards:
                recent_rewards = rewards[-32:]  # Last batch
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                max_reward = max(recent_rewards)
                perfect_count = len([r for r in recent_rewards if r >= 1.0])
                
                # Update best reward
                if avg_reward > best_reward:
                    best_reward = avg_reward
                
                # Display progress
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step {current_step}/200")
                print(f"📊 Batch Stats:")
                print(f"  • Average: {avg_reward:.3f}")
                print(f"  • Maximum: {max_reward:.3f}")
                print(f"  • Perfect (1.0): {perfect_count}/32")
                print(f"  • Best Avg So Far: {best_reward:.3f}")
                
                # Reward distribution
                print(f"\n📈 Reward Distribution:")
                for threshold in [0.0, 0.2, 0.3, 0.5, 0.8, 1.0]:
                    count = len([r for r in recent_rewards if r >= threshold])
                    bar = "█" * (count // 2)
                    print(f"  >={threshold:.1f}: {bar} ({count})")
                
                # Check convergence
                if avg_reward >= convergence_threshold:
                    print(f"\n✅ CONVERGED! Average reward {avg_reward:.3f} >= {convergence_threshold}")
                    print(f"🎉 Training successful at step {current_step}")
                    break
                
                # Check if stuck
                if current_step == last_step:
                    no_progress_count += 1
                else:
                    no_progress_count = 0
                    last_step = current_step
                
                if no_progress_count > max_no_progress_steps:
                    print(f"\n⚠️ Training appears stuck at step {current_step}")
                    break
            
            # Check if training completed
            if current_step >= 200:
                print(f"\n🏁 Training completed at step {current_step}")
                print(f"📊 Final best average reward: {best_reward:.3f}")
                break
            
            # Wait before next check
            time.sleep(30)
            
        except subprocess.TimeoutExpired:
            print("⏱️ Log retrieval timed out, retrying...")
            time.sleep(10)
        except KeyboardInterrupt:
            print("\n⛔ Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(10)
    
    # Final summary
    wandb_url = get_latest_wandb_run()
    if wandb_url:
        print(f"\n📊 W&B Run: {wandb_url}")
    
    print(f"\n{'='*60}")
    print("🏁 Training Monitor Complete")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Allow passing app ID as argument
    app_id = sys.argv[1] if len(sys.argv) > 1 else "ap-bnItgjMWmiV7CKAXzHUOWg"
    monitor_modal_logs(app_id)