#!/bin/bash
# Check training progress

echo "🔍 Checking BFCL training progress..."

# Get the latest running app
APP_ID=$(modal app list | grep "verifiers" | grep -E "running|ephemeral" | head -1 | awk -F'│' '{print $2}' | xargs)

if [ -z "$APP_ID" ]; then
    echo "❌ No running training found"
    exit 1
fi

echo "📱 App ID: $APP_ID"

# Get latest logs
echo "📊 Recent rewards:"
modal app logs $APP_ID 2>/dev/null | tail -500 | grep -E "│   [0-9]\." | tail -20

# Check current step
echo -e "\n📈 Training progress:"
modal app logs $APP_ID 2>/dev/null | tail -500 | grep -E "it/s\]$" | tail -5

# Count perfect rewards
echo -e "\n🎯 Perfect rewards (1.00):"
modal app logs $APP_ID 2>/dev/null | tail -500 | grep "│   1.00 │" | wc -l