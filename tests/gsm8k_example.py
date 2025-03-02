from verifiers.envs import ToolEnv
from verifiers.tools.calculator import calculate

# Example few-shot for GSM8K with calculator
CALCULATOR_FEW_SHOT = [
    [
        {
            'role': 'user',
            'content': 'Janet has 3 times as many marbles as Tom. If Tom has 12 marbles, how many marbles do they have together?'
        },
        {
            'role': 'assistant',
            'content': '''<reasoning>
Let's solve this step by step:
1. First, calculate how many marbles Janet has (3 times Tom's)
</reasoning>
<tool>
{
    "name": "calculate",
    "args": {"expression": "3 * 12"}
}
</tool>
<result>36</result>

<reasoning>
2. Now we know:
   - Tom has 12 marbles
   - Janet has 36 marbles
   Let's add them together
</reasoning>
<tool>
{
    "name": "calculate",
    "args": {"expression": "36 + 12"}
}
</tool>
<result>48</result>

<reasoning>
Together they have 48 marbles:
- Tom: 12 marbles
- Janet: 36 marbles (3 times Tom's)
- Total: 48 marbles
</reasoning>
<answer>48 marbles</answer>'''
        }
    ]
]

def main():
    # Initialize environment with calculator tool
    env = ToolEnv(
        dataset="gsm8k",
        tools=[calculate],
        few_shot=CALCULATOR_FEW_SHOT,
        max_steps=5  # Most GSM8K problems need 2-3 calculation steps
    )
    
    # Example usage
    messages = [{"role": "user", "content": "A store sells shoes at a 40% markup from their purchase price. If they buy shoes for $50 and sell 3 pairs, what is their profit?"}]
    
    # Note: In practice, you would use the environment with a language model
    # This is just to demonstrate the expected interaction format
    expected_solution = '''<reasoning>
Let's solve this step by step:
1. Calculate the selling price with 40% markup
</reasoning>
<tool>
{
    "name": "calculate",
    "args": {"expression": "50 * 1.4"}
}
</tool>
<result>70.0</result>

<reasoning>
The selling price is $70 per pair.
Now let's calculate the profit on 3 pairs:
Profit = (Selling price - Cost) * Number of pairs
</reasoning>
<tool>
{
    "name": "calculate",
    "args": {"expression": "(70 - 50) * 3"}
}
</tool>
<result>60.0</result>

<reasoning>
The store makes:
- $20 profit per pair ($70 - $50)
- On 3 pairs, total profit is $60
</reasoning>
<answer>$60</answer>'''

if __name__ == "__main__":
    main() 