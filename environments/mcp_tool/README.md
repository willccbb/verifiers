# mcp-tool

### Overview
- **Environment ID**: `mcp-tool`
- **Short description**: Multi-turn tool environment that spawns MCP servers in Prime Intellect sandboxes and exposes them as tools to models
- **Tags**: mcp, tools, multi-turn, sandbox, prime

### Datasets
- **Primary dataset(s)**: Configurable via `dataset_name` parameter (defaults to GSM8K)
- **Source links**: Uses `load_example_dataset` from verifiers
- **Split sizes**: Configurable via `num_examples` parameter

### Task
- **Type**: multi-turn tool use
- **Parser**: Default `Parser` with tool execution support
- **Rubric overview**: `ToolRubric` for MCP tool execution success and format adherence

### Quickstart
Run an evaluation with filesystem MCP server:

```bash
uv run vf-eval mcp-tool -a '{
  "mcp_launch_commands": [
    {
      "name": "filesystem", 
      "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "setup_commands": ["mkdir -p /tmp/workspace"]
    }
  ],
  "num_examples": 10
}'
```

Configure model and sampling:

```bash
uv run vf-eval mcp-tool \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{
    "mcp_launch_commands": [
      {
        "name": "filesystem",
        "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        "setup_commands": ["mkdir -p /tmp/workspace"]
      }
    ],
    "sandbox_config": {"cpu_cores": 2, "memory_gb": 4}
  }'
```

Notes:
- Requires Prime CLI configuration: `prime config set-api-key`
- Uses Node.js sandbox image for npx/npm support
- MCP servers run in isolated Prime Intellect sandboxes
- Automatic cleanup prevents resource leaks

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `mcp_launch_commands` | List[Dict] | — | **Required**. List of MCP server configurations |
| `dataset_name` | str | `"gsm8k"` | Dataset to load for evaluation |
| `split` | str | `"train"` | Dataset split to use |
| `num_examples` | int | `100` | Number of examples to use (-1 for all) |
| `sandbox_config` | Dict | `{}` | Prime sandbox configuration (cpu_cores, memory_gb, disk_size_gb) |
| `max_turns` | int | `10` | Maximum number of conversation turns |
| `prime_api_key` | str | `None` | Prime API key (or use CLI config) |

### MCP Server Configuration
Each MCP server configuration:

```json
{
  "name": "server_name",
  "command": ["npx", "-y", "@package/server", "/path"],
  "setup_commands": ["mkdir -p /path", "npm install"]
}
```

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | Weighted combination of tool execution success and format adherence |
| `tool_execution_success` | Percentage of MCP tool calls executed without errors |

### Prerequisites
1. **Prime CLI**: `pip install prime>=0.3.26` and `prime config set-api-key`
2. **API Access**: Valid Prime Intellect API credentials

### Pricing
Prime Sandbox usage (per hour while running):
- **CPU**: $0.05 per core/hour
- **Memory**: $0.01 per GB/hour  
- **Disk**: $0.001 per GB/hour

Example: 1 core, 2GB RAM, 10GB disk = $0.08/hour

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<p>No reports found. Run <code>uv run vf-eval mcp-tool -a '{"mcp_launch_commands": [{"name": "filesystem", "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}]}'</code> to generate one.</p>
<!-- vf:end:reports -->