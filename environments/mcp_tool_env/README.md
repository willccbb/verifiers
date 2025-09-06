# mcp-tool-env

### Overview
- **Environment ID**: `mcp-tool-env`
- **Short description**: Multi-turn tool environment that spawns MCP (Model Context Protocol) servers in Prime Intellect sandboxes and exposes them as tools to models
- **Tags**: mcp, tools, multi-turn, sandbox, prime, npx, uv

### Datasets
- **Primary dataset(s)**: Configurable via `dataset_name` parameter (defaults to GSM8K)
- **Source links**: Uses `load_example_dataset` from verifiers
- **Split sizes**: Configurable via `num_examples` parameter

### Task
- **Type**: multi-turn tool use
- **Parser**: Default `Parser` with tool execution support
- **Rubric overview**: `ToolRubric` for MCP tool execution success and format adherence

### Architecture

This environment provides a bridge between the Verifiers framework and MCP (Model Context Protocol) servers by:

1. **Sandbox Management**: Creates isolated Prime Intellect sandboxes for secure MCP server execution
2. **Server Spawning**: Launches MCP servers using configurable npx/uv commands
3. **Tool Discovery**: Automatically discovers available tools from spawned MCP servers
4. **Tool Exposure**: Exposes MCP tools as native Python functions to the model

### Quickstart

Run an evaluation with filesystem MCP server:

```bash
uv run vf-eval mcp-tool-env -a '{
  "mcp_launch_commands": [
    {
      "name": "filesystem", 
      "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "setup_commands": ["mkdir -p /tmp/workspace"]
    }
  ],
  "dataset_name": "gsm8k",
  "num_examples": 10
}'
```

Configure with multiple MCP servers:

```bash
uv run vf-eval mcp-tool-env \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{
    "mcp_launch_commands": [
      {
        "name": "filesystem",
        "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        "setup_commands": ["mkdir -p /tmp/workspace"]
      },
      {
        "name": "python_tools",
        "command": ["uv", "run", "python", "-m", "mcp_this"],
        "setup_commands": ["pip install mcp-this"]
      }
    ],
    "dataset_name": "math",
    "sandbox_config": {
      "cpu_cores": 2,
      "memory_gb": 4,
      "disk_size_gb": 20
    }
  }'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `mcp_launch_commands` | List[Dict] | — | **Required**. List of MCP server configurations |
| `dataset_name` | str | `"gsm8k"` | Dataset to load for evaluation |
| `split` | str | `"train"` | Dataset split to use |
| `num_examples` | int | `100` | Number of examples to use (-1 for all) |
| `sandbox_config` | Dict | `{}` | Prime sandbox configuration (cpu_cores, memory_gb, disk_size_gb) |
| `max_turns` | int | `10` | Maximum number of conversation turns |

### MCP Launch Command Format

Each MCP server configuration should include:

```json
{
  "name": "server_identifier",
  "command": ["command", "arg1", "arg2"],
  "setup_commands": ["optional", "setup", "commands"]
}
```

**Examples:**

- **Filesystem Server**: 
  ```json
  {
    "name": "filesystem",
    "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
    "setup_commands": ["mkdir -p /workspace"]
  }
  ```

- **Python Tools Server**:
  ```json
  {
    "name": "python_tools", 
    "command": ["uv", "run", "python", "-m", "mcp_server"],
    "setup_commands": ["pip install mcp-server-package"]
  }
  ```

- **Custom MCP Server**:
  ```json
  {
    "name": "custom",
    "command": ["node", "my-mcp-server.js"],
    "setup_commands": ["npm install", "npm run build"]
  }
  ```

### Sandbox Configuration

Configure Prime Intellect sandbox resources:

```json
{
  "cpu_cores": 2,
  "memory_gb": 4, 
  "disk_size_gb": 20,
  "image": "python:3.11-slim"
}
```

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Weighted combination of tool execution success and format adherence |
| `tool_execution_success` | Percentage of MCP tool calls that executed without errors |
| `format_reward` | Adherence to expected conversation format |

### Prerequisites

1. **Prime CLI**: Install and configure the Prime CLI:
   ```bash
   pip install prime>=0.3.26
   prime config set-api-key
   ```

2. **MCP Tools**: The environment automatically installs `mcp2cli` and common MCP servers in the sandbox.

3. **API Access**: Ensure you have Prime Intellect API access and sufficient credits for sandbox usage.

### Pricing Considerations

Prime Sandbox usage is charged per hour while running:
- **CPU**: $0.05 per core per hour
- **Memory**: $0.01 per GB per hour  
- **Disk**: $0.001 per GB per hour

**Example**: 2 cores, 4GB RAM, 20GB disk = $0.16/hour

The environment automatically cleans up sandboxes after use to minimize costs.

### Error Handling

The environment includes robust error handling for:
- Sandbox creation failures
- MCP server startup issues
- Tool discovery problems
- Network connectivity issues
- Resource cleanup

### Security

- MCP servers run in isolated Prime sandboxes
- No persistent data storage between evaluations
- Automatic cleanup prevents resource leaks
- Configurable resource limits prevent abuse

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<p>No reports found. Run <code>uv run vf-eval mcp-tool-env -a '{"mcp_launch_commands": [{"name": "filesystem", "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}]}'</code> to generate one.</p>
<!-- vf:end:reports -->
