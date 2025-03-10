# dify-mcp-client
`MCP Client` as Agent Strategy Plugin.
> [!IMPORTANT]
> Dify is not `MCP Server` but `MCP Host`. 

## How it works
Currently, each `MCP client` (ReAct Agent) node can connect a stdio `MCP server`.
1.  `Tool`, `Resource`, `Prompt` lists are converted into Dify Tools.
2.   Your selected LLM can see their `name`, `description`, `argument type`
3.   LLM calls Tools based on ReAct (Reason -> Act -> Observe Loop)

> [!NOTE]
> Most codes in this repository contains following files.
> #### Dify Officail Plugins
> https://github.com/langgenius/dify-official-plugins/tree/main/agent-strategies/cot_agent

## What I did (=no deleted, just added codes)
- copy `ReAct.py` and rename file as `mcpReAct.py`
- `config_json` GUI input by editing `mcpReAct.yaml` and `class mcpReActParams()` 

#### in `mcpReAct.py`, I added
- new 12 class methods for MCP 
- `__init__()` for initializing `AsyncExitStack` and `event loop`
- some codes in `_handle_invoke_action()` for MCP 
- MCP `setup` and `cleanup` in `_invoke()`
> [!IMPORTANT]
> ReAct while loop is as they are


## useful GitHub branch for developer

#### Dify Plugin SDKs (Python)
https://github.com/langgenius/dify-plugin-sdks

#### MCP Python SDK
https://github.com/modelcontextprotocol/python-sdk
<br>

> [!TIP]
> Especially useful following MCP client example<br>
> https://github.com/modelcontextprotocol/python-sdk/blob/main/examples/clients/simple-chatbot/mcp_simple_chatbot/main.py<br>

> [!NOTE]
> Dify plugin has `requirements.txt` which automatically installs python modules.<br>
> I write `mcp` in it, so you don't need to download MCP SDK.

## Before Start
> [!CAUTION]
> **No** `human in the loop` in this plugin, so connect **only reliable mcp server**.<br>
> To avoid it, decrease `max itereations`(default:`3`) to `1`, and use it repeatedly in Chatflow.<br>
> Don't forget to add a phrase such as *"ask for user's permission when calling tools"* in system prompt.

> [!WARNING]
> - Tools field shouldn't be blank. so I recommend you to **select built-in Dify plugin** like "current time".
> - `sse` connention doesn't support
