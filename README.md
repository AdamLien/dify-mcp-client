# dify-mcp-client
MCP Client as Agent Strategy Plugin. Dify is not MCP Server but MCP Host in this plugin. Currently, each MCP client (ReAct Agent) node can connect a stdio MCP server.  At first, {Tool, Resource, Prompt} lists are converted into Dify Tools. Then, any LLM can see their {name, description, argument type} and call tools.
