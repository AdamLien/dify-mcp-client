# dify-mcp-client
`MCP Client` as Agent Strategy Plugin.
> [!IMPORTANT]
> Dify is not `MCP Server` but `MCP Host`. 

![showcase1](./_assets/arxiv_mcp_server_test.png)

## How it works
Currently, each `MCP client` (ReAct Agent) node can connect a `MCP server`.
1.  `Tool`, `Resource`, `Prompt` lists are converted into Dify Tools.
2.   Your selected LLM can see their `name`, `description`, `argument type`
3.   The LLM calls Tools based on the ReAct loop (Reason → Act → Observe).

> [!NOTE]
> Most of the code in this repository contains the following files.
> #### Dify Official Plugins / Agent Strategies
> https://github.com/langgenius/dify-official-plugins/tree/main/agent-strategies/cot_agent

## What I did
- Copied `ReAct.py` and renamed file as `mcpReAct.py`
- Added `config_json` GUI input field by editing `mcpReAct.yaml` and `class mcpReActParams()` 

### in mcpReAct.py, I added
- New 12 functions for MCP 
- `__init__()` for initializing `AsyncExitStack` and `event loop`
- Some codes in `_handle_invoke_action()` for MCP 
- MCP setup and cleanup in `_invoke()`
- Add SSE MCP client (v0.0.2)
> [!IMPORTANT]
> ReAct while loop is as they are




## Caution and Limitation
> [!CAUTION]
> This plugin does **not** implement a **human-in-the-loop** mechanism by default, so connect **reliable mcp server only**.<br>
> To avoid it, decrease `max itereations`(default:`3`) to `1`, and use this Agent node repeatedly in Chatflow.<br>
> However, agent memory is reset by the end of Workflow.<br>
> Use `Conversaton Variable` to save history and pass it to QUERY.  
> Don't forget to add a phrase such as
> *"ask for user's permission when calling tools"* in INSTRUCTION.

> [!WARNING]
> - The Tools field should not be left blank. so **select Dify tools** like "current time".

# How to use this plugin 

## Install the plugin from GitHub (online)
- Enter the following GitHub repository name
```
https://github.com/3dify-project/dify-mcp-client/
```
- Dify > PLUGINS > + Install plugin > INSTALL FROM > GitHub
![difyUI1](./_assets/plugin_install_online.png)

## Install the plugin from .difypkg file (offline)
- Go to Releases https://github.com/3dify-project/dify-mcp-client/releases
- Select suitable version of `.difypkg`
- Dify > PLUGINS > + Install plugin > INSTALL FROM > Local Package File
![difyUI2](./_assets/plugin_install_offline.png)

## How to handle errors when installing plugins?

**Issue**: If you encounter the error message: `plugin verification has been enabled, and the plugin you want to install has a bad signature`, how to handle the issue? <br>
**Solution**: Add the following line to the end of your `/docker/.env` configuration file: 
```
FORCE_VERIFYING_SIGNATURE=false
```
Run the following commands to restart the Dify service:
```bash
cd docker
docker compose down
docker compose up -d
```
Once this field is added, the Dify platform will allow the installation of all plugins that are not listed (and thus not verified) in the Dify Marketplace.
> [!TIP]
> Marketplace need Approval. If stars☆ reach 100, I'll consider to make PR for them.

## Source code plugin deploy
steps are as follows.
[how-to-develop-and-deploy-plugin](https://github.com/3dify-project/dify-mcp-client?tab=readme-ov-file#how-to-develop-and-deploy-plugin)

## Where does this plugin show up?
- It takes few minutes to install
- Once installed, you can use it any workflows as Agent node
- Select "mcpReAct" strategy (otherwise no MCP)
![asAgentStrategiesNode](./_assets/asAgentStrategiesNode.png)

## Config
MCP Agent Plugin node require config_json like this to command or URL to awake an MCP server
```
{
    "mcpservers":{
        "name_of_mcpserver":{
            "url": "http://host.docker.internal:8080/sse"
        },
    }
}
```
> [!WARNING]
> - Currently support one MCP server per Agent node

## Chatflow Example
![showcase2](./_assets/everything_mcp_server_test_resource.png)
#### I provide this Dify ChatFlow for testing dify mcp plugin as .yml.
https://github.com/3dify-project/dify-mcp-client/tree/main/test/chatflow
#### After download DSL(yml) file, import it in Dify and you can test MCP using "Everything MCP server"
https://github.com/modelcontextprotocol/servers/tree/main/src/everything

# How to convert stdio MCP server into SSE MCP server
## Way1: Edit MCP server's code
If fastMCP server, change like this
```diff
if __name__ == "__main__":
-    mcp.run(transport="stdio")
+    mcp.run(transport="sse")
```

## Way2: via mcp-proxy
```
\mcp-proxy>uv venv -p 3.12
.venv\Scripts\activate
uv tool install mcp-proxy
```
### Check Node.js has installed and npx(.cmd) Path 
(Mac/Linux)
```
which npx
```
(Windows)
```
where npx
```

```
C:\Program Files\nodejs\npx
C:\Program Files\nodejs\npx.cmd
C:\Users\USER_NAME\AppData\Roaming\npm\npx
C:\Users\USER_NAME\AppData\Roaming\npm\npx.cmd
```

```
C:\Users\nagisa\mcp-proxy>mcp-proxy --sse-port=8080 --pass-environment -- C:\Program Files\nodejs\npx.cmd --arg1 -y --arg2 @modelcontextprotocol/server-everything
```

> [!Warning]
> Additional argument for mcp-proxy. Be careful when you use it. There may be security risk such as XSS, CSRF. (default: no CORS allowed)
> ```
> --allow-origin='*'
> ```

```
(mcp_proxy) C:\User\USER_NAME\mcp-proxy>mcp-proxy --sse-port=8080 --pass-environment -- C:\Program Files\nodejs\npx.cmd --arg1 -y --arg2 @modelcontextprotocol/server-everything
DEBUG:root:Starting stdio client and SSE server
DEBUG:asyncio:Using proactor: IocpProactor
DEBUG:mcp.server.lowlevel.server:Initializing server 'example-servers/everything'
DEBUG:mcp.server.sse:SseServerTransport initialized with endpoint: /messages/
INFO:     Started server process [53104]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8080 (Press CTRL+C to quit)
```

#🔨 How to develop and deploy plugin

### General plugin dev guide
https://github.com/3dify-project/dify-mcp-client/blob/main/GUIDE.md

### Dify plugin SDK daemon
In my case (Windows 11) ,need to download dify-plugin-windows-amd64.exe (v0.0.3)<br>
Choose your OS-compatible verson at here:<br>
https://github.com/langgenius/dify-plugin-daemon/releases <br>
Rename it as dify.exe

### Reference  
https://docs.dify.ai/plugins/quick-start/develop-plugins/initialize-development-tools

> [!NOTE]
> You can skip this stage if you pull or download codes of this repo
> ```
> dify plugin init
> ```
> Initial settings are as follow 
> ![InitialDifyPluginSetting](./_assets/initial_mcp_plugin_settings.png)

### Install python module
Python3.12+ is compatible. Dify plugin official installation guide use pip, but I used uv.
```
uv init --python=python3.12
.venv\Scripts\activate
```
Install python modules for plugin development
```
uv add werkzeug==3.0.3
uv add flask
uv add dify_plugin
```

### Copy and rename env.example to .env
I changed `REMOTE_INSTALL_HOST` from `debug.dify.ai` to `localhost` 
(Docker Compose environment)
click bug icon button to see these information

### Change directory
```
cd mcp_client
```

### Do Once
```
pip install -r requirements.txt
```

### Activate Dify plugin
```
python -m main
```
(ctrl+C to stop)
> [!TIP]
> REMOTE_INSTALL_KEY of .env often changes.
> If you encounter error messages like `handshake failed, invalid key`, renew it.

### package into .difypkg
./mcp_client is my default root name
```
dify plugin package ./ROOT_OF_YOUR_PROJECT
```

## Useful GitHub repositories for developers

#### Dify Plugin SDKs
https://github.com/langgenius/dify-plugin-sdks

#### MCP Python SDK
https://github.com/modelcontextprotocol/python-sdk
<br>

> [!TIP]
> Especially useful following MCP client example<br>
> https://github.com/modelcontextprotocol/python-sdk/blob/main/examples/clients/simple-chatbot/mcp_simple_chatbot/main.py<br>

> [!NOTE]
> Dify plugin has `requirements.txt` which automatically installs python modules.<br>
> I include `mcp` in it, so you don't need to download the MCP SDK separately.
