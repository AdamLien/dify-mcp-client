## Privacy
This plugin is released under the Apache License 2.0. Users are responsible for protecting their own privacy and securing their data according to the guidelines of their chosen LLM API platform. Please read the README for correct usage, as the default implementation does not include a Human-In-The-Loop mechanism.

### Q1: What data does the plugin access?
### A1:
The plugin itself does not actively collect personal information. However, the data it accesses depends on the connected MCP server. Depending on the server’s configuration, it may temporarily access user-related data (e.g., user names, file paths, etc.). Once the execution of the mcpReAct agent strategy node is complete, all such data is discarded.

### Q2: Does the plugin collect or store personal information?
### A2:
No. As an individual developer, I do not implement any persistent collection of personal data. The plugin is designed to run transiently, and any accessed information is not stored beyond the lifetime of a single operation.

### Q3: What about privacy in relation to LLM APIs?
### A3:
Since users choose their own LLM API platform, the privacy and security of data exchanged with those APIs are determined by the respective platforms. Please review and comply with the privacy policies of your chosen LLM provider.

### Q4: What is the user’s responsibility?
### A4:
Read the Documentation: Ensure you follow the usage guidelines provided in the README.
Trusted MCP Servers: Only connect to MCP servers you trust.
Privacy Management: Under Apache License 2.0, this plugin comes with no warranty regarding privacy or security. You are solely responsible for protecting your own data.