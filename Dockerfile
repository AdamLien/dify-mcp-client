# Use the official Dify plugin daemon base image
FROM langgenius/dify-plugin-daemon:0.1.1-local

# Install Node.js environment (version 22) for stdio MCP servers and UI-TARS SDK
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
RUN apt-get install nodejs -y

# Install npm package manager
RUN curl -qL https://www.npmjs.com/install.sh | sh

# Install stable dependencies globally
RUN npm install -g yargs ts-node typescript

# Verify installation
RUN node --version && npm --version && npx --version

# Note: Node.js is required for:
# - Support stdio MCP server built with TypeScript SDK
# - UI-TARS SDK execution
# - @ui-tars packages are intentionally NOT pre-installed due to frequent updates (fetched via npx at runtime)