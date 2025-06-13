# Use the official Dify plugin daemon base image
FROM langgenius/dify-plugin-daemon:0.1.2-local

# Install Node.js environment (version 22)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
 && apt-get install -y nodejs \
 && npm install -g npm

# Install UI-TARS SDK to node_modules
COPY package.json tsconfig.json ./
RUN npm install --omit=dev --silent --no-fund --no-audit
# Install ts-node globally
COPY . .
RUN npm install -g --silent ts-node

# Verify installation
RUN node -e "require('@ui-tars/sdk'); console.log('UI-TARS SDK ready')"

# Note: Node.js is required for:
# - Support stdio MCP server built with TypeScript SDK
# - UI-TARS SDK execution