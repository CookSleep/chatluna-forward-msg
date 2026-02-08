# ChatLuna Forward Msg

![Version](https://img.shields.io/badge/version-0.1.2-blue) ![License](https://img.shields.io/badge/license-GPLv3-brightgreen)

为 Koishi ChatLuna 提供 Napcat 合并转发消息的 **读取**、**发送**、**伪造发送** 与 **图片描述** 工具，便于在聊天场景中自动处理复杂消息内容。

## ✨ 功能特性

### 1. 📥 读取合并转发消息
- 支持读取 Napcat 合并转发消息并解析节点内容。
- 支持嵌套解析（可配置最大解析层数）。
- 支持在读取过程中对图片进行自动描述，输出更适合模型阅读的 JSON。

### 2. 📤 发送合并转发消息
- 支持以 Bot 身份发送合并转发消息。
- 支持文本与图片 URL 混合输入。
- 图片会自动下载并转为可发送格式（Base64）。

### 3. 🎭 伪造身份发送（娱乐用途）
- 支持指定用户 ID、昵称构造转发节点。
- 支持文本与图片 URL 混合输入。
- **伪造工具仅供娱乐，请勿用于欺骗、冒充、骚扰或其他违法违规用途。**

### 4. 🖼️ 图片描述工具
- 支持按图片 URL 直接生成描述。
- 可附加描述要求（requirement）。
- 可配置并发与请求超时参数。

## ⚙️ 主要配置

- `readTool`：读取转发消息工具配置。
- `sendTool`：Bot 身份发送工具配置。
- `fakeTool`：伪造身份发送工具配置。
- `describeImageTool`：图片描述工具配置。
- `imageService`：图片模型、提示词、并发与超时配置。

## ✅ 使用前置条件

为了让本插件正常读取和发送合并转发消息，请确保你已开启以下选项：

- 在 `chatluna` 插件的“对话行为选项”中启用：`attachForwardMsgIdToContext`。
- 在 `chatluna-character` 插件的“对话设置”中启用：`enableMessageId`。

## 🛡️ 使用声明

- 本项目中的伪造身份发送能力仅用于开发测试与娱乐场景。
- 使用者需自行确保符合当地法律法规与平台规则。
- 因不当使用产生的任何后果由使用者自行承担。

## 🤝 贡献

欢迎提交 Issue 或 Pull Request 来改进代码。
