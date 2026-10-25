<div align="center">

# ChatLuna Forward Msg

[![npm](https://img.shields.io/npm/v/koishi-plugin-chatluna-forward-msg?style=flat-square)](https://www.npmjs.com/package/koishi-plugin-chatluna-forward-msg)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue?style=flat-square)](LICENSE)

**让 ChatLuna 读懂、发出 QQ 合并转发消息**

为 [ChatLuna](https://github.com/ChatLunaLab/chatluna) 提供读取、发送、伪造发送合并转发消息和按 URL 描述图片的工具。<br>
通过 OneBot 接口工作，支持 NapCat 与 LLBot。

</div>

---

## ✨ 功能

### 📥 读取合并转发
- **逐条解析**：按消息 ID 读取合并转发，输出每条消息的发送者 ID、昵称、时间、引用的消息 ID 与内容，供模型阅读。
- **嵌套转发**：递归解析转发中的转发，层数可设（1–8 层，默认 3 层，含读取的记录本身）。
- **图片描述**：读取时调用多模态模型为图片生成文字描述；默认跳过表情包，只保留占位。
- **视频与文件**：以链接形式写入内容，不做解析。
- **缓存**：同一消息 ID 的读取结果、按内容哈希的图片描述都存入数据库，默认保存 1 天，减少重复请求。

### 📤 以 Bot 身份发送
- **图文混排**：每个节点可以包含文字和图片 URL，图片由插件下载后转为 Base64 上传。
- **自动定位目标**：不指定目标时发送到当前会话；也可以指定群号，或指定 QQ 号私聊发送。
- **显示昵称**：可设置固定昵称，留空则自动推断。

### 🎭 伪造身份发送
- **自定义发送者**：每个节点可指定任意用户 ID 与昵称，同样支持文字和图片 URL。
- 仅供娱乐，见下方「⚠️ 使用声明」。

### 🖼️ 按 URL 描述图片
- **单张描述**：传入图片 URL，返回多模态模型生成的描述。
- **补充要求**：可附加描述要求，追加到提示词末尾。

---

## ✅ 前置条件

- Koishi，以及已安装并配置好模型的 [ChatLuna](https://github.com/ChatLunaLab/chatluna)（`koishi-plugin-chatluna` 1.3.36 及以上）。
- 需要 `database` 与 `http` 服务。
- 通过 OneBot 适配器接入的 NapCat 或 LLBot；其他平台的会话中工具会直接返回错误。
- 使用图片描述时，需要一个支持图片输入的模型。
- 为让模型拿到合并转发的消息 ID，请开启：
  - `chatluna` 插件「对话行为选项」中的 `attachForwardMsgIdToContext`；
  - `chatluna-character` 插件「对话设置」中的 `enableMessageId`。

## 📦 安装

在 Koishi 控制台的插件市场中搜索 `chatluna-forward-msg` 安装，或在 Koishi 项目目录中运行：

```bash
npm install koishi-plugin-chatluna-forward-msg
```

## ⚙️ 配置

| 分组 | 说明 |
| --- | --- |
| `protocolService` | 选择 OneBot 实现：`napcat`（默认）或 `llbot`。 |
| `readTool` | 读取工具：开关、工具名（默认 `read_forward_msg`）、工具描述、嵌套解析层数、读取时是否描述图片。 |
| `sendTool` | Bot 身份发送工具：开关、工具名（默认 `send_forward_msg`）、工具描述、显示昵称。 |
| `fakeTool` | 伪造身份发送工具：开关、工具名（默认 `send_fake_msg`）、工具描述。 |
| `describeImageTool` | 图片描述工具：开关、工具名（默认 `describe_image_by_url`）、工具描述。 |
| `imageService` | 图片描述所用的多模态模型、提示词、是否跳过表情包、图片任务并发数（默认 20）、图片下载超时（默认 20 秒）。 |
| `cacheService` | 缓存开关、有效期（默认 1 天）、过期清理间隔（默认 10 分钟）；`storagePath` 只用于在启动时删除旧版的本地缓存目录。 |

## ⚠️ 使用声明

**使用本插件即代表你已阅读并同意以下内容，如不同意，请立即卸载本插件：**

- 伪造身份发送合并转发的能力仅用于开发测试与娱乐场景。
- 使用者需自行确保符合当地法律法规与平台规则。
- 使用本插件造成的任何后果由使用者个人承担，**包括但不限于账号被限制、封禁等**。

受平台服务端限制，本插件生成的合并转发并不完美，其缺陷可以被人轻易识破，平台也能检测出异常。请勿用于欺骗、冒充、骚扰或其他违法违规用途。

## 📄 许可

本项目基于 [GPL-3.0](LICENSE) 开源。

## 🙏 致谢

- [ChatLuna](https://github.com/ChatLunaLab/chatluna)：插件所扩展的 Koishi 大模型对话框架，工具注册与模型调用都基于它。
- [Koishi](https://koishi.chat/)：机器人框架。
- [NapCat](https://github.com/NapNeko/NapCatQQ) 与 [LLBot](https://github.com/LLOneBot/LuckyLilliaBot)：提供合并转发收发所用的 OneBot 接口。
