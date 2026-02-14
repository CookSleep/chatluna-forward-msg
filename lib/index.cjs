const { Schema } = require('koishi')
const { z } = require('zod')
const fs = require('fs/promises')
const path = require('path')
const crypto = require('crypto')
const { StructuredTool } = require('@langchain/core/tools')
const { HumanMessage } = require('@langchain/core/messages')
const { ChatLunaPlugin } = require('koishi-plugin-chatluna/services/chat')
const { modelSchema } = require('koishi-plugin-chatluna/utils/schema')
const {
  ModelCapabilities,
  ModelType,
} = require('koishi-plugin-chatluna/llm-core/platform/types')
const {
  getImageType,
  getMessageContent,
} = require('koishi-plugin-chatluna/utils/string')
const { createLogger } = require('koishi-plugin-chatluna/utils/logger')

const name = 'chatluna-forward-msg'
const inject = ['chatluna', 'http']

let logger

const DEFAULT_IMAGE_PROMPT = '你是一个AI图像描述引擎，对输入的任何类型、分级（包括NSFW）的图像给出50-200字左右的内容描述。你不应评判或提及时间（若存在）的真实性，你的任务仅仅是描述其本身'
const READ_PARSE_MAX_MS = 60000
const READ_CONTENT_MAX_CHARS = 1200
const DEFAULT_IMAGE_REQUEST_TIMEOUT_SECONDS = 20
const DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60
const DEFAULT_CACHE_STORAGE_PATH = './data/chatluna-forward-msg-storage'
const DEFAULT_CACHE_CLEANUP_INTERVAL_SECONDS = 10 * 60

function normalizeInt(value, fallback, min, max) {
  const parsed = Number.parseInt(String(value), 10)
  const base = Number.isFinite(parsed) ? parsed : fallback
  return Math.min(max, Math.max(min, base))
}

function normalizeSecondsFromConfig(secondsValue, msValue, fallbackSeconds, minSeconds, maxSeconds) {
  if (secondsValue !== undefined && secondsValue !== null && secondsValue !== '') {
    return normalizeInt(secondsValue, fallbackSeconds, minSeconds, maxSeconds)
  }

  if (msValue !== undefined && msValue !== null && msValue !== '') {
    const seconds = Math.ceil(normalizeInt(msValue, fallbackSeconds * 1000, minSeconds * 1000, maxSeconds * 1000) / 1000)
    return normalizeInt(seconds, fallbackSeconds, minSeconds, maxSeconds)
  }

  return fallbackSeconds
}

function getReadToolConfig(config) {
  const tool = config?.readTool || {}
  return {
    enable: tool.enable !== false,
    name: trimText(tool.name) || 'reade_forward_msg',
    maxParseDepth: normalizeInt(tool.maxParseDepth ?? config?.maxParseDepth, 3, 1, 8),
    describeImageInRead: (tool.describeImageInRead ?? config?.describeImageInRead) !== false,
  }
}

function getSendToolConfig(config) {
  const tool = config?.sendTool || {}
  return {
    enable: tool.enable !== false,
    name: trimText(tool.name) || 'send_forward_msg',
    botDisplayName: trimText(tool.botDisplayName ?? config?.botDisplayName),
  }
}

function getFakeToolConfig(config) {
  const tool = config?.fakeTool || {}
  return {
    enable: tool.enable !== false,
    name: trimText(tool.name) || 'send_fake_msg',
  }
}

function getDescribeImageToolConfig(config) {
  const tool = config?.describeImageTool || {}
  return {
    enable: tool.enable !== false,
    name: trimText(tool.name) || 'describe_image_by_url',
  }
}

function getImageServiceConfig(config) {
  const image = config?.imageService || {}
  const requestTimeoutSeconds = normalizeSecondsFromConfig(
    image.requestTimeoutSeconds ?? config?.requestTimeoutSeconds,
    image.requestTimeoutMs ?? config?.requestTimeoutMs,
    DEFAULT_IMAGE_REQUEST_TIMEOUT_SECONDS,
    1,
    120,
  )
  return {
    model: trimText(image.model ?? config?.imageModel) || '无',
    prompt: trimText(image.prompt ?? config?.imagePrompt) || DEFAULT_IMAGE_PROMPT,
    taskConcurrency: normalizeInt(image.taskConcurrency ?? config?.imageTaskConcurrency, 20, 1, 100),
    requestTimeoutSeconds,
    requestTimeoutMs: requestTimeoutSeconds * 1000,
  }
}

function getCacheServiceConfig(config) {
  const cache = config?.cacheService || {}
  const ttlSeconds = normalizeSecondsFromConfig(
    cache.ttlSeconds,
    cache.ttlMs,
    DEFAULT_CACHE_TTL_SECONDS,
    60,
    30 * 24 * 60 * 60,
  )
  const cleanupIntervalSeconds = normalizeSecondsFromConfig(
    cache.cleanupIntervalSeconds,
    cache.cleanupIntervalMs,
    DEFAULT_CACHE_CLEANUP_INTERVAL_SECONDS,
    60,
    24 * 60 * 60,
  )
  return {
    enable: cache.enable !== false,
    ttlSeconds,
    ttlMs: ttlSeconds * 1000,
    storagePath: trimText(cache.storagePath) || DEFAULT_CACHE_STORAGE_PATH,
    cleanupIntervalSeconds,
    cleanupIntervalMs: cleanupIntervalSeconds * 1000,
  }
}

class ForwardMsgCacheService {
  constructor(ctx, config) {
    this.ctx = ctx
    this.config = config
    this.storagePath = ''
    this.initialized = false
    this.cleanupRunning = false
    this.lastCleanupAt = 0
  }

  async init() {
    const cacheConfig = getCacheServiceConfig(this.config)
    this.storagePath = path.resolve(process.cwd(), cacheConfig.storagePath)
    await fs.mkdir(this.storagePath, { recursive: true })
    this.initialized = true
    await this.cleanupExpired()
  }

  async ensureReady() {
    if (this.initialized) return
    await this.init()
  }

  hashKey(key) {
    return crypto.createHash('sha1').update(trimText(key)).digest('hex')
  }

  resolveCacheFile(key) {
    const hashed = this.hashKey(key)
    return path.join(this.storagePath, `${hashed}.json`)
  }

  async get(key) {
    const cacheConfig = getCacheServiceConfig(this.config)
    if (!cacheConfig.enable) return null
    const normalizedKey = trimText(key)
    if (!normalizedKey) return null

    await this.ensureReady()
    await this.maybeCleanup()
    const filePath = this.resolveCacheFile(normalizedKey)

    try {
      const raw = await fs.readFile(filePath, 'utf8')
      const data = JSON.parse(raw)
      if (!data || typeof data !== 'object') return null
      if (trimText(data.key) !== normalizedKey) return null
      if (!Number.isFinite(data.expiresAt) || data.expiresAt <= Date.now()) {
        await fs.unlink(filePath).catch(() => {})
        return null
      }
      return data.payload
    } catch {
      return null
    }
  }

  async set(key, payload) {
    const cacheConfig = getCacheServiceConfig(this.config)
    if (!cacheConfig.enable) return
    const normalizedKey = trimText(key)
    if (!normalizedKey) return

    await this.ensureReady()
    await this.maybeCleanup()
    const filePath = this.resolveCacheFile(normalizedKey)
    const now = Date.now()
    const record = {
      key: normalizedKey,
      payload,
      createdAt: now,
      updatedAt: now,
      expiresAt: now + cacheConfig.ttlMs,
    }
    const tmpFile = `${filePath}.${now}.tmp`
    await fs.writeFile(tmpFile, JSON.stringify(record), 'utf8')
    await fs.rename(tmpFile, filePath)
  }

  async maybeCleanup() {
    const cacheConfig = getCacheServiceConfig(this.config)
    const now = Date.now()
    if ((now - this.lastCleanupAt) < cacheConfig.cleanupIntervalMs) return
    this.lastCleanupAt = now
    await this.cleanupExpired()
  }

  async cleanupExpired() {
    if (this.cleanupRunning) return
    this.cleanupRunning = true
    try {
      await this.ensureReady()
      const files = await fs.readdir(this.storagePath, { withFileTypes: true })
      const now = Date.now()
      await Promise.all(files.map(async (file) => {
        if (!file.isFile() || !file.name.endsWith('.json')) return
        const filePath = path.join(this.storagePath, file.name)
        try {
          const raw = await fs.readFile(filePath, 'utf8')
          const data = JSON.parse(raw)
          if (!Number.isFinite(data?.expiresAt) || data.expiresAt <= now) {
            await fs.unlink(filePath).catch(() => {})
          }
        } catch {
          await fs.unlink(filePath).catch(() => {})
        }
      }))
    } finally {
      this.cleanupRunning = false
    }
  }
}

function buildReadCacheKey({ messageId, maxDepth, describeImageInRead }) {
  return `v1:${trimText(messageId)}:depth=${maxDepth}:img=${describeImageInRead ? 1 : 0}`
}

function apply(ctx, config) {
  logger = createLogger(ctx, name)

  const plugin = new ChatLunaPlugin(ctx, config, 'forward-msg', false)
  const cacheService = new ForwardMsgCacheService(ctx, config)
  let imageModelRef
  let imageModelName = ''

  async function ensureImageModelRef() {
    const target = resolveModelName(ctx, config)
    if (!target) return undefined

    if (imageModelRef && imageModelName === target) {
      return imageModelRef
    }

    try {
      imageModelRef = await ctx.chatluna.createChatModel(target)
      imageModelName = target
      return imageModelRef
    } catch (error) {
      logger.warn('加载图片模型失败: %s', error?.message || String(error))
      return undefined
    }
  }

  ctx.on('ready', async () => {
    await cacheService.init()
    modelSchema(ctx)
    await ensureImageModelRef()

    const readTool = getReadToolConfig(config)
    const sendTool = getSendToolConfig(config)
    const fakeTool = getFakeToolConfig(config)
    const describeImageTool = getDescribeImageToolConfig(config)

    if (readTool.enable) {
      plugin.registerTool(readTool.name, {
        selector() {
          return true
        },
        createTool() {
          return new ReadForwardMsgTool({ ctx, config, ensureImageModelRef, cacheService })
        },
      })
    }

    if (sendTool.enable) {
      plugin.registerTool(sendTool.name, {
        selector() {
          return true
        },
        createTool() {
          return new SendForwardMsgTool({ ctx, config })
        },
      })
    }

    if (fakeTool.enable) {
      plugin.registerTool(fakeTool.name, {
        selector() {
          return true
        },
        createTool() {
          return new SendFakeMsgTool({ ctx, config })
        },
      })
    }

    if (describeImageTool.enable) {
      plugin.registerTool(describeImageTool.name, {
        selector() {
          return true
        },
        createTool() {
          return new DescribeImageByUrlTool({ ctx, config, ensureImageModelRef })
        },
      })
    }
  })
}

function resolveModelName(ctx, config) {
  const imageConfig = getImageServiceConfig(config)
  if (imageConfig.model && imageConfig.model !== '无') {
    return imageConfig.model
  }

  const allModels = ctx.chatluna.platform.listAllModels(ModelType.llm).value || []
  const imageModel = allModels.find((model) =>
    Array.isArray(model.capabilities) &&
    model.capabilities.includes(ModelCapabilities.ImageInput),
  )

  if (!imageModel) return ''
  return `${imageModel.platform}/${imageModel.name}`
}

function getSessionFromRunnable(runnable) {
  return runnable?.configurable?.session
}

function ensureNapcatSession(runnable) {
  const session = getSessionFromRunnable(runnable)
  if (!session) {
    return { error: 'No session context available.' }
  }

  if (session.platform !== 'onebot') {
    return { error: 'This tool only supports OneBot (Napcat) platform.' }
  }

  const internal = session.bot?.internal
  if (!internal) {
    return { error: 'Missing OneBot internal API handle.' }
  }

  return { session, internal }
}

async function callApi(internal, action, params) {
  if (typeof internal._get === 'function') {
    return await internal._get(action, params)
  }

  throw new Error(`OneBot internal API does not support action: ${action}`)
}

function parseMimeFromDataUrl(dataUrl) {
  const matched = /^data:([^;]+);base64,/i.exec(dataUrl || '')
  return trimText(matched?.[1]).toLowerCase()
}

function classifyImageError(error) {
  const status = error?.response?.status || error?.status
  const code = trimText(error?.code || error?.cause?.code || error?.errno)
  const message = trimText(error?.message || error?.cause?.message || '')

  if (status) return `http_${status}`
  if (
    code === 'ETIMEDOUT' ||
    code === 'UND_ERR_CONNECT_TIMEOUT' ||
    /timeout|timed out/i.test(message)
  ) {
    return 'timeout'
  }
  if (code === 'EAI_AGAIN' || code === 'ENOTFOUND' || /getaddrinfo/i.test(message)) {
    return 'dns'
  }
  if (code === 'ECONNRESET' || code === 'ECONNREFUSED') return 'network'
  if (/abort/i.test(message)) return 'aborted'
  if (code) return `code_${code}`
  return 'unknown'
}

function maskLongUrl(url) {
  const text = trimText(url)
  if (!text) return ''
  return text.length > 160 ? `${text.slice(0, 157)}...` : text
}

async function readImageAsDataUrlWithMeta(ctx, url, timeout) {
  if (!url || typeof url !== 'string') {
    throw new Error('Invalid image url.')
  }

  const trimmed = url.trim()
  if (!trimmed) {
    throw new Error('Invalid image url.')
  }

  if (trimmed.startsWith('data:image') && trimmed.includes('base64,')) {
    return {
      dataUrl: trimmed,
      mime: parseMimeFromDataUrl(trimmed) || 'image/jpeg',
    }
  }

  let response
  try {
    response = await ctx.http(trimmed, {
      method: 'get',
      responseType: 'arraybuffer',
      timeout,
      headers: {
        'User-Agent':
          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
      },
    })
  } catch (error) {
    const reason = classifyImageError(error)
    logger.warn(
      'image download failed, reason=%s, timeoutMs=%d, url=%s, message=%s',
      reason,
      timeout,
      maskLongUrl(trimmed),
      trimText(error?.message || String(error)),
    )
    throw error
  }

  const buffer = Buffer.from(response.data)
  const mime = getImageType(buffer) || 'image/jpeg'
  return {
    dataUrl: `data:${mime};base64,${buffer.toString('base64')}`,
    mime: trimText(mime).toLowerCase(),
  }
}

async function readImageAsDataUrl(ctx, url, timeout) {
  const meta = await readImageAsDataUrlWithMeta(ctx, url, timeout)
  return meta.dataUrl
}

async function describeImageWithModel({ modelRef, imagePrompt, dataUrl }) {
  if (!modelRef?.value) {
    return '未配置可用多模态模型，无法生成图片描述。'
  }

  const model = modelRef.value
  if (
    !Array.isArray(model.modelInfo?.capabilities) ||
    !model.modelInfo.capabilities.includes(ModelCapabilities.ImageInput)
  ) {
    return '当前图片模型不支持图像输入。'
  }

  const content = [
    { type: 'text', text: imagePrompt },
    { type: 'image_url', image_url: { url: dataUrl } },
  ]

  let result
  try {
    result = await model.invoke([new HumanMessage({ content })])
  } catch (error) {
    const reason = classifyImageError(error)
    logger.warn(
      'image model invoke failed, reason=%s, message=%s',
      reason,
      trimText(error?.message || String(error)),
    )
    throw error
  }
  return getMessageContent(result.content)
}

function toTextSegment(text) {
  return {
    type: 'text',
    data: {
      text: String(text || ''),
    },
  }
}

function toImageSegment(dataUrl) {
  return {
    type: 'image',
    data: {
      file: dataUrl,
    },
  }
}

function parseToolMessages(messages) {
  if (Array.isArray(messages)) return messages

  if (typeof messages === 'string') {
    const trimmed = messages.trim()
    if (!trimmed) return []

    try {
      const parsed = JSON.parse(trimmed)
      if (Array.isArray(parsed)) return parsed
      if (parsed && typeof parsed === 'object') return [parsed]
    } catch {
      return [{ text: trimmed }]
    }
  }

  if (messages && typeof messages === 'object') {
    return [messages]
  }

  return []
}

function normalizeImageUrls(item) {
  const urls = []

  if (typeof item.image === 'string') urls.push(item.image)
  if (typeof item.imageUrl === 'string') urls.push(item.imageUrl)

  if (Array.isArray(item.images)) {
    urls.push(...item.images.filter((x) => typeof x === 'string'))
  }

  if (Array.isArray(item.imageUrls)) {
    urls.push(...item.imageUrls.filter((x) => typeof x === 'string'))
  }

  return urls.map((u) => u.trim()).filter(Boolean)
}

function trimText(value) {
  if (typeof value === 'string') return value.trim()
  if (typeof value === 'number' || typeof value === 'bigint' || typeof value === 'boolean') {
    return String(value).trim()
  }
  return ''
}

function escapeMarkdownAlt(text) {
  return trimText(text).replace(/\r?\n+/g, ' ').replace(/\]/g, '\\]').slice(0, 300)
}

function toMarkdownImage(description, source) {
  const url = trimText(source)
  if (!url) return '[图片]'
  const alt = escapeMarkdownAlt(description) || '图片'
  return `![${alt}](${url})`
}

async function mapWithConcurrency(items, concurrency, worker) {
  const list = Array.isArray(items) ? items : []
  if (list.length === 0) return []

  const limit = Math.max(1, Number.parseInt(String(concurrency || 1), 10) || 1)
  const size = Math.min(limit, list.length)
  const results = new Array(list.length)
  let cursor = 0

  async function runner() {
    while (true) {
      const index = cursor
      cursor += 1
      if (index >= list.length) break
      results[index] = await worker(list[index], index)
    }
  }

  await Promise.all(Array.from({ length: size }, () => runner()))
  return results
}

function pickTarget(session, input) {
  const groupId = trimText(input.groupId) || trimText(input.targetGroupId)
  const userId = trimText(input.userId) || trimText(input.targetUserId)
  const targetType = trimText(input.targetType)

  if (groupId || targetType === 'group') {
    const id = groupId || trimText(session.guildId) || trimText(session.channelId)
    if (!id) return { error: 'Missing groupId. Provide groupId or run inside a group session.' }
    return { type: 'group', id }
  }

  if (userId || targetType === 'private') {
    const id = userId || trimText(session.userId)
    if (!id) return { error: 'Missing userId for private target.' }
    return { type: 'private', id }
  }

  const fallbackGroup = trimText(session.guildId) || trimText(session.channelId)
  if (fallbackGroup) {
    return { type: 'group', id: fallbackGroup }
  }

  return { error: 'Cannot determine target. Please provide groupId or userId.' }
}

async function sendForwardNodes({ internal, target, nodes }) {
  if (target.type === 'group') {
    return await callApi(internal, 'send_group_forward_msg', {
      group_id: target.id,
      message_seq: 0,
      messages: nodes,
    })
  }

  return await callApi(internal, 'send_private_forward_msg', {
    user_id: target.id,
    message_seq: 0,
    messages: nodes,
  })
}

async function fetchForwardMessages(internal, messageId) {
  const normalized = trimText(messageId)
  if (!normalized) return []

  const extractMessagesArray = (data) => {
    if (Array.isArray(data)) return data
    if (!data || typeof data !== 'object') return null

    const candidates = [
      data?.messages,
      data?.data?.messages,
      data?.data?.data?.messages,
      data?.result?.messages,
    ]
    for (const item of candidates) {
      if (Array.isArray(item)) return item
    }
    return null
  }

  const requestForwardByPayload = async (payload) => {
    try {
      const data = await callApi(internal, 'get_forward_msg', payload)
      const messages = extractMessagesArray(data)
      if (Array.isArray(messages)) return messages
      return []
    } catch {
      return null
    }
  }

  const primaryAttempts = [
    { message_id: normalized },
    { id: normalized },
  ]

  for (const payload of primaryAttempts) {
    const messages = await requestForwardByPayload(payload)
    if (Array.isArray(messages) && messages.length > 0) return messages
  }

  const getMessageById = async (id) => {
    try {
      if (typeof internal.getMsg === 'function') {
        return await internal.getMsg(id)
      }
      return await callApi(internal, 'get_msg', { message_id: id })
    } catch {
      return null
    }
  }

  const collectForwardIdsFromSegments = (segments) => {
    const result = []
    if (!Array.isArray(segments)) return result
    for (const segment of segments) {
      const type = trimText(segment?.type)
      if (type !== 'forward') continue
      const data = segment?.data || {}
      const id = trimText(data.id) || trimText(data.message_id)
      if (id) result.push(id)
    }
    return result
  }

  const msgData = await getMessageById(normalized)
  const nestedIds = [
    ...collectForwardIdsFromSegments(msgData?.message),
    ...collectForwardIdsFromSegments(msgData?.data?.message),
  ]
  const uniqueNestedIds = Array.from(new Set(nestedIds))

  for (const forwardId of uniqueNestedIds) {
    const messagesById = await requestForwardByPayload({ id: forwardId })
    if (Array.isArray(messagesById) && messagesById.length > 0) return messagesById

    const messagesByMsgId = await requestForwardByPayload({ message_id: forwardId })
    if (Array.isArray(messagesByMsgId) && messagesByMsgId.length > 0) return messagesByMsgId
  }

  const fallback = await requestForwardByPayload({ message_id: normalized })
  return Array.isArray(fallback) ? fallback : []
}

async function fetchMessageById(internal, messageId) {
  const normalized = trimText(messageId)
  if (!normalized) return null

  const payloads = dedupeOrdered([
    normalized,
    JSON.stringify({ message_id: normalized }),
    JSON.stringify({ id: normalized }),
    JSON.stringify({ message_seq: normalized }),
    JSON.stringify({ real_id: normalized }),
    JSON.stringify({ real_seq: normalized }),
  ])
  const parsePayload = (payload) => {
    if (!payload) return null
    if (payload[0] !== '{') return payload
    try {
      return JSON.parse(payload)
    } catch {
      return null
    }
  }

  const pickUsableMessage = (data) => {
    const message = normalizeOnebotMessageObject(data)
    if (!message || typeof message !== 'object') return null
    if (
      extractMessageId(message) ||
      trimText(message?.raw_message) ||
      Array.isArray(message?.message) ||
      Array.isArray(message?.content)
    ) {
      return message
    }
    return null
  }

  for (const payload of payloads) {
    const parsed = parsePayload(payload)
    if (!parsed) continue

    if (typeof internal.getMsg === 'function') {
      try {
        const data = await internal.getMsg(parsed)
        const message = pickUsableMessage(data)
        if (message) return message
      } catch {}
    }

    if (typeof parsed === 'object') {
      try {
        const data = await callApi(internal, 'get_msg', parsed)
        const message = pickUsableMessage(data)
        if (message) return message
      } catch {}
    }
  }

  return null
}

function normalizeOnebotMessageObject(message) {
  if (!message || typeof message !== 'object') return null
  const direct = message
  const nested = message?.data && typeof message.data === 'object' ? message.data : null
  const target = nested?.message || nested?.raw_message || nested?.message_id ? nested : direct
  if (!target || typeof target !== 'object') return null
  return target
}

function extractReplyIdFromRawMessage(rawMessage) {
  const raw = trimText(rawMessage)
  if (!raw) return ''
  const matched = /\[CQ:reply,[^\]]*?\bid=([^,\]]+)/i.exec(raw)
  return trimText(matched?.[1])
}

function normalizeExternalReplyMessage(message) {
  const normalized = normalizeOnebotMessageObject(message)
  if (!normalized) return null

  const sender = normalized?.sender || {}
  const nickname =
    trimText(sender?.nickname) ||
    trimText(sender?.card) ||
    trimText(normalized?.nickname) ||
    trimText(normalized?.user_id)

  const userId =
    trimText(sender?.user_id) ||
    trimText(normalized?.user_id) ||
    trimText(normalized?.sender_id)

  const contentParts = []
  const segments = getMessageSegments(normalized)
  for (const segment of segments) {
    const type = trimText(segment?.type)
    const data = segment?.data || {}
    if (type === 'text') {
      const text = trimText(data.text)
      if (text) contentParts.push(text)
      continue
    }
    if (type === 'image' || type === 'mface') {
      const source = trimText(data.url) || trimText(data.file) || trimText(data.path)
      if (source) contentParts.push(toMarkdownImage('', source))
    }
  }

  if (contentParts.length === 0) {
    const raw = trimText(normalized?.raw_message)
    if (raw) contentParts.push(raw)
  }

  return {
    messageId: extractMessageId(normalized),
    nickname,
    userId,
    time: formatMessageTime(normalized?.time),
    content: dedupeOrdered(contentParts).join('').slice(0, READ_CONTENT_MAX_CHARS),
  }
}

async function fetchBotQQNickname(internal) {
  try {
    if (typeof internal.getLoginInfo === 'function') {
      const data = await internal.getLoginInfo()
      return trimText(data?.nickname)
    }
    const data = await callApi(internal, 'get_login_info', {})
    return trimText(data?.nickname)
  } catch {
    return ''
  }
}

async function fetchGroupMember(internal, groupId, userId) {
  try {
    if (!groupId || !userId) return null
    if (typeof internal.getGroupMemberInfo === 'function') {
      return await internal.getGroupMemberInfo(groupId, userId, false)
    }
    return await callApi(internal, 'get_group_member_info', {
      group_id: groupId,
      user_id: userId,
      no_cache: false,
    })
  } catch {
    return null
  }
}

function collectNicknameCandidates(member, userId, fallbackNames = []) {
  const candidates = [
    member?.card,
    member?.remark,
    member?.displayName,
    member?.nick,
    member?.nickname,
    member?.name,
    member?.user?.nickname,
    member?.user?.name,
    ...fallbackNames,
    userId,
  ]

  return candidates
    .map((name) => trimText(name))
    .filter(Boolean)
}

async function resolveUserDisplayName({
  internal,
  target,
  userId,
  fallbackNames = [],
  nameCache,
}) {
  const normalizedUserId = trimText(userId)
  if (!normalizedUserId) return ''

  if (nameCache && nameCache.has(normalizedUserId)) {
    return nameCache.get(normalizedUserId)
  }

  let member = null
  if (target?.type === 'group') {
    member = await fetchGroupMember(internal, target.id, normalizedUserId)
  }

  const name =
    collectNicknameCandidates(member, normalizedUserId, fallbackNames)[0] ||
    normalizedUserId

  if (nameCache) {
    nameCache.set(normalizedUserId, name)
  }

  return name
}

async function resolveBotDisplayName({ internal, session, target, config, botId, nameCache }) {
  const sendToolConfig = getSendToolConfig(config)
  const qqNickname = await fetchBotQQNickname(internal)
  const fallbackNames = [
    qqNickname,
    sendToolConfig.botDisplayName,
    trimText(session.bot?.user?.name),
  ].filter(Boolean)

  if (target?.type === 'group') {
    return await resolveUserDisplayName({
      internal,
      target,
      userId: botId,
      fallbackNames,
      nameCache,
    })
  }

  return (
    fallbackNames[0] ||
    botId
  )
}

function extractSegmentsFromContent(content) {
  if (Array.isArray(content)) return content
  if (typeof content === 'string') return [toTextSegment(content)]
  return []
}

function dedupeOrdered(list) {
  const seen = new Set()
  const result = []
  for (const item of list || []) {
    const value = trimText(item)
    if (!value || seen.has(value)) continue
    seen.add(value)
    result.push(value)
  }
  return result
}

function formatMessageTime(value) {
  if (value === null || value === undefined || value === '') return ''

  let ms = NaN
  if (typeof value === 'number') {
    ms = value < 1e12 ? value * 1000 : value
  } else if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!trimmed) return ''
    const numeric = Number(trimmed)
    if (Number.isFinite(numeric)) {
      ms = numeric < 1e12 ? numeric * 1000 : numeric
    } else {
      const parsed = Date.parse(trimmed)
      ms = Number.isFinite(parsed) ? parsed : NaN
    }
  } else {
    return ''
  }

  if (!Number.isFinite(ms)) return ''
  const date = new Date(ms)
  if (Number.isNaN(date.getTime())) return ''

  const parts = new Intl.DateTimeFormat('zh-CN', {
    timeZone: 'Asia/Shanghai',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  }).formatToParts(date)

  const map = {}
  for (const part of parts) {
    map[part.type] = part.value
  }

  return `${map.year || '0000'}-${map.month || '00'}-${map.day || '00'} ${map.hour || '00'}:${map.minute || '00'}:${map.second || '00'}`
}

function extractMessageId(message) {
  return (
    trimText(message?.message_id) ||
    trimText(message?.id) ||
    trimText(message?.real_id) ||
    trimText(message?.message_seq) ||
    ''
  )
}

function collectMessageIndexKeys(message) {
  const keys = dedupeOrdered([
    trimText(message?.message_id),
    trimText(message?.id),
    trimText(message?.real_id),
    trimText(message?.message_seq),
    trimText(message?.real_seq),
  ]).filter(Boolean)
  return keys
}

function normalizeReadBudget(budget) {
  if (budget && typeof budget === 'object') return budget
  return {
    startedAt: Date.now(),
    maxMs: READ_PARSE_MAX_MS,
    truncated: false,
    timedOut: false,
  }
}

function shouldStopReadParse(budget) {
  if (!budget) return false
  const now = Date.now()
  if ((now - budget.startedAt) > budget.maxMs) {
    budget.truncated = true
    budget.timedOut = true
    return true
  }
  return false
}

function getReadBudgetRemainingMs(budget) {
  if (!budget) return Number.POSITIVE_INFINITY
  return Math.max(0, budget.maxMs - (Date.now() - budget.startedAt))
}

function getMessageSegments(message) {
  const fromContent = extractSegmentsFromContent(message?.content)
  if (fromContent.length > 0) return fromContent

  const fromDataContent = extractSegmentsFromContent(message?.data?.content)
  if (fromDataContent.length > 0) return fromDataContent

  const fromMessage = extractSegmentsFromContent(message?.message)
  if (fromMessage.length > 0) return fromMessage

  const fromDataMessage = extractSegmentsFromContent(message?.data?.message)
  if (fromDataMessage.length > 0) return fromDataMessage

  const raw = trimText(message?.raw_message)
  if (raw) return [toTextSegment(raw)]
  return []
}

async function parseForwardMessages({
  ctx,
  internal,
  ensureImageModelRef,
  config,
  messages,
  depth,
  visited,
  budget,
}) {
  const result = []
  const stagedEntries = []
  const imageTasks = []
  const readToolConfig = getReadToolConfig(config)
  const imageConfig = getImageServiceConfig(config)
  const parseBudget = normalizeReadBudget(budget)

  for (const message of messages || []) {
    if (shouldStopReadParse(parseBudget)) break

    const sender = message?.sender || {}
    const senderId =
      trimText(sender.user_id) || trimText(sender.uin) || trimText(message?.user_id) || ''
    const senderName =
      trimText(sender.nickname) || trimText(message?.nickname) || senderId || 'unknown'

    const messageId = extractMessageId(message)

    const entry = {
      messageId,
      indexKeys: collectMessageIndexKeys(message),
      nickname: senderName,
      userId: senderId,
      time: formatMessageTime(message?.time),
      quote: '',
      contentParts: [],
      forwards: [],
    }

    const segments = getMessageSegments(message)
    const imageSources = []

    for (const segment of segments) {
      const type = segment?.type
      const data = segment?.data || {}

      if (type === 'text') {
        const text = trimText(data.text)
        if (text) entry.contentParts.push(text)
        continue
      }

      if (type === 'reply') {
        const quoteId =
          trimText(data.id) ||
          trimText(data.message_id) ||
          trimText(data.real_id) ||
          trimText(data.message_seq) ||
          trimText(data.real_seq) ||
          trimText(data.seq)
        if (quoteId && !entry.quote) entry.quote = quoteId
        continue
      }

      if (type === 'image' || type === 'mface') {
        const source = trimText(data.url) || trimText(data.file) || trimText(data.path)
        if (source) imageSources.push(source)
        continue
      }

      if (type === 'node') {
        if (shouldStopReadParse(parseBudget)) break
        const nested = await parseForwardMessages({
          ctx,
          internal,
          ensureImageModelRef,
          config,
          messages: [{
            sender: {
              user_id: data.user_id,
              nickname: data.nickname,
            },
            time: message?.time,
            content: data.content,
          }],
          depth,
          visited,
          budget: parseBudget,
        })
        if (nested.length) {
          entry.forwards.push(...nested)
        }
        continue
      }

      if (type === 'forward') {
        if (depth <= 0) {
          entry.contentParts.push('[聊天记录]')
          continue
        }
        if (shouldStopReadParse(parseBudget)) break

        const nestedId = trimText(data.id) || trimText(data.message_id)
        const inlineContent = Array.isArray(data.content) ? data.content : []

        if (inlineContent.length > 0) {
          const nested = await parseForwardMessages({
            ctx,
            internal,
            ensureImageModelRef,
            config,
            messages: inlineContent,
            depth: depth - 1,
            visited,
            budget: parseBudget,
          })
          if (nested.length) {
            entry.forwards.push(...nested)
          }
          continue
        }

        if (!nestedId || visited.has(nestedId)) continue
        visited.add(nestedId)

        try {
          if (shouldStopReadParse(parseBudget)) break
          const nestedMessages = await fetchForwardMessages(internal, nestedId)
          const nested = await parseForwardMessages({
            ctx,
            internal,
            ensureImageModelRef,
            config,
            messages: nestedMessages,
            depth: depth - 1,
            visited,
            budget: parseBudget,
          })
          if (nested.length) {
            entry.forwards.push(...nested)
          }
        } catch (error) {
          entry.contentParts.push(`嵌套转发读取失败(${nestedId})：${error?.message || String(error)}`)
        }
      }
    }

    if (!entry.quote) {
      const quoteIdFromRaw = extractReplyIdFromRawMessage(message?.raw_message)
      if (quoteIdFromRaw) entry.quote = quoteIdFromRaw
    }

    if (imageSources.length) {
      const canDescribeImages =
        readToolConfig.describeImageInRead &&
        !shouldStopReadParse(parseBudget) &&
        getReadBudgetRemainingMs(parseBudget) > 3000

      if (!canDescribeImages) {
        for (const source of imageSources) {
          entry.contentParts.push(toMarkdownImage('', source))
        }
        logger.info(
          'image parse batch skipped describe, messageId=%s, imageCount=%d, reason=%s',
          entry.messageId || '-',
          imageSources.length,
          readToolConfig.describeImageInRead ? 'budget_low' : 'disabled',
        )
      } else {
        for (const source of imageSources) {
          imageTasks.push({
            entry,
            source,
            messageId: entry.messageId || '-',
          })
        }
      }
    }

    stagedEntries.push(entry)
    if (shouldStopReadParse(parseBudget)) break
  }

  if (imageTasks.length > 0) {
    const totalStart = Date.now()
    const groupedImageTasks = new Map()
    for (const task of imageTasks) {
      const key = trimText(task.source)
      if (!key) continue
      if (!groupedImageTasks.has(key)) {
        groupedImageTasks.set(key, {
          source: task.source,
          refs: [],
        })
      }
      groupedImageTasks.get(key).refs.push(task)
    }
    const uniqueImageTasks = Array.from(groupedImageTasks.values())
    const dedupSaved = Math.max(0, imageTasks.length - uniqueImageTasks.length)
    logger.info(
      'image parse global batch start, imageCount=%d, uniqueImageCount=%d, dedupSaved=%d, concurrency=%d, timeoutSeconds=%d',
      imageTasks.length,
      uniqueImageTasks.length,
      dedupSaved,
      imageConfig.taskConcurrency,
      imageConfig.requestTimeoutSeconds,
    )
    const images = await mapWithConcurrency(
      uniqueImageTasks,
      imageConfig.taskConcurrency,
      async (task) => {
        const imageStart = Date.now()
        if (shouldStopReadParse(parseBudget) || getReadBudgetRemainingMs(parseBudget) <= 500) {
          return { ...task, description: '', reason: 'budget_low', costMs: Date.now() - imageStart }
        }
        let description = ''
        let reason = ''
        try {
          const imageMeta = await readImageAsDataUrlWithMeta(ctx, task.source, imageConfig.requestTimeoutMs)
          if (imageMeta.mime === 'image/gif') {
            reason = 'gif_skipped'
          } else {
            const modelRef = await ensureImageModelRef()
            description = await describeImageWithModel({
              modelRef,
              imagePrompt: imageConfig.prompt,
              dataUrl: imageMeta.dataUrl,
            })
          }
        } catch (error) {
          reason = classifyImageError(error)
          description = `图片解析失败(${reason})：${error?.message || String(error)}`
        }
        return { ...task, description, reason, costMs: Date.now() - imageStart }
      },
    )
    for (const image of images) {
      const refs = Array.isArray(image.refs) ? image.refs : []
      for (const ref of refs) {
        ref.entry.contentParts.push(toMarkdownImage(image.description, image.source))
      }
    }
    const reasonCount = {}
    for (const image of images) {
      const reason = trimText(image.reason)
      if (!reason) continue
      reasonCount[reason] = (reasonCount[reason] || 0) + 1
    }
    logger.info(
      'image parse global batch done, imageCount=%d, uniqueImageCount=%d, dedupSaved=%d, described=%d, placeholder=%d, costMs=%d, reasons=%s',
      imageTasks.length,
      images.length,
      dedupSaved,
      images.filter((x) => trimText(x.description) && !trimText(x.reason)).length,
      images.filter((x) => !trimText(x.description)).length,
      Date.now() - totalStart,
      JSON.stringify(reasonCount),
    )
  }

  for (const entry of stagedEntries) {
    const dedupedContentParts = dedupeOrdered(entry.contentParts)
    const content = dedupedContentParts.join('').slice(0, READ_CONTENT_MAX_CHARS)
    if (content || entry.forwards.length > 0) {
      result.push({
        messageId: entry.messageId,
        indexKeys: Array.isArray(entry.indexKeys) ? entry.indexKeys : [],
        nickname: entry.nickname,
        userId: entry.userId,
        time: entry.time,
        quote: entry.quote,
        content,
        forward: entry.forwards,
      })
    }
  }

  const seen = new Set()
  const compact = []
  for (const item of result) {
    const forwardIds = Array.isArray(item.forward)
      ? item.forward.map((child) => trimText(child?.messageId)).join(',')
      : ''
    const key = `${item.messageId}|${item.userId}|${item.nickname}|${item.time}|${item.quote}|${item.content}|${forwardIds}`
    if (seen.has(key)) continue
    seen.add(key)
    compact.push(item)
  }

  return compact
}

class ReadForwardMsgTool extends StructuredTool {
  constructor(deps) {
    super({})
    this.deps = deps
    this.name = getReadToolConfig(deps.config).name
    this.description =
      '读取 Napcat 合并转发消息内容。支持递归解析嵌套合并转发，只提取文本与图片，并可调用多模态模型生成图片描述。'
    this.schema = z.object({
      messageId: z.string().min(1).describe('目标合并转发消息 ID。'),
      maxDepth: z
        .number()
        .int()
        .min(1)
        .max(8)
        .optional()
        .describe('嵌套合并转发解析层数（含读取的记录本身），默认使用插件配置值。'),
    })
  }

  async _call(input, _manager, runnable) {
    try {
      const checked = ensureNapcatSession(runnable)
      if (checked.error) return checked.error

      const { internal } = checked
      const messageId = trimText(input.messageId)
      const readToolConfig = getReadToolConfig(this.deps.config)
      const maxDepth = Number.isInteger(input.maxDepth)
        ? input.maxDepth
        : readToolConfig.maxParseDepth
      // maxDepth includes the root forward message itself. "depth" tracks remaining nested forwards.
      const nestedDepth = Math.max(0, maxDepth - 1)
      const cacheKey = buildReadCacheKey({
        messageId,
        maxDepth,
        describeImageInRead: readToolConfig.describeImageInRead,
      })
      const cached = await this.deps.cacheService.get(cacheKey)
      if (Array.isArray(cached)) {
        logger.info(
          'reade_forward_msg cache hit, count: %d, messageId: %s, maxDepth: %d',
          cached.length,
          messageId,
          maxDepth,
        )
        return JSON.stringify(cached, null, 2)
      }

      const rootMessages = await fetchForwardMessages(internal, messageId)
      const budget = {
        startedAt: Date.now(),
        maxMs: READ_PARSE_MAX_MS,
        truncated: false,
        timedOut: false,
      }
      logger.info('reade_forward_msg get_forward_msg count: %d, messageId: %s', rootMessages.length, messageId)
      const parsed = await parseForwardMessages({
        ctx: this.deps.ctx,
        internal,
        ensureImageModelRef: this.deps.ensureImageModelRef,
        config: this.deps.config,
        messages: rootMessages,
        depth: nestedDepth,
        visited: new Set([messageId]),
        budget,
      })
      if (budget.timedOut) {
        throw new Error(`read forward message parse timeout after ${READ_PARSE_MAX_MS}ms`)
      }
      logger.info(
        'reade_forward_msg parsed count: %d, truncated: %s, costMs: %d, messageId: %s',
        parsed.length,
        budget.truncated ? 'true' : 'false',
        Date.now() - budget.startedAt,
        messageId,
      )

      const normalizeOutputEntry = (item) => {
        const output = {
          nickname: trimText(item?.nickname),
          user_id: trimText(item?.userId),
          time: trimText(item?.time),
          content: trimText(item?.content),
        }

        const quoteId = trimText(item?.quote)
        if (quoteId) {
          output.hasQuote = true
        }

        if (Array.isArray(item?.forward) && item.forward.length > 0) {
          output.forward = item.forward.map((child) => normalizeOutputEntry(child))
        }
        return output
      }

      const messages = parsed.map((item) => normalizeOutputEntry(item))
      await this.deps.cacheService.set(cacheKey, messages)
      logger.info(
        'reade_forward_msg cache set, count: %d, messageId: %s, maxDepth: %d',
        messages.length,
        messageId,
        maxDepth,
      )

      return JSON.stringify(messages, null, 2)
    } catch (error) {
      logger.warn('reade_forward_msg failed', error)
      return `reade_forward_msg failed: ${error?.message || String(error)}`
    }
  }
}

class SendForwardMsgTool extends StructuredTool {
  constructor(deps) {
    super({})
    this.deps = deps
    this.name = getSendToolConfig(deps.config).name
    this.description =
      '以 Bot 自身身份发送合并转发消息，支持文本和图片 URL。图片会下载并转为 base64 data URL 后发送。'
    this.schema = z.object({
      messages: z
        .union([
          z.string(),
          z.array(
            z.object({
              text: z.string().optional(),
              image: z.string().optional(),
              imageUrl: z.string().optional(),
              images: z.array(z.string()).optional(),
              imageUrls: z.array(z.string()).optional(),
            }),
          ),
        ])
        .describe('消息数组，支持 JSON 字符串。每条可包含 text 与 image/imageUrls。'),
      targetType: z.enum(['group', 'private']).optional().describe('发送目标类型。'),
      groupId: z.string().optional().describe('目标群号。'),
      userId: z.string().optional().describe('目标用户 QQ 号。'),
      targetGroupId: z.string().optional().describe('groupId 别名。'),
      targetUserId: z.string().optional().describe('userId 别名。'),
    })
  }

  async _call(input, _manager, runnable) {
    try {
      const checked = ensureNapcatSession(runnable)
      if (checked.error) return checked.error

      const { session, internal } = checked
      const target = pickTarget(session, input)
      if (target.error) return target.error

      const botId = trimText(session.bot?.selfId) || trimText(session.selfId) || 'bot'
      const imageConfig = getImageServiceConfig(this.deps.config)
      const nameCache = new Map()
      const displayName = await resolveBotDisplayName({
        internal,
        session,
        target,
        config: this.deps.config,
        botId,
        nameCache,
      })

      const items = parseToolMessages(input.messages)
      if (!items.length) return 'messages is empty.'

      const nodes = []

      for (const item of items) {
        const text = trimText(item?.text || item?.content || item?.msg)
        const imageUrls = normalizeImageUrls(item || {})
        const content = []

        if (text) {
          content.push(toTextSegment(text))
        }

        const dataUrls = await mapWithConcurrency(
          imageUrls,
          imageConfig.taskConcurrency,
          (imageUrl) => readImageAsDataUrl(this.deps.ctx, imageUrl, imageConfig.requestTimeoutMs),
        )
        for (const dataUrl of dataUrls) {
          content.push(toImageSegment(dataUrl))
        }

        if (!content.length) continue

        nodes.push({
          type: 'node',
          data: {
            user_id: botId,
            nickname: displayName,
            message_seq: 0,
            content,
          },
        })
      }

      if (!nodes.length) {
        return 'No valid message node generated. Provide text or image URLs.'
      }

      const data = await sendForwardNodes({ internal, target, nodes })
      const messageId = data?.message_id || data?.res_id || ''
      return `发送成功，节点数：${nodes.length}，目标：${target.type}:${target.id}${messageId ? `，message_id:${messageId}` : ''}`
    } catch (error) {
      logger.warn('send_forward_msg failed', error)
      return `send_forward_msg failed: ${error?.message || String(error)}`
    }
  }
}

class SendFakeMsgTool extends StructuredTool {
  constructor(deps) {
    super({})
    this.deps = deps
    this.name = getFakeToolConfig(deps.config).name
    this.description =
      '以任意 senderId/senderName 伪造消息节点发送合并转发，支持文本和图片 URL。图片会下载并转为 base64 data URL 后发送。'
    this.schema = z.object({
      messages: z
        .union([
          z.string(),
          z.array(
            z.object({
              senderId: z.string().optional(),
              userId: z.string().optional(),
              id: z.string().optional(),
              uin: z.string().optional(),
              senderName: z.string().optional(),
              nickname: z.string().optional(),
              name: z.string().optional(),
              text: z.string().optional(),
              content: z.string().optional(),
              msg: z.string().optional(),
              image: z.string().optional(),
              imageUrl: z.string().optional(),
              images: z.array(z.string()).optional(),
              imageUrls: z.array(z.string()).optional(),
            }),
          ),
        ])
        .describe('伪造消息数组，支持 JSON 字符串。'),
      targetType: z.enum(['group', 'private']).optional().describe('发送目标类型。'),
      groupId: z.string().optional().describe('目标群号。'),
      userId: z.string().optional().describe('目标用户 QQ 号。'),
      targetGroupId: z.string().optional().describe('groupId 别名。'),
      targetUserId: z.string().optional().describe('userId 别名。'),
    })
  }

  async _call(input, _manager, runnable) {
    try {
      const checked = ensureNapcatSession(runnable)
      if (checked.error) return checked.error

      const { session, internal } = checked
      const target = pickTarget(session, input)
      if (target.error) return target.error

      const items = parseToolMessages(input.messages)
      if (!items.length) return 'messages is empty.'

      const imageConfig = getImageServiceConfig(this.deps.config)
      const nameCache = new Map()
      const nodes = []

      for (const item of items) {
        const senderId =
          trimText(item?.senderId) ||
          trimText(item?.userId) ||
          trimText(item?.id) ||
          trimText(item?.uin)

        if (!senderId) continue

        const senderName = await resolveUserDisplayName({
          internal,
          target,
          userId: senderId,
          fallbackNames: [
            trimText(item?.senderName),
            trimText(item?.nickname),
            trimText(item?.name),
          ].filter(Boolean),
          nameCache,
        })

        const text = trimText(item?.text || item?.content || item?.msg)
        const imageUrls = normalizeImageUrls(item || {})
        const content = []

        if (text) {
          content.push(toTextSegment(text))
        }

        const dataUrls = await mapWithConcurrency(
          imageUrls,
          imageConfig.taskConcurrency,
          (imageUrl) => readImageAsDataUrl(this.deps.ctx, imageUrl, imageConfig.requestTimeoutMs),
        )
        for (const dataUrl of dataUrls) {
          content.push(toImageSegment(dataUrl))
        }

        if (!content.length) continue

        nodes.push({
          type: 'node',
          data: {
            user_id: senderId,
            nickname: senderName,
            message_seq: 0,
            content,
          },
        })
      }

      if (!nodes.length) {
        return 'No valid fake message generated. Every message needs senderId and text/image.'
      }

      const data = await sendForwardNodes({ internal, target, nodes })
      const messageId = data?.message_id || data?.res_id || ''
      return `发送成功，伪造节点数：${nodes.length}，目标：${target.type}:${target.id}${messageId ? `，message_id:${messageId}` : ''}`
    } catch (error) {
      logger.warn('send_fake_msg failed', error)
      return `send_fake_msg failed: ${error?.message || String(error)}`
    }
  }
}

class DescribeImageByUrlTool extends StructuredTool {
  constructor(deps) {
    super({})
    this.deps = deps
    this.name = getDescribeImageToolConfig(deps.config).name
    this.description =
      '对指定 URL 图片生成图像描述。可传入 requirement（描述要求），系统会把该要求追加到图片描述提示词末尾。'
    this.schema = z.object({
      url: z.string().optional().describe('图片 URL。'),
      imageUrl: z.string().optional().describe('url 别名。'),
      requirement: z.string().optional().describe('补充描述要求，会追加到提示词末尾。'),
      descriptionRequirement: z.string().optional().describe('requirement 别名。'),
    })
  }

  async _call(input) {
    try {
      const sourceUrl = trimText(input?.url) || trimText(input?.imageUrl)
      if (!sourceUrl) {
        return 'Missing image url. Please provide url or imageUrl.'
      }

      const requirement =
        trimText(input?.requirement) || trimText(input?.descriptionRequirement)
      const imageConfig = getImageServiceConfig(this.deps.config)
      const finalPrompt = requirement
        ? `${imageConfig.prompt}\n\n补充描述要求：${requirement}`
        : imageConfig.prompt

      const imageMeta = await readImageAsDataUrlWithMeta(
        this.deps.ctx,
        sourceUrl,
        imageConfig.requestTimeoutMs,
      )
      if (imageMeta.mime === 'image/gif') {
        return JSON.stringify({
          url: sourceUrl,
          skipped: true,
          reason: 'gif_skipped',
          description: '',
        }, null, 2)
      }

      const modelRef = await this.deps.ensureImageModelRef()
      const description = await describeImageWithModel({
        modelRef,
        imagePrompt: finalPrompt,
        dataUrl: imageMeta.dataUrl,
      })

      return JSON.stringify({
        url: sourceUrl,
        requirement: requirement || '',
        description: trimText(description),
      }, null, 2)
    } catch (error) {
      logger.warn('describe_image_by_url failed', error)
      return `describe_image_by_url failed: ${error?.message || String(error)}`
    }
  }
}

const Config = Schema.intersect([
  Schema.object({
    readTool: Schema.object({
      enable: Schema.boolean().default(true).description('是否启用（读取并解析合并转发消息，只提取文本和图片；图片可调用多模态模型生成描述）。'),
      name: Schema.string().default('reade_forward_msg').description('读取合并转发工具名（提供给 ChatLuna 调用）。'),
      maxParseDepth: Schema.number().min(1).max(8).default(3).description('嵌套合并转发解析层数（含读取的记录本身）。'),
      describeImageInRead: Schema.boolean().default(true).description('读取合并转发时是否为图片生成描述。'),
    }).description('读取工具'),
    sendTool: Schema.object({
      enable: Schema.boolean().default(true).description('是否启用（以 Bot 自身身份发送合并转发，支持文本与图片 URL，自动转 Base64 上传）。'),
      name: Schema.string().default('send_forward_msg').description('Bot 身份发送合并转发工具名（提供给 ChatLuna 调用）。'),
      botDisplayName: Schema.string().default('').description('发送时显示昵称，留空则自动推断。'),
    }).description('发送工具'),
    fakeTool: Schema.object({
      enable: Schema.boolean().default(true).description('是否启用（以任意用户 ID/昵称伪造节点发送合并转发，支持文本与图片 URL，自动转 Base64 上传）。'),
      name: Schema.string().default('send_fake_msg').description('伪造身份发送合并转发工具名（提供给 ChatLuna 调用）。'),
    }).description('伪造工具'),
    describeImageTool: Schema.object({
      enable: Schema.boolean().default(true).description('是否启用（按 URL 获取图像描述，可附加补充描述要求）。'),
      name: Schema.string().default('describe_image_by_url').description('按 URL 获取图像描述工具名（提供给 ChatLuna 调用）。'),
    }).description('图像描述工具'),
    imageService: Schema.object({
      model: Schema.dynamic('model').default('无').description('用于图片描述的多模态模型。'),
      prompt: Schema.string().role('textarea').default(DEFAULT_IMAGE_PROMPT).description('图片描述提示词。'),
      taskConcurrency: Schema.number().min(1).max(100).default(20).description('图片任务并发数（读取描述、发送下载共用）。'),
      requestTimeoutSeconds: Schema.number().min(1).max(120).default(DEFAULT_IMAGE_REQUEST_TIMEOUT_SECONDS).description('下载图片请求超时时间（秒）。'),
    }).description('图片描述服务'),
    cacheService: Schema.object({
      enable: Schema.boolean().default(true).description('是否启用读取结果缓存。启用后同 message_id 会优先返回缓存，减少重复请求。'),
      ttlSeconds: Schema.number().min(60).max(30 * 24 * 60 * 60).default(DEFAULT_CACHE_TTL_SECONDS).description('缓存有效期（秒），默认 1 天。'),
      storagePath: Schema.path({ filters: ['directory'] }).default(DEFAULT_CACHE_STORAGE_PATH).description('缓存目录（仅本地存储使用）。'),
      cleanupIntervalSeconds: Schema.number().min(60).max(24 * 60 * 60).default(DEFAULT_CACHE_CLEANUP_INTERVAL_SECONDS).description('后台清理过期缓存的最小间隔（秒）。'),
    }).description('缓存服务'),
  }),
])

const usage = `
## chatluna-forward-msg

使用前请确认已开启以下选项，否则本插件可能无法获取到合并转发所需的上下文信息：

- 在 chatluna 插件的“对话行为选项”中启用：attachForwardMsgIdToContext。
- 在 chatluna-character 插件的“对话设置”中启用：enableMessageId。
`

module.exports = {
  name,
  inject,
  usage,
  Config,
  apply,
}
