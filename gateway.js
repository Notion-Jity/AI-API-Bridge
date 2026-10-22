#!/usr/bin/env node
/**
 * gateway.js — Anthropic Messages API -> OpenAI-compatible API proxy.
 *
 * Runs on the internet-capable laptop. Lets Claude Code (on an offline host)
 * talk to any OpenAI-compatible upstream (DeepSeek / Zhipu GLM / Kimi / Qwen /
 * OpenAI / ...) using your own API key.
 *
 * Usage:
 *   1. Edit the CONFIG section below (upstream base URL, your key, model name).
 *   2. Run:  node gateway.js
 *   3. On the Claude Code host set:
 *        ANTHROPIC_BASE_URL = http://<this-laptop-ip>:8090
 *        ANTHROPIC_AUTH_TOKEN = anything (ignored unless requireAuth=true)
 *        ANTHROPIC_MODEL = any name present in modelMap (or leave it; the map falls back)
 *   4. Open firewall port 8090 for inbound TCP.
 *
 * Zero dependencies — Node built-in modules only.
 */

const http = require('http');
const https = require('https');
const { URL } = require('url');
const { StringDecoder } = require('string_decoder');

/* ============================== CONFIG ============================== */
const CONFIG = {
  /* --- 上游（OpenAI 兼容）--- */
  // DeepSeek  : https://api.deepseek.com                  model: deepseek-chat
  // 智谱 GLM  : https://open.bigmodel.cn/api/paas/v4      model: glm-4.6 (看你的套餐)
  // Kimi      : https://api.moonshot.cn/v1                model: moonshot-v1-8k
  // 通义      : https://dashscope.aliyuncs.com/compatible-mode/v1  model: qwen-plus
  // OpenAI    : https://api.openai.com/v1                 model: gpt-4o
  upstreamBase: 'http://127.0.0.1:8080/v1',   // 使用的api url
  upstreamKey:  'pwd',							//上游API-KEY
  upstreamModel:'gpt-6.1-sol-high',              // 模型名

  /* --- 监听 --- */
  listenPort: 8090,
  listenHost: '0.0.0.0',

  /* 上游请求超时（毫秒）。0 = 不超时。外网抖动时避免请求永久挂起 */
  upstreamTimeout: 120000,

  /* --- 认证（可选）--- */
  // 开启后，客户端必须带 Authorization: Bearer <authToken> 才会被放行。
  // Claude Code 的 ANTHROPIC_AUTH_TOKEN 填成和这里一样的值即可。
  requireAuth: false,
  authToken: '',

  /* --- 模型映射：Claude Code 发来的模型名 -> 上游模型名 --- */
  // 映射值写 '@upstream' 表示「用上面 upstreamModel 指定的模型」，
  // 这样换上游模型时只需改 upstreamModel 一处。不认识的模型名落到 '*'。
  modelMap: {
    'glm-5.3':            '@upstream',
    'claude-sonnet-4-5':  '@upstream',
    'claude-opus-4-5':    '@upstream',
    'claude-haiku-4-5':   '@upstream',
    '*':                  '@upstream',
  },

  /* 调试：打印收到的请求体（不要在生产开，key/内容会刷屏） */
  debug: false,
};
/* ==================================================================== */

function log(...a) { console.log(new Date().toISOString(), ...a); }

// Startup sanity checks — fail early with a clear message instead of crashing mid-request.
function assertConfig() {
  const problems = [];
  if (!CONFIG.upstreamBase || !/^https?:\/\//.test(CONFIG.upstreamBase)) {
    problems.push('upstreamBase 不是合法的 http(s) URL');
  }
  if (!CONFIG.upstreamKey || CONFIG.upstreamKey.startsWith('sk-在此填入')) {
    problems.push('upstreamKey 还是占位符，请填入你自己的 API key');
  } else if (!/^[\x20-\x7E]+$/.test(CONFIG.upstreamKey)) {
    problems.push('upstreamKey 含有非 ASCII 字符（可能误填了中文），请检查');
  }
  if (!CONFIG.upstreamModel) problems.push('upstreamModel 不能为空');
  if (problems.length) {
    console.error('gateway 配置有误，无法启动：');
    problems.forEach(p => console.error('  - ' + p));
    process.exit(1);
  }
}
assertConfig();

function jsonErr(status, msg) {
  return JSON.stringify({ type: 'error', error: { type: 'api_error', message: msg } });
}

function resolveUpstreamModel(incoming) {
  const m = Object.prototype.hasOwnProperty.call(CONFIG.modelMap, incoming)
    ? CONFIG.modelMap[incoming] : CONFIG.modelMap['*'];
  return (m === '@upstream' || m === undefined || m === null) ? CONFIG.upstreamModel : m;
}

// Resolve the full /chat/completions endpoint from the user-provided base.
// Handles: bare host (adds /v1), already-versioned base (/v1, /v4, ...), and
// a base that already IS the full chat/completions endpoint.
function chatEndpoint() {
  let base = CONFIG.upstreamBase.replace(/\/+$/, '');
  if (/\/chat\/completions$/.test(base)) return base;
  if (/\/v\d+$/.test(base)) return base + '/chat/completions';
  return base + '/v1/chat/completions';
}

/* ---------------- Anthropic request -> OpenAI request ---------------- */

function convText(content) {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return String(content || '');
  return content
    .map(b => {
      if (!b) return '';
      if (b.type === 'text') return b.text || '';
      if (b.type === 'tool_reference') {
        return `Discovered tool: ${b.tool_name || 'unknown'}. Its definition is available in the tools list.`;
      }
      if (b.type === 'image') return '[image omitted]';
      if (b.type === 'tool_result') {
        const inner = convText(b.content);
        return b.is_error ? `[tool_error]\n${inner}` : `[tool_result]\n${inner}`;
      }
      return '';
    })
    .join('\n');
}

function convMessage(m) {
  const role = m.role;
  const content = m.content;

  // user 消息里的 tool_result 块要转成 OpenAI 的 role:"tool"
  if (Array.isArray(content) && content.some(b => b && b.type === 'tool_result')) {
    const out = [];
    let textParts = [];
    for (const b of content) {
      if (b.type === 'text') textParts.push(b.text || '');
      else if (b.type === 'image') textParts.push('[image omitted]');
      else if (b.type === 'tool_result') {
        if (textParts.length) { out.push({ role: 'user', content: textParts.join('\n') }); textParts = []; }
        const toolContent = convText(b.content) || '[empty tool result]';
        out.push({ role: 'tool', tool_call_id: b.tool_use_id || '', content: b.is_error ? `Error: ${toolContent}` : toolContent });
      }
    }
    if (textParts.length) out.push({ role: 'user', content: textParts.join('\n') });
    return out;
  }

  // assistant 消息里的 tool_use 块 -> OpenAI tool_calls
  if (role === 'assistant' && Array.isArray(content) && content.some(b => b && b.type === 'tool_use')) {
    const text = content.filter(b => b.type === 'text').map(b => b.text || '').join('\n');
    const toolCalls = content.filter(b => b.type === 'tool_use').map(b => ({
      id: b.id || `call_${Math.random().toString(36).slice(2, 10)}`,
      type: 'function',
      function: {
        name: b.name,
        arguments: typeof b.input === 'string' ? b.input : JSON.stringify(b.input || {}),
      },
    }));
    return [{ role: 'assistant', content: text || null, tool_calls: toolCalls }];
  }

  // 普通消息
  if (typeof content === 'string') return [{ role, content }];
  if (Array.isArray(content)) {
    const text = content.filter(b => b.type === 'text').map(b => b.text || '').join('\n');
    return [{ role, content: text || '' }];
  }
  return [{ role, content: String(content || '') }];
}

function toOpenAIRequest(anBody) {
  const messages = [];
  if (anBody.system) {
    messages.push({ role: 'system', content: typeof anBody.system === 'string' ? anBody.system : convText(anBody.system) });
  }
  for (const m of (anBody.messages || [])) {
    messages.push(...convMessage(m));
  }
  // Preserve every tool response so tool calls and results stay paired.
  const cleaned = messages.filter(m => {
    if (m.role === 'tool') return true;
    return !(typeof m.content === 'string' && m.content === '');
  });

  const req = {
    model: resolveUpstreamModel(anBody.model),
    messages: cleaned,
    stream: !!anBody.stream,
  };
  if (anBody.max_tokens) req.max_tokens = anBody.max_tokens;
  if (anBody.temperature !== undefined) req.temperature = anBody.temperature;
  if (anBody.top_p !== undefined) req.top_p = anBody.top_p;

  // tools
  if (Array.isArray(anBody.tools) && anBody.tools.length) {
    req.tools = anBody.tools.map(t => ({
      type: 'function',
      function: {
        name: t.name,
        description: t.description || '',
        parameters: t.input_schema || { type: 'object', properties: {} },
      },
    }));
    if (anBody.tool_choice && typeof anBody.tool_choice === 'object' && anBody.tool_choice.type === 'tool') {
      req.tool_choice = { type: 'function', function: { name: anBody.tool_choice.name } };
    } else if (req.tools.length) {
      req.tool_choice = 'auto';
    }
  }
  return req;
}

/* ---------------- OpenAI response -> Anthropic response ---------------- */

function validateUpstreamResponse(openai) {
  if (!openai || typeof openai !== 'object') throw new Error('invalid upstream response');
  if (openai.error) {
    const detail = typeof openai.error === 'string' ? openai.error : JSON.stringify(openai.error);
    throw new Error(`upstream error: ${detail.slice(0, 1000)}`);
  }
}

function toAnthropicMessage(openai, requestedModel) {
  validateUpstreamResponse(openai);
  if (!Array.isArray(openai.choices) || !openai.choices[0]?.message) {
    throw new Error('upstream response has no assistant message');
  }
  const msg = openai.choices[0].message;
  const content = [];
  // reasoning_content (Zhipu/DeepSeek style) -> thinking block, placed before text
  if (msg.reasoning_content) {
    content.push({ type: 'thinking', thinking: msg.reasoning_content });
  }
  if (msg.content) content.push({ type: 'text', text: msg.content });
  if (Array.isArray(msg.tool_calls)) {
    for (const tc of msg.tool_calls) {
      if (!tc.id || !tc.function?.name) throw new Error('incomplete upstream tool call');
      let input;
      try { input = JSON.parse(tc.function.arguments || '{}'); }
      catch (e) { throw new Error(`invalid arguments for upstream tool ${tc.function.name}`); }
      if (!input || typeof input !== 'object' || Array.isArray(input)) {
        throw new Error(`upstream tool ${tc.function.name} arguments must be an object`);
      }
      content.push({ type: 'tool_use', id: tc.id, name: tc.function.name, input });
    }
  }
  if (!content.some(b => b.type === 'text' || b.type === 'tool_use')) {
    throw new Error('upstream returned no text or tool calls');
  }
  const finish = openai.choices[0].finish_reason;
  if (!['stop', 'tool_calls', 'length'].includes(finish)) {
    throw new Error(`unsupported upstream finish_reason: ${finish}`);
  }
  const stopReason = finish === 'tool_calls' ? 'tool_use' : finish === 'length' ? 'max_tokens' : 'end_turn';
  const usage = openai.usage || {};
  return {
    id: `msg_${(openai.id || 'x').replace(/^chatcmpl-/, '')}`,
    type: 'message',
    role: 'assistant',
    model: requestedModel,
    content,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: usage.prompt_tokens || 0,
      output_tokens: usage.completion_tokens || 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    },
  };
}

/* ---------------- 流式转换：OpenAI SSE -> Anthropic SSE ---------------- */

function anEvent(event, data) {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function makeStreamer(requestedModel) {
  const state = {
    started: false, blockStarted: false, blockType: null, blockIndex: 0,
    lastUsage: null, finish: null, hasOutput: false, tools: new Map(),
  };

  function sendMessageStart() {
    state.started = true;
    return anEvent('message_start', {
      type: 'message_start',
      message: {
        id: `msg_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`,
        type: 'message', role: 'assistant', model: requestedModel,
        content: [], usage: { input_tokens: 0, output_tokens: 0, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 },
      },
    });
  }

  function openTextBlock() {
    if (!state.started) sendMessageStart();
    state.blockStarted = true; state.blockType = 'text';
    return anEvent('content_block_start', {
      type: 'content_block_start', index: state.blockIndex,
      content_block: { type: 'text', text: '' },
    });
  }

  function openThinkingBlock() {
    if (!state.started) sendMessageStart();
    state.blockStarted = true; state.blockType = 'thinking';
    return anEvent('content_block_start', {
      type: 'content_block_start', index: state.blockIndex,
      content_block: { type: 'thinking', thinking: '' },
    });
  }

  function openToolBlock(id, name) {
    if (!state.started) sendMessageStart();
    state.blockStarted = true; state.blockType = 'tool_use';
    return anEvent('content_block_start', {
      type: 'content_block_start', index: state.blockIndex,
      content_block: { type: 'tool_use', id, name, input: {} },
    });
  }

  function handleOpenAIChunk(chunk) {
    validateUpstreamResponse(chunk);
    let out = '';
    if (chunk.usage) state.lastUsage = chunk.usage;
    if (!Array.isArray(chunk.choices)) throw new Error('upstream chunk has no choices');
    if (!chunk.choices.length) return out;
    if (state.finish) throw new Error('upstream sent a choice after finish_reason');
    const choice = chunk.choices[0];
    const delta = choice.delta || {};
    const finish = choice.finish_reason || null;

    if (!state.started) out += sendMessageStart();

    // 推理内容（智谱/DeepSeek 的 reasoning_content）→ thinking 块。
    // 关键：思考期间持续向客户端转发事件，避免长时间零输出导致
    // 客户端（如 Claude Code）首字节超时后断开重试。
    if (delta.reasoning_content) {
      if (!state.blockStarted || state.blockType !== 'thinking') {
        if (state.blockStarted) out += closeBlock();
        out += openThinkingBlock();
      }
      out += anEvent('content_block_delta', {
        type: 'content_block_delta', index: state.blockIndex,
        delta: { type: 'thinking_delta', thinking: delta.reasoning_content },
      });
    }

    // 文本
    if (delta.content) {
      if (typeof delta.content !== 'string') throw new Error('invalid upstream text delta');
      state.hasOutput = true;
      if (!state.blockStarted || state.blockType !== 'text') {
        if (state.blockStarted) out += closeBlock();
        out += openTextBlock();
      }
      out += anEvent('content_block_delta', {
        type: 'content_block_delta', index: state.blockIndex,
        delta: { type: 'text_delta', text: delta.content },
      });
    }

    // Buffer by upstream index: parallel calls and split names may interleave.
    if (Array.isArray(delta.tool_calls)) {
      for (const tc of delta.tool_calls) {
        if (!Number.isInteger(tc.index) || tc.index < 0) throw new Error('upstream tool call has no valid index');
        let tool = state.tools.get(tc.index);
        if (!tool) {
          tool = { id: '', name: '', arguments: '' };
          state.tools.set(tc.index, tool);
        }
        if (tc.id) tool.id = tc.id;
        if (tc.function?.name) tool.name += tc.function.name;
        if (tc.function?.arguments) tool.arguments += tc.function.arguments;
      }
    }

    if (finish) {
      if (!['stop', 'tool_calls', 'length'].includes(finish)) {
        throw new Error(`unsupported upstream finish_reason: ${finish}`);
      }
      state.finish = finish;
      if (state.blockStarted) out += closeBlock();
    }
    return out;
  }

  function closeBlock() {
    if (!state.blockStarted) return '';
    const idx = state.blockIndex;
    state.blockStarted = false;
    state.blockType = null;
    state.blockIndex += 1;
    return anEvent('content_block_stop', { type: 'content_block_stop', index: idx });
  }

  function finalize() {
    if (!state.finish) throw new Error('upstream stream ended without finish_reason');
    if (state.finish === 'tool_calls' && !state.tools.size) {
      throw new Error('upstream finished with tool_calls but supplied no calls');
    }
    for (const tool of state.tools.values()) {
      if (!tool.id || !tool.name) throw new Error('incomplete upstream tool call');
      let input;
      try { input = JSON.parse(tool.arguments || '{}'); }
      catch (e) { throw new Error(`invalid arguments for upstream tool ${tool.name}`); }
      if (!input || typeof input !== 'object' || Array.isArray(input)) {
        throw new Error(`upstream tool ${tool.name} arguments must be an object`);
      }
    }
    if (!state.hasOutput && !state.tools.size) throw new Error('upstream returned no text or tool calls');
    let out = '';
    if (state.blockStarted) out += closeBlock();
    if (!state.started) out += sendMessageStart();
    for (const tool of state.tools.values()) {
      out += openToolBlock(tool.id, tool.name);
      out += anEvent('content_block_delta', {
        type: 'content_block_delta', index: state.blockIndex,
        delta: { type: 'input_json_delta', partial_json: tool.arguments || '{}' },
      });
      out += closeBlock();
    }
    const stopReason = state.finish === 'tool_calls' ? 'tool_use' : state.finish === 'length' ? 'max_tokens' : 'end_turn';
    out += anEvent('message_delta', {
      type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null },
      usage: { output_tokens: state.lastUsage?.completion_tokens || 0 },
    });
    out += anEvent('message_stop', { type: 'message_stop' });
    return out;
  }

  return { handleOpenAIChunk, finalize };
}

/* ---------------- 上游调用 ---------------- */

function callUpstream(req, anBody, onData, onDone, onErr) {
  const oaiReq = toOpenAIRequest(anBody);
  const u = new URL(chatEndpoint());
  const body = JSON.stringify(oaiReq);
  const requestId = Math.random().toString(36).slice(2, 10);
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${CONFIG.upstreamKey}`,
    'Content-Length': Buffer.byteLength(body),
  };
  let settled = false;
  let upstreamResponse;
  let hreq;
  function fail(err) {
    if (settled) return;
    settled = true;
    log(`[upstream ${requestId} error]`, err.message);
    onErr(err);
    upstreamResponse?.destroy();
    hreq?.destroy();
  }
  function done() {
    if (settled) return;
    settled = true;
    onDone();
  }

  if (CONFIG.debug) log(`[upstream ${requestId} req]`, body.slice(0, 800));
  const isHttps = u.protocol === 'https:';
  const mod = isHttps ? https : http;
  hreq = mod.request(
    {
      hostname: u.hostname,
      port: u.port || (isHttps ? 443 : 80),
      path: u.pathname + u.search,
      method: 'POST',
      headers,
    },
    res => {
      upstreamResponse = res;
      const status = res.statusCode || 0;
      const contentType = (res.headers['content-type'] || '').toLowerCase();
      const isSSE = contentType.split(';')[0].trim() === 'text/event-stream';
      log(`[upstream ${requestId} response]`, status, contentType || '(no content-type)');
      res.on('error', fail);
      res.on('aborted', () => fail(new Error('upstream response aborted')));

      if (status < 200 || status >= 300 || !anBody.stream || !isSSE) {
        const chunks = [];
        res.on('data', c => { if (!settled) chunks.push(c); });
        res.on('end', () => {
          if (settled) return;
          const raw = Buffer.concat(chunks).toString('utf8');
          try {
            if (status < 200 || status >= 300) throw new Error(`upstream ${status}: ${raw.slice(0, 1000)}`);
            let parsed;
            try { parsed = JSON.parse(raw); }
            catch (e) { throw new Error(`invalid upstream JSON (${contentType}): ${raw.slice(0, 500)}`); }
            validateUpstreamResponse(parsed);
            if (anBody.stream) {
              throw new Error(`expected upstream text/event-stream, received ${contentType || 'no content-type'}`);
            }
            onData(JSON.stringify(toAnthropicMessage(parsed, anBody.model)));
            done();
          } catch (e) { fail(e); }
        });
        return;
      }

      const decoder = new StringDecoder('utf8');
      const streamer = makeStreamer(anBody.model);
      let buf = '';
      let eventData = [];
      let eventType = '';
      let sawDone = false;
      function dispatch() {
        if (!eventData.length) { eventType = ''; return; }
        const data = eventData.join('\n');
        const type = eventType;
        eventData = [];
        eventType = '';
        if (type === 'error') throw new Error(`upstream SSE error: ${data.slice(0, 1000)}`);
        if (data.trim() === '[DONE]') { sawDone = true; return; }
        if (sawDone) throw new Error('upstream sent data after [DONE]');
        let chunk;
        try { chunk = JSON.parse(data); }
        catch (e) { throw new Error(`invalid upstream SSE JSON: ${data.slice(0, 500)}`); }
        const output = streamer.handleOpenAIChunk(chunk);
        if (output) onData(output);
      }
      function consume(text) {
        buf += text;
        let idx;
        while ((idx = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, idx).replace(/\r$/, '');
          buf = buf.slice(idx + 1);
          if (line === '') dispatch();
          else if (line.startsWith('data:')) eventData.push(line.slice(5).replace(/^ /, ''));
          else if (line.startsWith('event:')) eventType = line.slice(6).trim();
        }
      }
      res.on('data', c => {
        if (settled) return;
        try { consume(decoder.write(c)); }
        catch (e) { fail(e); }
      });
      res.on('end', () => {
        if (settled) return;
        try {
          consume(decoder.end());
          if (buf || eventData.length) throw new Error('upstream ended with an incomplete SSE event');
          onData(streamer.finalize());
          done();
        } catch (e) { fail(e); }
      });
    }
  );
  hreq.on('error', fail);
  if (CONFIG.upstreamTimeout > 0) {
    hreq.setTimeout(CONFIG.upstreamTimeout, () => {
      hreq.destroy(new Error(`upstream request timed out after ${CONFIG.upstreamTimeout}ms`));
    });
  }
  hreq.end(body);
  return hreq;
}

/* ---------------- HTTP server ---------------- */

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  const path = url.pathname;

  if (CONFIG.requireAuth) {
    const auth = req.headers['authorization'] || req.headers['x-api-key'] || '';
    const token = auth.replace(/^Bearer\s+/i, '').trim();
    if (token !== CONFIG.authToken) {
      res.writeHead(401, { 'Content-Type': 'application/json' });
      return res.end(jsonErr(401, 'unauthorized'));
    }
  }

  // GET /v1/models
  if (req.method === 'GET' && path === '/v1/models') {
    const models = [];
    const seen = new Set();
    const pushModel = m => { if (!seen.has(m)) { seen.add(m); models.push(m); } };
    for (const incoming of Object.keys(CONFIG.modelMap)) {
      if (incoming === '*') continue;
      pushModel(resolveUpstreamModel(incoming));
    }
    pushModel(resolveUpstreamModel('*'));
    pushModel(CONFIG.upstreamModel);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({
      data: models.map(m => ({ id: m, object: 'model', created: 1626777600, owned_by: 'gateway' })),
    }));
  }

  // POST /v1/messages/count_tokens —— 粗略估算
  if (req.method === 'POST' && path === '/v1/messages/count_tokens') {
    let body = '';
    req.on('data', c => body += c);
    req.on('end', () => {
      try {
        const b = JSON.parse(body);
        const text = JSON.stringify(b).length;
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ input_tokens: Math.ceil(text / 3.5) }));
      } catch (e) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ input_tokens: 0 }));
      }
    });
    return;
  }

  // POST /v1/messages
  if (req.method === 'POST' && path === '/v1/messages') {
    let body = '';
    req.on('data', c => body += c);
    req.on('end', () => {
      let anBody;
      try { anBody = JSON.parse(body); }
      catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        return res.end(jsonErr(400, 'invalid json body'));
      }

      const isStream = !!anBody.stream;

      if (!isStream) {
        callUpstream(req, anBody,
          data => {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(data);
          },
          () => {},
          err => {
            res.writeHead(502, { 'Content-Type': 'application/json' });
            res.end(jsonErr(502, err.message));
          }
        );
        return;
      }

      // 流式
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      });
      callUpstream(req, anBody,
        data => { res.write(data); },
        () => res.end(),
        err => {
          res.write(anEvent('error', { type: 'error', error: { type: 'api_error', message: err.message } }));
          res.end();
        }
      );
      req.on('close', () => { /* client may abort; upstream request handled internally */ });
    });
    return;
  }

  // 其余
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(jsonErr(404, `not found: ${path}`));
});

server.listen(CONFIG.listenPort, CONFIG.listenHost, () => {
  log(`gateway listening on http://${CONFIG.listenHost}:${CONFIG.listenPort}`);
  log(`upstream endpoint: ${chatEndpoint()}  model: ${CONFIG.upstreamModel}`);
  log(`model map: ${JSON.stringify(CONFIG.modelMap)}`);
});
