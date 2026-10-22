// mock_upstream.js — local fake OpenAI-compatible upstream for testing gateway.js
// Serves /v1/chat/completions (stream + non-stream) with canned data.
const http = require('http');

const server = http.createServer((req, res) => {
  let body = '';
  req.on('data', c => body += c);
  req.on('end', () => {
    const parsed = JSON.parse(body || '{}');
    console.log('[mock received] model=', parsed.model, 'messages=', JSON.stringify(parsed.messages), 'tools?', Array.isArray(parsed.tools));
    const wantTools = Array.isArray(parsed.tools) && parsed.tools.length;
    const isStream = !!parsed.stream;

    if (!isStream) {
      // non-stream: single JSON object
      const msg = wantTools
        ? { role: 'assistant', content: null, tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Beijing"}' } }] }
        : { role: 'assistant', content: 'Hello from mock upstream' };
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({
        id: 'chatcmpl-mock', object: 'chat.completion', created: 0, model: parsed.model,
        choices: [{ index: 0, message: msg, finish_reason: wantTools ? 'tool_calls' : 'stop' }],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      }));
    }

    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    const sse = o => res.write(`data: ${JSON.stringify(o)}\n\n`);

    // streaming
    // 先发一段 reasoning_content（模拟智谱/DeepSeek 推理流），验证网关转成 thinking 块
    sse({ id: 'cmpl-r', object: 'chat.completion.chunk', choices: [{ index: 0, delta: { role: 'assistant', reasoning_content: '思考中：先分析问题……' }, finish_reason: null }] });
    sse({ id: 'cmpl-r', object: 'chat.completion.chunk', choices: [{ index: 0, delta: { reasoning_content: '得出结论。' }, finish_reason: null }] });

    if (wantTools) {
      sse({ id: 'cmpl-1', object: 'chat.completion.chunk', choices: [{ index: 0, delta: { role: 'assistant', content: null }, finish_reason: null }] });
      sse({ id: 'cmpl-1', object: 'chat.completion.chunk', choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'get_weather', arguments: '' } }] }, finish_reason: null }] });
      sse({ id: 'cmpl-1', object: 'chat.completion.chunk', choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '{"city":' } }] }, finish_reason: null }] });
      sse({ id: 'cmpl-1', object: 'chat.completion.chunk', choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '"Beijing"}' } }] }, finish_reason: null }] });
      sse({ id: 'cmpl-1', object: 'chat.completion.chunk', choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] });
    } else {
      sse({ id: 'cmpl-2', object: 'chat.completion.chunk', choices: [{ index: 0, delta: { role: 'assistant', content: '你好' }, finish_reason: null }] });
      sse({ id: 'cmpl-2', object: 'chat.completion.chunk', choices: [{ index: 0, delta: { content: '，世界' }, finish_reason: null }] });
      sse({ id: 'cmpl-2', object: 'chat.completion.chunk', choices: [{ index: 0, delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 10, completion_tokens: 4, total_tokens: 14 } });
    }
    res.write('data: [DONE]\n\n');
    res.end();
  });
});

server.listen(8092, '127.0.0.1', () => console.log('mock upstream on 8092'));
