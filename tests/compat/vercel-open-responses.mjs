import { createOpenResponses } from '@ai-sdk/open-responses';
import { generateText, streamText } from 'ai';

const baseUrl = process.env.FUGUE_COMPAT_BASE_URL ?? 'http://127.0.0.1:18765';
const provider = createOpenResponses({
  name: 'fugue',
  url: `${baseUrl.replace(/\/$/, '')}/v1/responses`,
  headers: { Authorization: 'Bearer fugue-compatibility-key' },
});
const model = provider('fugue-candidate');

const generated = await generateText({ model, prompt: 'hello' });
if (generated.text !== 'Fugue compatibility response') {
  throw new Error(`unexpected synchronous response: ${generated.text}`);
}

const streamed = streamText({ model, prompt: 'hello' });
let text = '';
for await (const delta of streamed.textStream) {
  text += delta;
}
if (text !== 'Fugue compatibility response') {
  throw new Error(`unexpected streaming response: ${text}`);
}
