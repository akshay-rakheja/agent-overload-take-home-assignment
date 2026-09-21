// A real Response whose headers arrive and whose body fails after its first chunk.
export function interruptedJson(status = 200) {
  let sentChunk = false;
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (!sentChunk) {
        sentChunk = true;
        controller.enqueue(new TextEncoder().encode('{"status":'));
      } else {
        controller.error(new TypeError('Connection lost: private transport detail'));
      }
    },
  });
  return new Response(body, { status, headers: { 'Content-Type': 'application/json' } });
}
