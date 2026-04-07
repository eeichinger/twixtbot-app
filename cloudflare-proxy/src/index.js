export default {
  async fetch(request) {
    // Only allow GET
    if (request.method !== 'GET') {
      return new Response('Method not allowed', { status: 405 });
    }

    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET',
        },
      });
    }

    // Target URL is passed as the `url` query parameter
    const incoming = new URL(request.url);
    const target = incoming.searchParams.get('url');
    if (!target) {
      return new Response('Missing ?url= parameter', { status: 400 });
    }

    // Parse and validate the target URL
    let targetUrl;
    try {
      targetUrl = new URL(target);
    } catch {
      return new Response('Invalid URL', { status: 400 });
    }

    // Only proxy littlegolem.net
    if (!targetUrl.hostname.endsWith('littlegolem.net')) {
      return new Response('Only littlegolem.net URLs are allowed', { status: 403 });
    }

    // Fetch from LG with a 10s timeout
    let lgResponse;
    try {
      const abort = new AbortController();
      const timer = setTimeout(() => abort.abort(), 10_000);
      lgResponse = await fetch(targetUrl.toString(), {
        signal: abort.signal,
        headers: {
          'User-Agent': 'Mozilla/5.0 (compatible; twixtbot-app/1.0)',
        },
      });
      clearTimeout(timer);
    } catch (err) {
      const msg = err.name === 'AbortError' ? 'Timeout fetching from littlegolem.net' : 'Failed to fetch from littlegolem.net: ' + err.message;
      console.error(msg, targetUrl.toString());
      return new Response(msg, { status: 502 });
    }
    console.log(`LG ${lgResponse.status} ${targetUrl.toString()}`);

    // Forward the response body with CORS headers added
    const headers = new Headers(lgResponse.headers);
    headers.set('Access-Control-Allow-Origin', '*');
    headers.set('Access-Control-Allow-Methods', 'GET');

    return new Response(lgResponse.body, {
      status: lgResponse.status,
      headers,
    });
  },
};
