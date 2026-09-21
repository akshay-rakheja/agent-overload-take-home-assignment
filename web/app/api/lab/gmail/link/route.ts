import { proxyLab } from '../../_proxy';
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const POST = (request: Request) => proxyLab(request, '/lab/gmail/link', 'POST');
