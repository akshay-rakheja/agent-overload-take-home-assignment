import { proxyLab } from '../../_proxy';
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const GET = (request: Request) => proxyLab(request, '/lab/gmail/status');
