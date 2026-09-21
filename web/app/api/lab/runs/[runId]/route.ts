import { proxyLab } from '../../_proxy';
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const GET = (request: Request, { params }: { params: { runId: string } }) => proxyLab(request, `/lab/runs/${params.runId}`);
