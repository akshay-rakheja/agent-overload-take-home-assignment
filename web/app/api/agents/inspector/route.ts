import fs from 'fs';
import path from 'path';

export const runtime = 'nodejs';

function resolveServerBase(system?: string | null): string {
  if (system === 'baseline') {
    return process.env.BASELINE_SERVER_URL || 'http://localhost:8001';
  }
  if (system === 'enhanced' || system === 'enhanced_deterministic' || system === 'enhanced_jev' || system === 'jev') {
    return process.env.ENHANCED_SERVER_URL || 'http://localhost:8002';
  }
  return process.env.PY_SERVER_URL || 'http://localhost:8001';
}

function getBaselineDataDir(): string {
  const possibleDirs = [
    path.resolve(process.cwd(), '../../openpoke-evaluation-baseline/server/data'),
    path.resolve(process.cwd(), '../openpoke-evaluation-baseline/server/data'),
    path.resolve(process.cwd(), 'openpoke-evaluation-baseline/server/data'),
    '/Users/akshayrakheja/Documents/general-magic-take-home/openpoke-evaluation-baseline/server/data',
  ];
  return possibleDirs.find((d) => fs.existsSync(d)) || possibleDirs[0];
}

function readHistoricalBaselineState(): any {
  try {
    const baseDir = getBaselineDataDir();
    const rosterPath = path.join(baseDir, 'execution_agents', 'roster.json');
    let rawRoster: any[] = [];
    if (fs.existsSync(rosterPath)) {
      try {
        const parsed = JSON.parse(fs.readFileSync(rosterPath, 'utf8'));
        rawRoster = Array.isArray(parsed) ? parsed : Array.isArray(parsed?.agents) ? parsed.agents : [];
      } catch {}
    }

    const roster = rawRoster.map((item: any, idx: number) => {
      const name = typeof item === 'string' ? item : (item.name || 'agent');
      let useCount = 1;
      const slug = name
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]/g, '-')
        .replace(/-+/g, '-')
        .replace(/^-|-$/g, '') || 'agent';
      const agentLogPath = path.join(baseDir, 'execution_agents', `${slug}.log`);
      if (fs.existsSync(agentLogPath)) {
        try {
          const content = fs.readFileSync(agentLogPath, 'utf8');
          const count = (content.match(/<agent_request/g) || []).length;
          if (count > 0) useCount = count;
        } catch {}
      }
      return {
        agent_id: `baseline-${name}`,
        name,
        purpose: `Historical baseline execution agent: ${name}`,
        status: 'hot',
        use_count: useCount,
        created_at: null,
        last_used_at: null,
      };
    });

    const candidates = roster.map((r, idx) => ({
      rank: idx + 1,
      agent_id: r.agent_id,
      name: r.name,
      purpose: r.purpose,
      status: r.status,
      score: null,
      reasons: ['Historical full-roster exposure'],
    }));

    let latestTurn: any = null;
    const convPath = path.join(baseDir, 'conversation', 'poke_conversation.log');
    if (fs.existsSync(convPath)) {
      const content = fs.readFileSync(convPath, 'utf8');
      const lines = content.split('\n').filter(Boolean);
      let lastUserMsg = '';
      let lastAgentMsg = '';
      for (const line of lines) {
        if (line.includes('<user_message')) {
          const match = line.match(/<user_message[^>]*>(.*?)<\/user_message>/);
          if (match) lastUserMsg = match[1];
        } else if (line.includes('<agent_message')) {
          const match = line.match(/<agent_message[^>]*>(.*?)<\/agent_message>/);
          if (match) lastAgentMsg = match[1];
        }
      }

      if (lastUserMsg) {
        let action = 'direct_response';
        let selectedName: string | null = null;
        let instructions: string | null = null;

        if (lastAgentMsg) {
          const nameMatch = lastAgentMsg.match(/\[(SUCCESS|FAILED)\]\s*([^:]+):/);
          if (nameMatch) {
            selectedName = nameMatch[2].trim();
          }
        }

        // Check execution agent logs for recent tool dispatch
        if (!selectedName) {
          try {
            const execFiles = fs
              .readdirSync(path.join(baseDir, 'execution_agents'))
              .filter((f) => f.endsWith('.log'));
            let latestReqTime = 0;
            for (const f of execFiles) {
              const fullP = path.join(baseDir, 'execution_agents', f);
              const stat = fs.statSync(fullP);
              if (stat.mtimeMs > latestReqTime) {
                const fContent = fs.readFileSync(fullP, 'utf8');
                const reqMatches = [...fContent.matchAll(/<agent_request[^>]*>([\s\S]*?)<\/agent_request>/g)];
                if (reqMatches.length > 0) {
                  const lastReq = reqMatches[reqMatches.length - 1];
                  latestReqTime = stat.mtimeMs;
                  selectedName = f.replace(/\.log$/, '').replace(/-/g, '_');
                  instructions = lastReq[1].trim();
                }
              }
            }
          } catch {}
        }

        if (selectedName) {
          const slug = selectedName
            .trim()
            .toLowerCase()
            .replace(/[^a-z0-9]/g, '-')
            .replace(/-+/g, '-')
            .replace(/^-|-$/g, '') || 'agent';
          const agentLogPath = path.join(baseDir, 'execution_agents', `${slug}.log`);
          if (fs.existsSync(agentLogPath)) {
            try {
              const agentLogContent = fs.readFileSync(agentLogPath, 'utf8');
              const reqMatches = [...agentLogContent.matchAll(/<agent_request[^>]*>([\s\S]*?)<\/agent_request>/g)];
              if (reqMatches.length > 1) {
                action = 'reuse';
              } else {
                action = 'create_new';
              }
              if (!instructions && reqMatches.length > 0) {
                instructions = reqMatches[reqMatches.length - 1][1].trim();
              }
            } catch {
              action = 'create_new';
            }
          } else {
            action = 'create_new';
          }
        }

        latestTurn = {
          timestamp: new Date().toISOString(),
          routing_action: action,
          recommended_action: null,
          recommended_agent_id: null,
          selected_agent_id: selectedName ? `baseline-${selectedName}` : null,
          selected_agent_name: selectedName,
          instructions,
          user_message: lastUserMsg,
        };
      }
    }

    return {
      ok: true,
      system: 'baseline',
      roster_count: roster.length,
      roster,
      candidate_ids: candidates.map((c) => c.agent_id),
      candidates,
      latest_turn: latestTurn,
    };
  } catch (err: any) {
    return {
      ok: true,
      system: 'baseline',
      roster_count: 0,
      roster: [],
      candidate_ids: [],
      candidates: [],
      latest_turn: null,
      error: err?.message,
    };
  }
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const system = searchParams.get('system');
  const serverBase = resolveServerBase(system);
  const url = `${serverBase.replace(/\/$/, '')}/api/v1/agents/inspector${system ? `?system=${encodeURIComponent(system)}` : ''}`;

  try {
    const res = await fetch(url, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      cache: 'no-store',
    });

    if (res.status === 404 && system === 'baseline') {
      const fallbackData = readHistoricalBaselineState();
      return new Response(JSON.stringify(fallbackData), {
        status: 200,
        headers: { 'Content-Type': 'application/json; charset=utf-8' },
      });
    }

    const bodyText = await res.text();
    return new Response(bodyText || '{}', {
      status: res.status,
      headers: { 'Content-Type': 'application/json; charset=utf-8' },
    });
  } catch (error: any) {
    if (system === 'baseline') {
      const fallbackData = readHistoricalBaselineState();
      return new Response(JSON.stringify(fallbackData), {
        status: 200,
        headers: { 'Content-Type': 'application/json; charset=utf-8' },
      });
    }
    const message = error?.message || 'Failed to reach Python server';
    return new Response(JSON.stringify({ ok: false, error: message }), {
      status: 502,
      headers: { 'Content-Type': 'application/json; charset=utf-8' },
    });
  }
}
