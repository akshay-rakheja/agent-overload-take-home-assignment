# Dual-Chat with Live Split-View Execution Agent Inspector Design

**Issue:** [#23](https://github.com/akshay-rakheja/agent-overload-take-home-assignment/issues/23)  
**Parent Epic:** [#13](https://github.com/akshay-rakheja/agent-overload-take-home-assignment/issues/13)  
**Status:** Approved  
**Date:** 2026-09-22  

---

## 1. Objective

Enable interactive, side-by-side comparison between **Baseline OpenPoke** (port 8001) and **Enhanced OpenPoke** (port 8002) in the primary web interface (`http://127.0.0.1:3000`). A single user prompt is sent simultaneously to both runtimes. Beneath the dual chat streams, a split-view **Execution Agent Live Inspector** shows:
1. All execution agents listed or created in each system's directory/roster.
2. The exact candidate set provided to the Interaction Agent for that turn (bounded top-5 in Enhanced vs. full roster in Baseline).
3. The real-time routing decision highlighted clearly:
   - 🟢 **Reused Agent:** Green badge, highlighted card, incremented use count.
   - 🔵 **Created New Agent:** Blue badge, newly spawned agent card.
   - ⚪ **Abstained:** Clear indicator when the interaction agent answered directly without dispatching.

---

## 2. Architecture & Data Flow

```
                      [Browser UI: http://127.0.0.1:3000]
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
     POST /api/chat?system=baseline             POST /api/chat?system=enhanced
     GET  /api/chat/history?system=baseline     GET  /api/chat/history?system=enhanced
     GET  /api/agents/inspector?system=baseline GET  /api/agents/inspector?system=enhanced
            │                                                   │
            ▼ (proxies to :8001)                                ▼ (proxies to :8002)
   [Baseline FastAPI :8001]                           [Enhanced FastAPI :8002]
   - Full-roster exposure                             - Bounded working set (top-5)
   - Historical model selection                       - JEV routing (reuse / create / abstain)
   - /api/v1/agents/inspector                         - /api/v1/agents/inspector
```

### 2.1 Multi-Target Next.js Proxy Routes
- **`web/app/api/chat/route.ts`**:
  Accepts query parameter `?system=baseline` or `?system=enhanced`. Routes payload to `http://127.0.0.1:8001/api/v1/chat/send` or `http://127.0.0.1:8002/api/v1/chat/send`.
- **`web/app/api/chat/history/route.ts`**:
  Accepts `?system=baseline` or `?system=enhanced`.
  - `GET`: returns `{ messages: [...] }` from the target system.
  - `DELETE`: if `system` is specified, clears that system; if omitted, clears both backends simultaneously.
- **`web/app/api/agents/inspector/route.ts`**:
  Proxies `GET` requests to `http://127.0.0.1:8001/api/v1/agents/inspector` or `http://127.0.0.1:8002/api/v1/agents/inspector`.

### 2.2 FastAPI Backend Inspector Route (`/api/v1/agents/inspector`)
Added to `server/routes/chat.py` (or a dedicated inspector router mounted in `server/app.py`):
Returns:
```json
{
  "system": "baseline" | "enhanced",
  "roster": [
    {
      "agent_id": "string (UUID)",
      "name": "string",
      "purpose": "string",
      "status": "hot" | "cold" | "dormant",
      "use_count": 2,
      "created_at": "ISO-8601 string",
      "last_used_at": "ISO-8601 string"
    }
  ],
  "candidate_ids": ["uuid1", "uuid2", ...],
  "latest_turn": {
    "routing_action": "reuse" | "create_new" | "abstain",
    "selected_agent_id": "uuid" | null,
    "selected_agent_name": "string" | null,
    "instructions": "string" | null,
    "timestamp": "ISO-8601 string" | null
  }
}
```

---

## 3. UI Component Design

### 3.1 Layout Structure (`web/app/page.tsx`)
- **Header:** Title, Evaluation Lab link, Settings modal trigger, and a unified **"Clear Both"** action.
- **Upper Section: Dual-Column Chat Window:**
  - Left: **Baseline OpenPoke (Port 8001)**
  - Right: **Enhanced OpenPoke (Port 8002)**
  - Both columns feature independent message bubbles, typing indicators, and auto-scroll.
- **Middle Section: Single Unified Message Input:**
  - On submit, dispatches the user's prompt in parallel to both `/api/chat?system=baseline` and `/api/chat?system=enhanced`.
  - Input is disabled until at least one system is ready, with optimistic rendering in both streams.
- **Lower Section: Split-View Live Agent Inspector (`AgentInspectorPanel.tsx`):**
  - Rendered beneath each chat column.
  - **Candidates Section:** Shows agents that were exposed to the interaction agent in the prompt (top-5 in Enhanced, full list in Baseline) in a highlighted bounding box.
  - **Latest Action Banner:**
    - 🟢 `REUSED: [Agent Name]` with incremented count.
    - 🔵 `CREATED NEW: [Agent Name]` with newly registered purpose.
    - ⚪ `ABSTAINED: Answered directly without agent dispatch`.
  - **Full Roster Section:** Collapsible or scrollable list of all registered execution agents with status and metadata.

---

## 4. Testing & Verification

1. **Backend Tests:**
   - Test `/api/v1/agents/inspector` endpoint with mock roster and turn history.
   - Verify candidate filtering and routing action extraction.
2. **Frontend Component Tests:**
   - Unit tests for `AgentInspectorPanel.tsx` covering `reuse`, `create_new`, and `abstain` states.
   - Unit tests for multi-target proxy routes (`/api/chat`, `/api/chat/history`, `/api/agents/inspector`).
3. **End-to-End Live Verification:**
   - Connect to live 8001 and 8002 servers.
   - Send test prompt. Verify dual responses render.
   - Verify execution agents populate in the split inspector in real time.
