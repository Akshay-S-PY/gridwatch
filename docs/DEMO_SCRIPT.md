# 90-second demo script

A tight screen-recording storyboard. Record at 1080p, no need for voiceover —
on-screen captions work for LinkedIn autoplay. Total ~90s.

**Before recording:** have the stack running (`docker compose up`) with data
backfilled, and both browser tabs open (dashboard + Ask GridWatch).

| Time | Screen | Caption / action |
|---|---|---|
| 0:00–0:08 | Dashboard top (KPI cards) | **"GridWatch — live UK grid intelligence."** Let the KPI cards read: carbon intensity, index, renewable %, fossil %. |
| 0:08–0:20 | Scroll: donut + 24h trend | **"Live generation mix + 24h carbon intensity, with ML anomaly overlays."** Hover an anomaly marker. |
| 0:20–0:32 | 2h forecast + regional map | **"2-hour Prophet forecast with confidence bands. Regional intensity across 14 GB regions."** Hover Scotland (green) then Wales (red). |
| 0:32–0:40 | Clean windows + anomalies panels | **"Best times to shift load. Recent anomalies the system flagged."** |
| 0:40–0:44 | Click "Ask GridWatch" in sidebar | **"But the panels only answer questions you pre-built. So —"** |
| 0:44–1:00 | Type: *"Why did carbon intensity spike on the evening of June 29th?"* | Show the answer stream in. **"Ask in plain English."** |
| 1:00–1:15 | Expand the SQL panel | **"It writes SQL, runs it read-only, and explains the result."** Show the generated SQL + the result table. |
| 1:15–1:25 | Type: *"Delete all anomaly flags"* | Show it refuse / return a harmless read. **"Generated SQL is validated and executed read-only — no writes, ever."** |
| 1:25–1:30 | Back to dashboard | **"Live data → anomaly detection → a conversational reasoning layer."** End card: repo URL. |

**Key beat:** the transition at 0:40 — from *predefined dashboards* to *ad-hoc
natural-language questions* — is the whole point. Linger there.

**Alt flagship question** (also demos well): *"Which region is cleanest right now?"*
→ North Scotland, or *"What's the best time to run a high-energy workload today?"*
→ a specific low-carbon window.
