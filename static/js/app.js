const chatThread = document.getElementById('chatThread');
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatMessage');

const paperForm = document.getElementById('paperForm');
const unitCodeInput = document.getElementById('unitCode');
const yearInput = document.getElementById('year');
const startPaperBtn = document.getElementById('startPaperBtn');
const nextBatchBtn = document.getElementById('nextBatchBtn');
const paperContent = document.getElementById('paperContent');

function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    }[c] || c));
}

function formatAssistantText(text) {
    if (!text) return '';
    let t = text.trim();

    // Heuristic: insert line breaks before enumerated items and options
    // Helps when LLM returns everything in one line
    t = t
        // Newline before question numbers like "1.", "2.", etc.
        .replace(/\s+(?=(\d+\.)\s)/g, '\n')
        // Newline before options like "A.", "B.", "C.", "D."
        .replace(/\s+(?=([A-D]\.)\s)/g, '\n')
        // Newline before "Answer:" or "Answers:"
        .replace(/\s+(?=(Answers?:))/gi, '\n$1');

    // Escape HTML then apply lightweight markdown replacements
    let safe = escapeHtml(t);
    // Bold (markdown-style)
    safe = safe.replace(/\*\*(.+?)\*\*/g, '<strong>$1<\/strong>');
    // Emphasis (optional)
    safe = safe.replace(/\*(.+?)\*/g, '<em>$1<\/em>');
    // Highlight Answers header
    safe = safe.replace(/(^|\n)(Answers?:)/gi, '$1<strong>$2<\/strong>');

    // Paragraphize
    const html = safe
        .split(/\n+/)
        .filter(Boolean)
        .map(l => `<p>${l}</p>`) 
        .join('');
    return html;
}

function appendMessage(role, text) {
    const msg = document.createElement('div');
    msg.className = `message ${role === 'user' ? 'me' : 'them'}`;
    const content = role === 'assistant' ? formatAssistantText(text) : escapeHtml(text);
    msg.innerHTML = `<div class="bubble ${role}">${content}</div>`;
    chatThread.appendChild(msg);
    chatThread.scrollTop = chatThread.scrollHeight;
}

async function apiPost(path, body) {
	const res = await fetch(`${window.API_BASE}${path}`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body)
	});
	if (!res.ok) throw new Error(await res.text());
	return res.json();
}

function renderPaperResponse(html) {
    // Server returns markdown-like text; render minimally and robustly
    // Identify question blocks by "**Question N:**" markers
    const qBlocks = html.match(/\*\*Question\s+\d+:\*\*[\s\S]*?(?=(\*\*Question\s+\d+:\*\*|$))/g) || [];
    paperContent.innerHTML = '';

    qBlocks.forEach(raw => {
        const header = raw.match(/\*\*Question\s+(\d+):\*\*/);
        const qNum = header ? parseInt(header[1], 10) : null;
        const body = raw.replace(/\*\*Question\s+\d+:\*\*/,'').trim();
        const answerMatch = body.match(/\*\*Answer:\*\*[\s\S]*/);
        const cleaned = body.replace(/\*\*Answer:\*\*/,'').trim();

        const container = document.createElement('div');
        container.className = 'question';
        container.innerHTML = `
            <h3>Question ${qNum ?? ''}</h3>
            <div>${formatAssistantText(cleaned)}</div>
            <div class="actions">
                <button class="btn btn-secondary" data-clarify="${qNum}">HELP</button>
            </div>
            <div class="answer">
                <label for="ans-${qNum}">Your answer</label>
                <textarea id="ans-${qNum}"></textarea>
                <div style="margin-top:8px; display:flex; gap:8px;">
                    <button class="btn btn-primary" data-submit="${qNum}">Submit</button>
                </div>
            </div>
        `;
        paperContent.appendChild(container);
    });

    paperContent.querySelectorAll('[data-clarify]').forEach(btn => {
		btn.addEventListener('click', async () => {
			const qn = parseInt(btn.getAttribute('data-clarify'), 10);
			btn.disabled = true;
			try {
				const { response } = await apiPost('/api/pastpaper/clarify', { sessionId: window.SESSION_ID, questionNumber: qn });
                appendMessage('assistant', response);
			} catch (e) {
				appendMessage('assistant', 'Failed to clarify.');
			} finally {
				btn.disabled = false;
			}
		});
	});

	paperContent.querySelectorAll('[data-submit]').forEach(btn => {
		btn.addEventListener('click', async () => {
			const qn = parseInt(btn.getAttribute('data-submit'), 10);
			const textarea = document.getElementById(`ans-${qn}`);
			const answer = textarea.value.trim();
			if (!answer) return;
			btn.disabled = true;
			try {
				const { response } = await apiPost('/api/pastpaper/answer', { sessionId: window.SESSION_ID, questionNumber: qn, answer });
				appendMessage('assistant', response);
			} catch (e) {
				appendMessage('assistant', 'Failed to submit answer.');
			} finally {
				btn.disabled = false;
			}
		});
	});
}

chatForm.addEventListener('submit', async (e) => {
	e.preventDefault();
	const text = chatInput.value.trim();
	if (!text) return;
	appendMessage('user', text);
	chatInput.value = '';
    try {
        // Simple heuristic routing
        if (/\b(start|begin|go through|work through|practice)\b.*(past paper|exam|paper)|\b[A-Z]{3}\d{3}\b.*\b20\d{2}\b|\bpast paper\b/i.test(text)) {
            // Past paper flow
            const { response } = await apiPost('/api/pastpaper/start', { sessionId: window.SESSION_ID, message: text });
            appendMessage('assistant', response);
            nextBatchBtn.disabled = false;
            renderPaperResponse(response);
        } else if (/(^|\b)(quiz|test)(\s+me)?(\s+(on|about))?\b/i.test(text) || /practice quiz|mcq|multiple[- ]choice|generate (a )?quiz|ask me (some )?questions/i.test(text)) {
            // Quiz flow
            const topicMatch = text.match(/(?:quiz|test)\s+me\s+(?:on|about)\s+(.+)/i);
            const topic = topicMatch && topicMatch[1] ? topicMatch[1].trim() : text;
            const { quiz } = await apiPost('/api/quiz', { sessionId: window.SESSION_ID, topic });
            appendMessage('assistant', quiz);
        } else {
            // Default QA
            const { answer } = await apiPost('/api/qa', { sessionId: window.SESSION_ID, message: text });
            appendMessage('assistant', answer);
        }
    } catch (e) {
		appendMessage('assistant', 'Something went wrong.');
	}
});

paperForm.addEventListener('submit', async (e) => {
	e.preventDefault();
	const unitCode = unitCodeInput.value.trim();
	const year = yearInput.value.trim();
	if (!unitCode || !year) return;
	startPaperBtn.disabled = true;
	try {
		const { response } = await apiPost('/api/pastpaper/start', { sessionId: window.SESSION_ID, unitCode, year });
		appendMessage('assistant', response);
		nextBatchBtn.disabled = false;
		renderPaperResponse(response);
	} catch (e) {
		appendMessage('assistant', 'Failed to start past paper.');
	} finally {
		startPaperBtn.disabled = false;
	}
});

nextBatchBtn.addEventListener('click', async () => {
	nextBatchBtn.disabled = true;
	try {
		const { response } = await apiPost('/api/pastpaper/continue', { sessionId: window.SESSION_ID });
		appendMessage('assistant', response);
		renderPaperResponse(response);
		// If final questions, keep disabled
		if (/final questions|completed all questions/i.test(response)) {
			nextBatchBtn.disabled = true;
		} else {
			nextBatchBtn.disabled = false;
		}
	} catch (e) {
		appendMessage('assistant', 'Failed to load next questions.');
		nextBatchBtn.disabled = false;
	}
});

document.getElementById('newSessionBtn').addEventListener('click', () => {
	window.SESSION_ID = crypto.randomUUID();
	chatThread.innerHTML = '';
	paperContent.innerHTML = '';
	nextBatchBtn.disabled = true;
	appendMessage('assistant', 'New session started.');
});
