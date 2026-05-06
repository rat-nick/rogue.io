const Leaderboard = (() => {
  let container = null;

  function init(el) {
    container = el;
  }

  function update() {
    if (!container) return;
    const entries = State.leaderboard;
    if (!entries || entries.length === 0) return;

    const ownId = State.playerId;
    let html = '<div class="lb-title">Leaderboard</div>';
    const top = entries.slice(0, 10);
    for (let i = 0; i < top.length; i++) {
      const [name, score] = top[i];
      // Highlight own name
      // We compare by name since we don't have playerId in leaderboard entries
      const isOwn = name === State.getName(ownId);
      html += `<div class="lb-row${isOwn ? ' lb-own' : ''}">
        <span class="lb-rank">${i + 1}</span>
        <span class="lb-name">${escapeHtml(name)}</span>
        <span class="lb-score">${Math.floor(score)}</span>
      </div>`;
    }
    container.innerHTML = html;
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  return { init, update };
})();
