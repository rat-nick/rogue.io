const Input = (() => {
  let mouseScreenX = 0;
  let mouseScreenY = 0;
  let mouseWorldX = 10000;
  let mouseWorldY = 10000;
  let splitPending = false;
  let ejectPending = false;

  function init() {
    window.addEventListener('mousemove', (e) => {
      mouseScreenX = e.clientX;
      mouseScreenY = e.clientY;
    });

    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space') {
        e.preventDefault();
        splitPending = true;
      }
      if (e.code === 'KeyW') ejectPending = true;
    });

    // Touch support: use touch position as mouse
    window.addEventListener(
      'touchmove',
      (e) => {
        e.preventDefault();
        const t = e.touches[0];
        mouseScreenX = t.clientX;
        mouseScreenY = t.clientY;
      },
      { passive: false }
    );
  }

  function updateMouseWorld() {
    const w = Viewport.screenToWorld(mouseScreenX, mouseScreenY);
    mouseWorldX = w.wx;
    mouseWorldY = w.wy;
  }

  function consumeSplit() {
    const v = splitPending;
    splitPending = false;
    return v;
  }

  function consumeEject() {
    const v = ejectPending;
    ejectPending = false;
    return v;
  }

  return {
    get mouseWorldX() {
      return mouseWorldX;
    },
    get mouseWorldY() {
      return mouseWorldY;
    },
    init,
    updateMouseWorld,
    consumeSplit,
    consumeEject,
  };
})();
