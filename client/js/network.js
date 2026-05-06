const Network = (() => {
  // Message type constants (must match server/protocol.py)
  const MSG_INIT = 0x10;
  const MSG_TICK = 0x11;
  const MSG_DEAD = 0x12;

  let ws = null;
  let connected = false;
  let playerName = '';
  let reconnectDelay = 500;
  let onDeadCallback = null;

  function connect(host, name, onDead) {
    playerName = name;
    onDeadCallback = onDead;
    _open(host);
  }

  function _open(host) {
    try {
      ws = new WebSocket(host);
    } catch (e) {
      console.error('WS open error', e);
      _scheduleReconnect(host);
      return;
    }
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      connected = true;
      reconnectDelay = 500;
      console.log('Connected to server');
      // Send player name as first message
      ws.send(playerName);
    };

    ws.onmessage = (evt) => {
      const data = evt.data;
      if (!(data instanceof ArrayBuffer)) return;
      try {
        const msg = msgpack.decode(new Uint8Array(data));
        const type = msg[0];
        if (type === MSG_INIT) State.init(msg);
        else if (type === MSG_TICK) State.applyTick(msg);
        else if (type === MSG_DEAD) {
          if (onDeadCallback) onDeadCallback(msg[1], msg[2]);
        }
      } catch (e) {
        console.warn('Failed to decode message', e);
      }
    };

    ws.onclose = () => {
      connected = false;
      _scheduleReconnect(host);
    };

    ws.onerror = (e) => {
      console.error('WebSocket error', e);
    };
  }

  function _scheduleReconnect(host) {
    console.log(`Reconnecting in ${reconnectDelay}ms...`);
    setTimeout(() => _open(host), reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, 10000);
  }

  // Client -> Server: 10-byte binary input packet
  // Format: !BffB  (1 + 4 + 4 + 1 = 10 bytes, big-endian)
  const _inputBuf = new ArrayBuffer(10);
  const _inputView = new DataView(_inputBuf);

  function sendInput(worldX, worldY, split, eject) {
    if (!connected || !ws || ws.readyState !== WebSocket.OPEN) return;
    _inputView.setUint8(0, 0x01); // msg_type
    _inputView.setFloat32(1, worldX, false); // big-endian
    _inputView.setFloat32(5, worldY, false);
    const flags = (split ? 0x01 : 0) | (eject ? 0x02 : 0);
    _inputView.setUint8(9, flags);
    ws.send(_inputBuf);
  }

  function disconnect() {
    if (ws) {
      ws.onclose = null;
      ws.close();
      ws = null;
    }
    connected = false;
  }

  return {
    get connected() {
      return connected;
    },
    connect,
    sendInput,
    disconnect,
  };
})();
