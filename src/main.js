const { getCurrentWindow } = window.__TAURI__.window;

// ============================================
// Utilities
// ============================================

const parseDuration = (dur) => parseFloat(dur) * (dur.includes('ms') ? 1 : 1000);

// ============================================
// Portal State Machine
// ============================================

const PortalState = {
  COLD: 'cold',   // Queued/disconnected - grid void background
  WARM: 'warm',   // Connecting - hyperspace data stream
  HOT: 'hot',     // Connected - burst flash, then video

  currentState: null,
  element: null,
  container: null,
  backgrounds: {},
  listeners: [],

  init() {
    this.element = document.querySelector('.video-mask');
    this.container = document.querySelector('.video-container');
    this.backgrounds = {
      cold: document.querySelector('.portal-background-cold'),
      warm: document.querySelector('.portal-background-warm'),
      hot: document.querySelector('.portal-background-hot')
    };
    this.setState(this.COLD);
    return this;
  },

  setState(newState) {
    return new Promise((resolve) => {
      if (!this.element) {
        resolve();
        return;
      }

      const previousState = this.currentState;
      this.currentState = newState;

      // Handle state-specific transitions
      if (newState === this.HOT) {
        // Notify listeners immediately at start of transition (before shrink)
        this.listeners.forEach(fn => fn(newState, previousState));

        // Keep warm state active during shrink, then swap everything
        // Don't change mask state or backgrounds yet - keep tunnel visible

        VideoMask.shrinkThenExpand({
          onShrinkComplete: () => {
            // Now that portal is closed, swap state and backgrounds
            this.element.classList.remove('state-cold', 'state-warm', 'state-hot');
            this.element.classList.add(`state-${newState}`);

            Object.entries(this.backgrounds).forEach(([state, el]) => {
              if (el) {
                el.classList.toggle('active', state === newState);
              }
            });
            if (this.backgrounds.hot) {
              this.backgrounds.hot.classList.add('flash');
            }
            this.container?.classList.add('connected');
          }
        }).then(() => {
          resolve();
        });
      } else {
        // Update mask state immediately for non-HOT states
        this.element.classList.remove('state-cold', 'state-warm', 'state-hot');
        this.element.classList.add(`state-${newState}`);
        // Update background visibility immediately for non-HOT states
        Object.entries(this.backgrounds).forEach(([state, el]) => {
          if (el) {
            el.classList.toggle('active', state === newState);
          }
        });

        // Notify listeners
        this.listeners.forEach(fn => fn(newState, previousState));
        this.container?.classList.remove('connected');
        resolve();
      }
    });
  },

  onStateChange(callback) {
    this.listeners.push(callback);
    return () => {
      this.listeners = this.listeners.filter(fn => fn !== callback);
    };
  },

  getState() {
    return this.currentState;
  },

  is(state) {
    return this.currentState === state;
  }
};

window.PortalState = PortalState;

// ============================================
// Video Mask Controller
// ============================================

const VideoMask = {
  element: null,

  init() {
    this.element = document.querySelector('.video-mask');
    return this;
  },

  shrinkThenExpand(options = {}) {
    return new Promise((resolve) => {
      if (!this.element) {
        resolve();
        return;
      }

      const shrinkDuration = options.shrinkDuration || '1.5s';
      const expandDuration = options.expandDuration || '0.4s';
      const targetSize = options.targetSize || '400px';
      const feather = options.feather || '40px';
      const onShrinkComplete = options.onShrinkComplete || (() => {});

      // Phase 1: Shrink aperture closed (tunnel visible through shrinking hole)
      this.element.style.setProperty('--mask-duration', shrinkDuration);
      this.element.classList.add('animating', 'shrinking');

      requestAnimationFrame(() => {
        this.element.style.setProperty('--mask-size', '0px');
        this.element.style.setProperty('--ring-width', '0px');
        this.element.style.setProperty('--ring-height', '0px');
        this.element.classList.remove('expanded');
      });

      setTimeout(() => {
        // Shrink complete - call callback to swap content
        onShrinkComplete();

        // Phase 2: Rapid expansion to reveal video
        this.element.classList.remove('shrinking');
        this.element.style.setProperty('--mask-duration', expandDuration);
        this.element.style.setProperty('--mask-feather', feather);
        this.element.style.setProperty('--mask-aspect', '1');
        this.element.style.setProperty('--ring-width', '160px');
        this.element.style.setProperty('--ring-height', '200px');

        // Trigger expansion
        requestAnimationFrame(() => {
          this.element.style.setProperty('--mask-size', targetSize);
        });

        setTimeout(() => {
          this.element.classList.remove('animating');
          this.element.classList.add('expanded');
          resolve();
        }, parseDuration(expandDuration));
      }, parseDuration(shrinkDuration));
    });
  },

  setImmediate(expanded, size = '200px') {
    if (!this.element) return this;

    this.element.classList.remove('animating');
    if (expanded) {
      this.element.style.setProperty('--mask-size', size);
      this.element.classList.add('expanded');
    } else {
      this.element.style.setProperty('--mask-size', '0px');
      this.element.classList.remove('expanded');
    }
    return this;
  }
};

window.VideoMask = VideoMask;

// ============================================
// Terminal Status Controller
// ============================================

const TerminalStatus = {
  element: null,
  textElement: null,
  typewriterTimeout: null,

  messages: {
    cold: 'ENTER KEY:',
    warm: 'ESTABLISHING LINK...',
    hot: 'CONNECTED'
  },

  init() {
    this.element = document.getElementById('terminal-status');
    this.textElement = document.getElementById('terminal-text');

    // Set initial state
    this.setState('cold');

    // Listen for portal state changes
    PortalState.onStateChange((newState) => {
      this.setState(newState);
    });

    return this;
  },

  setState(state) {
    if (!this.element || !this.textElement) return;

    // Update state class for colors
    this.element.classList.remove('state-cold', 'state-warm', 'state-hot');
    this.element.classList.add(`state-${state}`);

    // Type out the new message
    const message = this.messages[state] || '';
    this.typeMessage(message);
  },

  typeMessage(message, speed = 30) {
    if (!this.textElement) return;

    // Clear any existing typewriter
    if (this.typewriterTimeout) {
      clearTimeout(this.typewriterTimeout);
    }

    this.textElement.classList.add('typing');
    this.textElement.textContent = '';

    let i = 0;
    const type = () => {
      if (i < message.length) {
        this.textElement.textContent += message.charAt(i);
        i++;
        this.typewriterTimeout = setTimeout(type, speed);
      } else {
        this.textElement.classList.remove('typing');
      }
    };

    type();
  },

  setMessage(message) {
    this.typeMessage(message);
  }
};

window.TerminalStatus = TerminalStatus;

// ============================================
// Terminal Input Auto-resize
// ============================================

function setupTerminalInput() {
  const input = document.getElementById('terminal-input');
  if (!input) return;

  // Create a hidden span to measure text width
  const measureSpan = document.createElement('span');
  measureSpan.style.cssText = `
    position: absolute;
    visibility: hidden;
    white-space: pre;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  `;
  document.body.appendChild(measureSpan);

  function resizeInput() {
    measureSpan.textContent = input.value || '';
    // Add small buffer for caret, minimum 1ch
    const width = Math.max(measureSpan.offsetWidth + 12, 10);
    input.style.width = `${width}px`;
  }

  input.addEventListener('input', resizeInput);
  resizeInput(); // Initial size
}

// ============================================
// Window Controls
// ============================================

function setupWindowControls() {
  const appWindow = getCurrentWindow();
  const minBtn = document.getElementById('titlebar-minimize');
  const maxBtn = document.getElementById('titlebar-maximize');
  const closeBtn = document.getElementById('titlebar-close');

  minBtn?.addEventListener('click', () => appWindow.minimize());

  maxBtn?.addEventListener('click', async () => {
    if (await appWindow.isMaximized()) {
      appWindow.unmaximize();
    } else {
      appWindow.maximize();
    }
  });

  closeBtn?.addEventListener('click', () => appWindow.close());
}

// ============================================
// Settings Panel
// ============================================

function setupPanels() {
  const appWindow = getCurrentWindow();
  const settingsPanel = document.getElementById('settings-panel');
  const settingsToggle = document.getElementById('settings-toggle');
  const settingsClose = document.getElementById('settings-close');
  const video = document.querySelector('.video-container video');

  function toggleSettingsPanel() {
    const isOpening = !settingsPanel?.classList.contains('open');
    settingsPanel?.classList.toggle('open');

    // Pause/resume video when settings panel toggles
    if (video) {
      if (isOpening) {
        video.pause();
      } else {
        video.play();
      }
    }
  }

  settingsToggle?.addEventListener('click', toggleSettingsPanel);
  settingsClose?.addEventListener('click', toggleSettingsPanel);

  // Keyboard panel
  const keyboardPanel = document.getElementById('keyboard-panel');
  const keyboardToggle = document.getElementById('keyboard-toggle');

  const baseHeight = 500;
  const keyboardPanelHeight = 120;

  async function toggleKeyboardPanel() {
    const isOpen = keyboardPanel?.classList.contains('open');
    const size = await appWindow.innerSize();
    const width = size.width / window.devicePixelRatio;

    if (isOpen) {
      keyboardPanel?.classList.remove('open');
      setTimeout(async () => {
        await appWindow.setSize(new window.__TAURI__.dpi.LogicalSize(width, baseHeight));
      }, 300);
    } else {
      await appWindow.setSize(new window.__TAURI__.dpi.LogicalSize(width, baseHeight + keyboardPanelHeight));
      keyboardPanel?.classList.add('open');
    }
  }

  keyboardToggle?.addEventListener('click', toggleKeyboardPanel);
}

// ============================================
// Play Button
// ============================================

function setupPlayButton() {
  const playBtn = document.getElementById('play-btn');
  const logoContainer = document.getElementById('logo-container');
  const terminalInput = document.getElementById('terminal-input');

  playBtn?.addEventListener('click', async () => {
    // Step through states one at a time for testing
    const currentState = PortalState.getState();

    if (currentState === PortalState.COLD) {
      // Clear the terminal input when leaving cold state
      if (terminalInput) terminalInput.value = '';
      await PortalState.setState(PortalState.WARM);
    } else if (currentState === PortalState.WARM) {
      await PortalState.setState(PortalState.HOT);
      logoContainer?.classList.add('hidden');
    }
    // In HOT state, button does nothing (or could reset to COLD if needed)
  });
}

// ============================================
// Initialization
// ============================================

window.addEventListener("DOMContentLoaded", () => {
  setupWindowControls();
  setupPanels();
  setupPlayButton();
  setupTerminalInput();
  VideoMask.init();
  PortalState.init();
  TerminalStatus.init();
});
