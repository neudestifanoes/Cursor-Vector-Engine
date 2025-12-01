import React, { useEffect, useState } from "react";
import "./index.css";

const BACKEND_URL = "http://127.0.0.1:8000";
const WS_URL = "ws://127.0.0.1:8000/ws";

// Helper: map label -> arrow symbol
function directionArrow(label) {
  if (!label) return "•";
  const lower = label.toLowerCase();
  if (lower === "up") return "↑";
  if (lower === "down") return "↓";
  if (lower === "left") return "←";
  if (lower === "right") return "→";
  return "•";
}

function App() {
  const [health, setHealth] = useState(null);
  const [loadingHealth, setLoadingHealth] = useState(true);
  const [healthError, setHealthError] = useState(null);

  const [prediction, setPrediction] = useState(null);
  const [predictLoading, setPredictLoading] = useState(false);
  const [predictError, setPredictError] = useState(null);

  const [wsStatus, setWsStatus] = useState("disconnected");

  // "dashboard" | "stimulus" | "cursor"
  const [view, setView] = useState("dashboard");

  // Cursor position in percentages (0–100 in each axis)
  const [cursorPos, setCursorPos] = useState({ x: 50, y: 50 });

  // --- Helper: update cursor position based on a direction label ---
  const step = 10; // percentage step per prediction

  const moveCursor = (label) => {
    if (!label) return;

    const lower = label.toLowerCase();
    setCursorPos((prev) => {
      let { x, y } = prev;

      if (lower === "up") {
        y = Math.max(0, y - step);
      } else if (lower === "down") {
        y = Math.min(100, y + step);
      } else if (lower === "left") {
        x = Math.max(0, x - step);
      } else if (lower === "right") {
        x = Math.min(100, x + step);
      }

      return { x, y };
    });
  };

  // --- Health check on load ---
  useEffect(() => {
    const fetchHealth = async () => {
      try {
        setLoadingHealth(true);
        const resp = await fetch(`${BACKEND_URL}/health`);
        if (!resp.ok) {
          throw new Error(`HTTP ${resp.status}`);
        }
        const data = await resp.json();
        setHealth(data);
      } catch (err) {
        setHealthError(err.message || "Error fetching health");
      } finally {
        setLoadingHealth(false);
      }
    };

    fetchHealth();
  }, []);

  // --- WebSocket connection for live predictions ---
  useEffect(() => {
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      setWsStatus("connected");
      try {
        ws.send("hello from frontend");
      } catch (_) {}
    };

    ws.onclose = () => {
      setWsStatus("disconnected");
    };

    ws.onerror = () => {
      setWsStatus("error");
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "prediction") {
          const predLabel = msg.predicted_label;
          setPrediction({
            model_name: msg.model_name,
            predicted_label: predLabel,
            class_probabilities: msg.class_probabilities,
          });
          setPredictError(null);

          // update cursor whenever a new prediction comes in
          moveCursor(predLabel);
        }
      } catch (err) {
        console.error("Error parsing WS message:", err);
      }
    };

    return () => {
      ws.close();
    };
  }, []);

  // --- Helper to generate a random test trial (8 x 1792) ---
  const generateRandomTrial = (channels = 8, samples = 1792) => {
    const data = [];
    for (let c = 0; c < channels; c++) {
      const channel = [];
      for (let s = 0; s < samples; s++) {
        channel.push(Math.random() * 2 - 1); // noise [-1, 1]
      }
      data.push(channel);
    }
    return data;
  };

  // --- Call /predict with a random trial ---
  const handleTestPredict = async () => {
    setPredictError(null);
    setPredictLoading(true);

    try {
      const trialData = generateRandomTrial();

      const resp = await fetch(`${BACKEND_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          data: trialData,
          model_name: "lda",
        }),
      });

      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${text}`);
      }

      const result = await resp.json();
      // We also set local state, though WS broadcast usually updates it anyway.
      setPrediction(result);
      moveCursor(result.predicted_label);
    } catch (err) {
      setPredictError(err.message || "Prediction error");
    } finally {
      setPredictLoading(false);
    }
  };

  const handleResetCursor = () => {
    setCursorPos({ x: 50, y: 50 });
  };

  // --- Stimulus View: 4 flicker boxes arranged UP / LEFT / RIGHT / DOWN ---
  const StimulusView = () => {
    return (
      <div>
        <p style={styles.helperText}>
          This screen shows four flickering visual targets. Each box flickers at
          a different frequency and is labeled with a direction. During real
          experiments, the participant fixates on one box (e.g., UP), and the
          SSVEP response is decoded from the EEG.
        </p>

        <div className="stimulus-layout">
          {/* UP: top center, e.g. 10 Hz */}
          <div className="stimulus-box flicker-10hz pos-up">
            <div className="stimulus-label">10 Hz – Up</div>
            <span>UP</span>
          </div>

          {/* RIGHT: center right, e.g. 12 Hz */}
          <div className="stimulus-box flicker-12hz pos-right">
            <div className="stimulus-label">12 Hz – Right</div>
            <span>RIGHT</span>
          </div>

          {/* LEFT: center left, e.g. 15 Hz */}
          <div className="stimulus-box flicker-15hz pos-left">
            <div className="stimulus-label">15 Hz – Left</div>
            <span>LEFT</span>
          </div>

          {/* DOWN: bottom center, e.g. 20 Hz */}
          <div className="stimulus-box flicker-20hz pos-down">
            <div className="stimulus-label">20 Hz – Down</div>
            <span>DOWN</span>
          </div>
        </div>
      </div>
    );
  };

  // --- Dashboard View: status + big central arrow + detailed probs ---
  const DashboardView = () => {
    const currentLabel = prediction?.predicted_label || null;
    const arrow = directionArrow(currentLabel);

    return (
      <>
        {/* System status */}
        <section style={styles.section}>
          <h2 style={styles.sectionTitle}>System Status</h2>

          <div style={styles.statusRow}>
            <div>
              <p style={styles.statusLabel}>Backend:</p>
              {loadingHealth && <p>Checking backend health...</p>}
              {healthError && <p style={styles.error}>Error: {healthError}</p>}
              {health && (
                <div style={styles.healthBox}>
                  <p>
                    <strong>Status:</strong> {health.status}
                  </p>
                  <p>
                    <strong>LDA loaded:</strong>{" "}
                    {health.lda_loaded ? "Yes ✅" : "No ❌"}
                  </p>
                  <p>
                    <strong>SVM loaded:</strong>{" "}
                    {health.svm_loaded ? "Yes ✅" : "No ❌"}
                  </p>
                </div>
              )}
            </div>

            <div style={styles.wsBox}>
              <p style={styles.statusLabel}>WebSocket:</p>
              <p style={styles.wsStatus(wsStatus)}>
                {wsStatus === "connected" && "Connected ✅"}
                {wsStatus === "disconnected" && "Disconnected ⚪"}
                {wsStatus === "error" && "Error ⚠️"}
              </p>
              <p style={styles.helperTextSmall}>
                Listens for live predictions from{" "}
                <code>ws://127.0.0.1:8000/ws</code>.
              </p>
            </div>
          </div>
        </section>

        {/* Central arrow display */}
        <section style={styles.section}>
          <h2 style={styles.sectionTitle}>Live Direction Output</h2>
          <p style={styles.helperText}>
            This arrow reflects the latest decoded direction from the SSVEP
            classifier. In a real experiment, it will update continuously as the
            user focuses on different flickering targets.
          </p>

          <div style={styles.arrowBox}>
            <div style={styles.arrowSymbol}>{arrow}</div>
            <p style={styles.arrowLabel}>
              {currentLabel ? currentLabel.toUpperCase() : "NO PREDICTION YET"}
            </p>
          </div>
        </section>

        {/* Test prediction */}
        <section style={styles.section}>
          <h2 style={styles.sectionTitle}>Test Prediction</h2>
          <p style={styles.helperText}>
            This sends a random 8×1792 trial to <code>/predict</code> using the
            LDA model. Any prediction made by the backend (from here or from
            a streaming script) is also broadcast over WebSocket and displayed
            above via the arrow and below in detail.
          </p>

          <button
            style={styles.button}
            onClick={handleTestPredict}
            disabled={predictLoading}
          >
            {predictLoading ? "Running..." : "Run Test Trial"}
          </button>

          {predictError && <p style={styles.error}>Error: {predictError}</p>}

          {prediction && (
            <div style={styles.predictionBox}>
              <p>
                <strong>Model:</strong> {prediction.model_name}
              </p>
              <p>
                <strong>Predicted Label:</strong>{" "}
                <span style={styles.predictedLabel}>
                  {prediction.predicted_label}
                </span>
              </p>
              <div style={styles.probabilities}>
                <strong>Class Probabilities:</strong>
                <ul>
                  {Object.entries(
                    prediction.class_probabilities || {}
                  ).map(([cls, prob]) => (
                    <li key={cls}>
                      {cls}: {(prob * 100).toFixed(1)}%
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </section>
      </>
    );
  };

  // --- Cursor Demo View: cursor area + controls ---
  const CursorView = () => {
    const currentLabel = prediction?.predicted_label || null;
    const arrow = directionArrow(currentLabel);

    return (
      <>
        <section style={styles.section}>
          <h2 style={styles.sectionTitle}>Cursor Demo</h2>
          <p style={styles.helperText}>
            This view shows a simulated cursor moving in response to decoded
            directions. Right now, directions are generated from synthetic
            trials. With real EEG streaming, the cursor would move based on
            the participant&apos;s SSVEP responses.
          </p>

          <div className="cursor-area">
            <div
              className="cursor-dot"
              style={{
                left: `${cursorPos.x}%`,
                top: `${cursorPos.y}%`,
                transform: "translate(-50%, -50%)",
              }}
            />
          </div>

          <div style={styles.cursorControls}>
            <button
              style={styles.button}
              onClick={handleTestPredict}
              disabled={predictLoading}
            >
              {predictLoading ? "Running..." : "Run Test Trial"}
            </button>
            <button
              style={{ ...styles.button, background: "#6b7280" }}
              onClick={handleResetCursor}
            >
              Reset Cursor
            </button>
          </div>

          <div style={styles.cursorInfo}>
            <p>
              <strong>Last Direction:</strong>{" "}
              {currentLabel ? currentLabel.toUpperCase() : "None"}
            </p>
            <p>
              <strong>Arrow:</strong> {arrow}
            </p>
            <p>
              <strong>Position:</strong>{" "}
              {`x=${cursorPos.x.toFixed(0)}%, y=${cursorPos.y.toFixed(0)}%`}
            </p>
          </div>
        </section>
      </>
    );
  };

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        {/* Top navigation between Dashboard / Stimulus / Cursor */}
        <div style={styles.headerRow}>
          <h1 style={styles.title}>Neural Cursor Control</h1>
          <div style={styles.navPills}>
            <button
              style={{
                ...styles.navPill,
                ...(view === "dashboard" ? styles.navPillActive : {}),
              }}
              onClick={() => setView("dashboard")}
            >
              Dashboard
            </button>
            <button
              style={{
                ...styles.navPill,
                ...(view === "stimulus" ? styles.navPillActive : {}),
              }}
              onClick={() => setView("stimulus")}
            >
              SSVEP Stimulus
            </button>
            <button
              style={{
                ...styles.navPill,
                ...(view === "cursor" ? styles.navPillActive : {}),
              }}
              onClick={() => setView("cursor")}
            >
              Cursor Demo
            </button>
          </div>
        </div>

        {view === "dashboard" ? (
          <DashboardView />
        ) : view === "stimulus" ? (
          <StimulusView />
        ) : (
          <CursorView />
        )}
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background: "#0f172a",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
    color: "#e5e7eb",
    padding: "1.5rem",
  },
  card: {
    maxWidth: "960px",
    width: "100%",
    background: "#020617",
    borderRadius: "16px",
    padding: "24px",
    boxShadow:
      "0 20px 25px -5px rgba(0,0,0,0.4), 0 10px 10px -5px rgba(0,0,0,0.25)",
    border: "1px solid #1e293b",
  },
  headerRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: "1rem",
  },
  title: {
    fontSize: "1.6rem",
  },
  navPills: {
    display: "flex",
    gap: "0.5rem",
  },
  navPill: {
    padding: "0.35rem 0.9rem",
    borderRadius: "999px",
    border: "1px solid #1e293b",
    background: "transparent",
    color: "#e5e7eb",
    cursor: "pointer",
    fontSize: "0.85rem",
  },
  navPillActive: {
    background: "#3b82f6",
    borderColor: "#3b82f6",
  },
  section: {
    marginTop: "1.5rem",
  },
  sectionTitle: {
    fontSize: "1.1rem",
    marginBottom: "0.5rem",
  },
  statusRow: {
    display: "flex",
    gap: "1rem",
    flexWrap: "wrap",
  },
  statusLabel: {
    fontSize: "0.9rem",
    color: "#9ca3af",
    marginBottom: "0.25rem",
  },
  healthBox: {
    background: "#020617",
    borderRadius: "12px",
    padding: "12px",
    border: "1px solid #1e293b",
    minWidth: "220px",
  },
  wsBox: {
    background: "#020617",
    borderRadius: "12px",
    padding: "12px",
    border: "1px solid #1e293b",
    minWidth: "220px",
  },
  wsStatus: (status) => ({
    fontWeight: 600,
    color:
      status === "connected"
        ? "#22c55e"
        : status === "error"
        ? "#f97373"
        : "#e5e7eb",
  }),
  predictionBox: {
    marginTop: "0.75rem",
    background: "#020617",
    borderRadius: "12px",
    padding: "12px",
    border: "1px solid #1e293b",
  },
  predictedLabel: {
    fontSize: "1.1rem",
    color: "#22c55e",
  },
  probabilities: {
    marginTop: "0.5rem",
  },
  button: {
    marginTop: "0.25rem",
    padding: "0.5rem 1rem",
    borderRadius: "999px",
    border: "none",
    cursor: "pointer",
    background: "#3b82f6",
    color: "white",
    fontWeight: 500,
  },
  error: {
    color: "#f97373",
    marginTop: "0.5rem",
  },
  helperText: {
    fontSize: "0.9rem",
    color: "#9ca3af",
    marginBottom: "0.5rem",
  },
  helperTextSmall: {
    fontSize: "0.8rem",
    color: "#9ca3af",
    marginTop: "0.25rem",
  },
  arrowBox: {
    marginTop: "0.75rem",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    padding: "1.5rem 0.5rem",
    borderRadius: "12px",
    border: "1px solid #1e293b",
    background: "#020617",
  },
  arrowSymbol: {
    fontSize: "4rem",
    lineHeight: 1,
    marginBottom: "0.25rem",
  },
  arrowLabel: {
    fontSize: "0.95rem",
    letterSpacing: "0.08em",
    color: "#9ca3af",
  },
  cursorControls: {
    marginTop: "0.75rem",
    display: "flex",
    gap: "0.5rem",
    flexWrap: "wrap",
  },
  cursorInfo: {
    marginTop: "0.5rem",
    fontSize: "0.9rem",
    color: "#9ca3af",
  },
};

export default App;
