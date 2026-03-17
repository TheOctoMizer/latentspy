# LatentSpy API Specification

This document provides technical details for consuming the LatentSpy real-time monitoring API. The server uses **Server-Sent Events (SSE)** to stream training metrics and health alerts as they are recorded.

## Base URL
Default: `http://localhost:8000`

---

## 1. Real-Time Stream (SSE)
**Endpoint**: `GET /events`  
**Protocol**: Server-Sent Events (Content-Type: `text/event-stream`)

This endpoint streams all updates from the current active training experiment. If a new experiment begins, a reset event is sent.

### Event Types

#### `new_experiment`
Triggered when a training script starts or a new experiment entry is detected in the database.
- **Payload**:
  ```json
  {
    "id": 12 
  }
  ```
- **Usage**: Frontend should clear existing charts or reset state when this event is received.

#### `metric`
Streams individual metric values (activation norms, rank, patchiness, etc.) for a specific layer.
- **Payload**:
  ```json
  {
    "id": 48961,
    "step": 105,
    "layer_name": "transformer.h.0.mlp",
    "metric_name": "patchiness",
    "value": 0.1245,
    "is_validation": false,
    "timestamp": "2026-03-17T10:52:00.123"
  }
  ```
- **Note**: `id` is the unique database row ID. `step` is the training step.

#### `alert`
Streams health warnings or critical failure alerts detected by the monitor.
- **Payload**:
  ```json
  {
    "id": 501,
    "step": 105,
    "layer_name": "transformer.h.0.mlp",
    "level": "WARNING",
    "message": "Low effective rank detected (1.82 < 2.0)",
    "timestamp": "2026-03-17T10:52:00.456"
  }
  ```
- **Levels**: `WARNING`, `CRITICAL`.

---

## 2. Dashboard UI
**Endpoint**: `GET /`  
**Returns**: `text/html`

Serves the built-in, single-page dashboard. Useful for reference or quick monitoring without a custom frontend.

---

## Consumer Implementation Guide (JavaScript)

To connect to the stream in a frontend application:

```javascript
const eventSource = new EventSource("http://localhost:8000/events");

// Handle metrics
eventSource.addEventListener("metric", (event) => {
    const data = JSON.parse(event.data);
    console.log(`Step ${data.step} | ${data.layer_name} | ${data.metric_name}: ${data.value}`);
    // Update your charts here
});

// Handle health alerts
eventSource.addEventListener("alert", (event) => {
    const data = JSON.parse(event.data);
    alert(`[${data.level}] ${data.layer_name}: ${data.message}`);
});

// Handle experiment resets
eventSource.addEventListener("new_experiment", (event) => {
    const data = JSON.parse(event.data);
    console.log("New Experiment Detected ID:", data.id);
    // Clear local cache/charts
});

// Error handling
eventSource.onerror = (error) => {
    console.error("SSE connection failed:", error);
};
```

### Best Practices for Frontend Devs:
1.  **Buffering**: Metrics can arrive rapidly. Consider throttling chart updates to ~10Hz or only updating every N data points to keep the UI smooth.
2.  **State Management**: Since events are atomic, you'll need to group `metric` events by `layer_name` and `metric_name` to build time-series arrays for plotting.
3.  **Historical Data**: The current SSE stream only sends *new* data from the moment of connection. For loading historical data from the same run, the frontend should query the database directly or we can add a REST endpoint `GET /history/{experiment_id}` if required.
