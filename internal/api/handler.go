package api

import (
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/local-inference-lab/fleet/internal/fleet"
)

type Handler struct {
	manager *fleet.Manager
	logger  *slog.Logger
}

func NewHandler(manager *fleet.Manager, logger *slog.Logger) http.Handler {
	return &Handler{manager: manager, logger: logger}
}

func RequireBearerToken(next http.Handler, token string) http.Handler {
	expected := []byte("Bearer " + token)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && (r.URL.Path == "/healthz" || r.URL.Path == "/readyz") {
			next.ServeHTTP(w, r)
			return
		}
		provided := []byte(r.Header.Get("Authorization"))
		if len(token) == 0 || len(provided) != len(expected) || subtle.ConstantTimeCompare(provided, expected) != 1 {
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Cache-Control", "no-store")
			w.Header().Set("WWW-Authenticate", `Bearer realm="lil-fleet"`)
			writeError(w, http.StatusUnauthorized, "unauthorized", "a valid bearer token is required", nil)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	path := strings.Trim(r.URL.Path, "/")
	parts := strings.Split(path, "/")

	switch {
	case r.Method == http.MethodGet && path == "healthz":
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	case r.Method == http.MethodGet && path == "readyz":
		h.handleReady(w)
	case r.Method == http.MethodGet && path == "v1/models":
		h.handleModels(w)
	case len(parts) == 3 && parts[0] == "v1" && parts[1] == "models" && r.Method == http.MethodGet:
		h.handleModel(w, parts[2])
	case len(parts) == 4 && parts[0] == "v1" && parts[1] == "models" && parts[3] == "load" && r.Method == http.MethodPost:
		h.requireEmptyBody(w, r, func() { h.handleActivate(w, parts[2]) })
	case len(parts) == 4 && parts[0] == "v1" && parts[1] == "models" && parts[3] == "unload" && r.Method == http.MethodPost:
		h.requireEmptyBody(w, r, func() { h.handleUnload(w, parts[2]) })
	case r.Method == http.MethodPost && path == "v1/deployments/switch":
		h.handleSwitch(w, r)
	case r.Method == http.MethodGet && path == "v1/deployments":
		writeJSON(w, http.StatusOK, map[string]any{"deployments": h.manager.Statuses()})
	case len(parts) == 3 && parts[0] == "v1" && parts[1] == "deployments" && r.Method == http.MethodGet:
		h.handleDeployment(w, parts[2])
	case len(parts) == 3 && parts[0] == "v1" && parts[1] == "operations" && r.Method == http.MethodGet:
		h.handleOperation(w, parts[2])
	default:
		writeError(w, http.StatusNotFound, "not_found", "route not found", nil)
	}
}

func (h *Handler) handleReady(w http.ResponseWriter) {
	statuses := h.manager.Statuses()
	for _, status := range statuses {
		if status.Phase == fleet.PhaseUnknown {
			writeError(w, http.StatusServiceUnavailable, "backend_unavailable", "deployment backend is unavailable", nil)
			return
		}
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

func (h *Handler) handleModels(w http.ResponseWriter) {
	type modelResponse struct {
		Model  fleet.ModelInfo `json:"model"`
		Status fleet.Status    `json:"status"`
	}
	models := h.manager.Models()
	result := make([]modelResponse, 0, len(models))
	for _, model := range models {
		status, _ := h.manager.Status(model.ID)
		result = append(result, modelResponse{Model: model, Status: status})
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": result})
}

func (h *Handler) handleModel(w http.ResponseWriter, id string) {
	var info *fleet.ModelInfo
	for _, model := range h.manager.Models() {
		if model.ID == id {
			copy := model
			info = &copy
			break
		}
	}
	status, ok := h.manager.Status(id)
	if info == nil || !ok {
		writeError(w, http.StatusNotFound, "model_not_found", "model is not configured", nil)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"model": info, "status": status})
}

func (h *Handler) handleDeployment(w http.ResponseWriter, id string) {
	status, ok := h.manager.Status(id)
	if !ok {
		writeError(w, http.StatusNotFound, "model_not_found", "model is not configured", nil)
		return
	}
	writeJSON(w, http.StatusOK, status)
}

func (h *Handler) handleSwitch(w http.ResponseWriter, r *http.Request) {
	var request struct {
		ModelID string `json:"model_id"`
	}
	if err := decodeStrict(r, &request); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error(), nil)
		return
	}
	if request.ModelID == "" {
		writeError(w, http.StatusBadRequest, "invalid_request", "model_id is required", nil)
		return
	}
	h.handleActivate(w, request.ModelID)
}

func (h *Handler) handleActivate(w http.ResponseWriter, id string) {
	op, noOp, err := h.manager.Activate(id)
	h.writeLifecycleResult(w, op, noOp, err)
}

func (h *Handler) handleUnload(w http.ResponseWriter, id string) {
	op, noOp, err := h.manager.Unload(id)
	h.writeLifecycleResult(w, op, noOp, err)
}

func (h *Handler) writeLifecycleResult(w http.ResponseWriter, op fleet.Operation, noOp bool, err error) {
	if err != nil {
		var conflict *fleet.ConflictError
		if errors.As(err, &conflict) {
			writeError(w, http.StatusConflict, "operation_in_progress", err.Error(), map[string]any{"operation": conflict.Operation})
			return
		}
		var insufficient *fleet.InsufficientResourcesError
		if errors.As(err, &insufficient) {
			writeError(w, http.StatusConflict, "insufficient_resources", err.Error(), nil)
			return
		}
		writeError(w, http.StatusNotFound, "model_not_found", err.Error(), nil)
		return
	}
	if noOp && op.ID == "" {
		writeJSON(w, http.StatusOK, map[string]any{"changed": false})
		return
	}
	status := http.StatusAccepted
	if noOp {
		status = http.StatusAccepted
	}
	w.Header().Set("Location", "/v1/operations/"+op.ID)
	writeJSON(w, status, map[string]any{"changed": !noOp, "operation": op})
}

func (h *Handler) handleOperation(w http.ResponseWriter, id string) {
	op, ok := h.manager.Operation(id)
	if !ok {
		writeError(w, http.StatusNotFound, "operation_not_found", "operation not found", nil)
		return
	}
	writeJSON(w, http.StatusOK, op)
}

func (h *Handler) requireEmptyBody(w http.ResponseWriter, r *http.Request, next func()) {
	defer r.Body.Close()
	var buf [1]byte
	n, err := r.Body.Read(buf[:])
	if n != 0 || (err != nil && !errors.Is(err, io.EOF)) {
		writeError(w, http.StatusBadRequest, "overrides_not_allowed", "this endpoint accepts no request body; configure deployment overrides in the manifest", nil)
		return
	}
	next()
}

func decodeStrict(r *http.Request, target any) error {
	defer r.Body.Close()
	decoder := json.NewDecoder(io.LimitReader(r.Body, 1<<20))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return fmt.Errorf("decode JSON: %w", err)
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("request must contain exactly one JSON object")
		}
		return fmt.Errorf("decode trailing JSON: %w", err)
	}
	return nil
}

func writeError(w http.ResponseWriter, status int, code, message string, details map[string]any) {
	payload := map[string]any{"error": map[string]any{"code": code, "message": message}}
	if details != nil {
		payload["details"] = details
	}
	writeJSON(w, status, payload)
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
