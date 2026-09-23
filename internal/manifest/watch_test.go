package manifest

import (
	"context"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

func TestWatchAppliesValidChangesAndRetainsLastGoodManifest(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fleet.json")
	writeWatchManifest(t, path, "alpha")
	initial, err := Load(path)
	if err != nil {
		t.Fatalf("Load(initial) error = %v", err)
	}

	applied := make(chan string, 2)
	errors := make(chan error, 2)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go Watch(ctx, path, time.Millisecond, initial, func(next *Manifest) error {
		applied <- next.Models[0].ID
		return nil
	}, func(err error) {
		errors <- err
	})

	if err := os.WriteFile(path, []byte(`{"version":1,"models":[]}`), 0o600); err != nil {
		t.Fatalf("write invalid manifest: %v", err)
	}
	select {
	case <-errors:
	case <-time.After(time.Second):
		t.Fatal("watcher did not report invalid manifest")
	}
	select {
	case model := <-applied:
		t.Fatalf("invalid manifest applied model %q", model)
	case <-time.After(10 * time.Millisecond):
	}

	writeWatchManifest(t, path, "beta")
	select {
	case model := <-applied:
		if model != "beta" {
			t.Fatalf("applied model = %q, want beta", model)
		}
	case <-time.After(time.Second):
		t.Fatal("watcher did not apply valid manifest change")
	}
}

func TestWatchRetriesRejectedManifest(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fleet.json")
	writeWatchManifest(t, path, "alpha")
	initial, err := Load(path)
	if err != nil {
		t.Fatalf("Load(initial) error = %v", err)
	}

	attempts := make(chan struct{}, 2)
	var attemptCount atomic.Int32
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go Watch(ctx, path, time.Millisecond, initial, func(*Manifest) error {
		count := attemptCount.Add(1)
		attempts <- struct{}{}
		if count == 1 {
			return context.DeadlineExceeded
		}
		return nil
	}, nil)

	writeWatchManifest(t, path, "beta")
	for count := 0; count < 2; count++ {
		select {
		case <-attempts:
		case <-time.After(time.Second):
			t.Fatal("watcher did not retry rejected manifest")
		}
	}
}

func writeWatchManifest(t *testing.T, path, modelID string) {
	t.Helper()
	body := `{"version":1,"runtime":{"poll_interval":"1ms"},"models":[{"id":"` + modelID + `","image":"image"}]}`
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
}
