package manifest

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// Watch reloads path at interval and applies each distinct valid manifest.
// Invalid or temporarily inapplicable changes leave the last applied manifest
// in place and are retried until the file or runtime state changes.
func Watch(
	ctx context.Context,
	path string,
	interval time.Duration,
	initial *Manifest,
	apply func(*Manifest) error,
	reportError func(error),
) {
	if interval <= 0 {
		panic("manifest watch interval must be positive")
	}

	applied := manifestDigest(initial)
	lastError := ""
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			next, err := Load(path)
			if err == nil && manifestDigest(next) != applied {
				err = apply(next)
				if err == nil {
					applied = manifestDigest(next)
				}
			}
			if err != nil {
				message := err.Error()
				if message != lastError && reportError != nil {
					reportError(fmt.Errorf("reload manifest: %w", err))
				}
				lastError = message
				continue
			}
			lastError = ""
		}
	}
}

func manifestDigest(value *Manifest) [sha256.Size]byte {
	encoded, err := json.Marshal(value)
	if err != nil {
		panic(errors.New("manifest contains a value that cannot be encoded as JSON"))
	}
	return sha256.Sum256(encoded)
}
