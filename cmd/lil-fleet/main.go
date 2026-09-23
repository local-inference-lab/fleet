package main

import (
	"context"
	"errors"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/local-inference-lab/fleet/internal/api"
	"github.com/local-inference-lab/fleet/internal/docker"
	"github.com/local-inference-lab/fleet/internal/fleet"
	"github.com/local-inference-lab/fleet/internal/manifest"
)

func main() {
	manifestPath := flag.String("manifest", "fleet.json", "path to the fleet manifest (watched for changes)")
	tokenFile := flag.String("token-file", "", "path to a bearer token file protecting /v1 routes")
	flag.Parse()

	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	cfg, err := manifest.Load(*manifestPath)
	if err != nil {
		logger.Error("load manifest", "error", err)
		os.Exit(1)
	}

	driver := docker.NewCLIDriver(cfg.Runtime.DockerBinary)
	manager := fleet.NewManager(cfg, driver, logger)
	manager.Start()
	defer manager.Close()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go manifest.Watch(ctx, *manifestPath, time.Second, cfg, func(next *manifest.Manifest) error {
		if err := manager.ReloadManifest(next); err != nil {
			return err
		}
		logger.Info("reloaded fleet manifest", "manifest", *manifestPath, "models", len(next.Models))
		return nil
	}, func(err error) {
		logger.Warn("manifest change not applied; retaining last valid configuration", "error", err)
	})

	handler := api.NewHandler(manager, logger)
	var httpHandler http.Handler = handler
	if *tokenFile != "" {
		rawToken, err := os.ReadFile(*tokenFile)
		if err != nil {
			logger.Error("read API token", "error", err)
			os.Exit(1)
		}
		token := strings.TrimSpace(string(rawToken))
		if token == "" {
			logger.Error("read API token", "error", "token file is empty")
			os.Exit(1)
		}
		httpHandler = api.RequireBearerToken(handler, token)
	}
	server := &http.Server{
		Addr:              cfg.API.Listen,
		Handler:           httpHandler,
		ReadHeaderTimeout: 5 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			logger.Error("shutdown HTTP server", "error", err)
		}
	}()

	logger.Info("fleet API listening", "address", cfg.API.Listen, "manifest", *manifestPath)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		logger.Error("serve HTTP", "error", err)
		os.Exit(1)
	}
}
