package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
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
	insecureNoAuth := flag.Bool("insecure-no-auth", false, "allow serving /v1 without -token-file on a non-loopback api.listen address")
	flag.Parse()

	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	cfg, err := manifest.Load(*manifestPath)
	if err != nil {
		logger.Error("load manifest", "error", err)
		os.Exit(1)
	}

	if err := checkListenAuthentication(cfg.API.Listen, *tokenFile != "", *insecureNoAuth); err != nil {
		logger.Error("refusing to start", "error", err)
		os.Exit(1)
	}
	if *tokenFile == "" && *insecureNoAuth {
		logger.Warn("serving lifecycle routes without authentication", "address", cfg.API.Listen)
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
		if info, err := os.Stat(*tokenFile); err == nil && info.Mode().Perm()&0o077 != 0 {
			logger.Warn("API token file is readable by group or others; restrict it with chmod 600",
				"path", *tokenFile, "mode", info.Mode().Perm().String())
		}
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
	server := newServer(cfg.API.Listen, httpHandler)

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

// newServer bounds every connection phase. All routes answer from memory (load
// and unload return 202 and run asynchronously), so there are no long-lived or
// streaming responses for these timeouts to cut off.
func newServer(listen string, handler http.Handler) *http.Server {
	return &http.Server{
		Addr:              listen,
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      60 * time.Second,
		IdleTimeout:       120 * time.Second,
		MaxHeaderBytes:    64 << 10,
	}
}

// checkListenAuthentication fails closed: lifecycle routes control GPUs and
// containers, so an unauthenticated API may only listen on loopback unless the
// operator explicitly opts out.
func checkListenAuthentication(listen string, tokenConfigured, insecure bool) error {
	if tokenConfigured || insecure || isLoopbackListen(listen) {
		return nil
	}
	return fmt.Errorf("api.listen %q is not loopback; supply -token-file or pass -insecure-no-auth", listen)
}

func isLoopbackListen(listen string) bool {
	host, _, err := net.SplitHostPort(listen)
	if err != nil {
		return false
	}
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
