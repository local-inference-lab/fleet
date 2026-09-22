.PHONY: build test vet check

GOCACHE ?= /tmp/lil-fleet-go-cache

build:
	GOCACHE=$(GOCACHE) go build ./cmd/lil-fleet

test:
	GOCACHE=$(GOCACHE) go test -race ./...

vet:
	GOCACHE=$(GOCACHE) go vet ./...

check: test vet
