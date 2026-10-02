FROM golang:1.26.8-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/lil-fleet ./cmd/lil-fleet

FROM alpine:3.23
RUN apk add --no-cache ca-certificates docker-cli
COPY --from=build /out/lil-fleet /usr/local/bin/lil-fleet
# Never run as root; compose overrides the uid/gid to match the host.
USER 65532:65532
ENTRYPOINT ["lil-fleet"]
CMD ["-manifest", "/etc/lil-fleet/fleet.json"]
